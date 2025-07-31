from abc import ABC, abstractmethod
from optyx.core.channel import Diagram
from cotengra import (
    ReusableHyperOptimizer,
    HyperOptimizer,
    HyperCompressedOptimizer,
    ReusableHyperCompressedOptimizer
)
from discopy.tensor import Tensor
import numpy as np
import perceval as pcvl
from dataclasses import dataclass
from optyx.core.channel import Ty, mode, bit

@dataclass(frozen=True)
class EvalResult:
    """
    Class to encapsulate the result of an evaluation.
    """
    result_tensor: Tensor
    types: Ty

    def get_tensor(self):
        """
        Get the result tensor.

        Returns:
            Tensor: The result tensor.
        """
        return self.result_tensor

    def prob_dist(self):
        """
        Get the probability distribution from the result tensor.

        Returns:
            dict: A dictionary mapping occupation configurations to probabilities.
        """
        assert len(self.result_tensor.dom) == 0, "Result tensor must represent a state without inputs."
        assert any(t in {bit, mode} for t in self.types), "Types must contain at least one 'bit' or 'mode'."

        return self._convert_array_to_prob_dist(self.result_tensor.array)

    def prob(self, occupation: tuple):
        """
        Get the probability of a specific occupation configuration.

        Args:
            occupation: The occupation configuration to query.

        Returns:
            float: The probability of the specified occupation configuration.
        """
        prob_dist = self.prob_dist()
        return prob_dist.get(occupation, 0.0)

    def _convert_array_to_prob_dist(self):
        """
        Convert a result array to a probability distribution.

        Args:
            result: The array to convert.

        Returns:
            A list of tuples of the form (occupation configuration, probability).
        """
        dictionary = {idx: val for idx, val in np.ndenumerate(self.result_tensor.array) if val != 0}

        # need to calculate a marginal prob_dist


        return


class AbstractBackend(ABC):
    """
    Abstract base class for backend implementations.
    All backends must implement the `eval` method.
    """

    def __init__(self):
        """
        Initialize the backend.
        This method can be overridden by subclasses to set up specific configurations.
        """
        self._quimb_tensor = None
        self._discopy_tensor = None
        self._matrix = None

    def _get_matrix(
        self,
        diagram: Diagram,
        cache: bool = True
    ):
        """
        Get the matrix representation of the diagram.

        Args:
            diagram (Diagram): The diagram to convert.
            cache (bool): Whether to cache the result.

        Returns:
            np.ndarray: The matrix representation of the diagram.
        """
        try:
            matrix = diagram.to_path().array
            if cache:
                self._matrix = matrix
            return matrix
        except NotImplementedError as e:
            raise NotImplementedError(
                "The diagram cannot be converted to a matrix. It is not linear optical."
            ) from e

    def _get_quimb_tensor(
        self,
        diagram: Diagram,
        cache: bool = True
    ):
        quimb_tensor = self._get_discopy_tensor(diagram, cache).to_quimb()
        if cache:
            self._quimb_tensor = quimb_tensor

        return quimb_tensor

    def _umatrix_to_perceval_circuit(self, matrix) -> pcvl.Circuit:
        m = pcvl.Matrix(matrix.T)
        return pcvl.components.Unitary(U=m)

    def _get_discopy_tensor(
        self,
        diagram: Diagram,
        cache: bool = True
    ):
        if diagram.is_pure():
            discopy_tensor = diagram.get_kraus().to_tensor()
        else:
            discopy_tensor = diagram.to_tensor()
        if cache:
            self._discopy_tensor = discopy_tensor

        return discopy_tensor

    @abstractmethod
    def eval(self, *args, **kwargs) -> EvalResult:
        """
        Evaluate the backend with the given arguments.

        Args:
            *args: Positional arguments for evaluation.
            **kwargs: Keyword arguments for evaluation.

        Returns:
            The result of the evaluation.
        """
        pass


class QuimbBackend(AbstractBackend):
    """
    Backend implementation using Quimb.
    """

    def __init__(self, hyperoptimiser=None, contraction_params=None):
        """
        Initialize the Quimb backend.

        Args:
            hyperoptimiser: An optional hyperoptimiser for contraction optimization.
            contraction_params: Optional parameters for the contraction.
        """
        self.hyperoptimiser = hyperoptimiser
        self.contraction_params = contraction_params or {}
        self._quimb_tensor = None
        self._discopy_tensor = None

    def eval(self, diagram):
        """
        Evaluate the diagram using Quimb.

        Args:
            diagram (Diagram): The diagram to evaluate.

        Returns:
            The result of the evaluation.
        """
        quimb_tn = self._get_quimb_tensor(diagram)

        if self.hyperoptimiser is None:
            results=quimb_tn^...
        else:
            if isinstance(
                self.hyperoptimiser,
                (ReusableHyperOptimizer, HyperOptimizer)
            ):
                results = quimb_tn.contract(
                    optimize=self.hyperoptimiser,
                    **self.contraction_params
                )
            elif isinstance(
                self.hyperoptimiser,
                (ReusableHyperCompressedOptimizer, HyperCompressedOptimizer)
            ):
                results = quimb_tn.contract_compressed(
                    optimize=self.hyperoptimiser,
                    **self.contraction_params
                )
            else:
                raise ValueError(
                    "Unsupported hyperoptimiser type. Use ReusableHyperOptimizer, HyperOptimizer, ReusableHyperCompressedOptimizer, or HyperCompressedOptimizer."
                )

        if not isinstance(results, (complex, float, int)):
            results = results.data

        return EvalResult(
            Tensor(
                "Result",
                dom=self._discopy_tensor.dom,
                cod=self._discopy_tensor.cod,
                array=results
            ),
            types=diagram.cod
        )


class DiscopyBackend(AbstractBackend):
    """
    Backend implementation using Discopy.
    """

    def eval(self, diagram):
        """
        Evaluate the diagram using Discopy.

        Args:
            diagram (Diagram): The diagram to evaluate.

        Returns:
            The result of the evaluation.
        """
        discopy_tensor = self._get_discopy_tensor(diagram).eval()
        return EvalResult(discopy_tensor)


class PercevalBackend(AbstractBackend):
    """
    Backend implementation using Perceval.
    """

    def __init__(self, perceval_backend=None):
        """
        Initialize the Perceval backend.

        Args:
            perceval_backend: An optional Perceval backend configuration.
        """
        if perceval_backend is None:
            self.perceval_backend = pcvl.BackendFactory.get_backend("SLOS")
        else:
            self.perceval_backend = perceval_backend

    def eval(self, diagram, perceval_state):
        """
        Evaluate the diagram using Perceval. Works only for unitary operations.

        Args:
            diagram (Diagram): The diagram to evaluate.

        Returns:
            The result of the evaluation.
        """

        sim = pcvl.Simulator(self.perceval_backend)
        matrix = self._get_matrix(diagram)

        if not np.allclose(
            np.eye(matrix.shape[0]),
            matrix.dot(matrix.conj().T)
        ):
            raise ValueError("The provided diagram does not represent a unitary operation.")

        perceval_circuit = self._umatrix_to_perceval_circuit(matrix)
        sim.set_circuit(perceval_circuit)
        result = sim.probs(perceval_state)
        result = {tuple(k): v for k, v in result.items()}

        array = np.zeros(
            *[int(i) for i in self._discopy_tensor.dom],
            *[int(i) for i in self._discopy_tensor.cod]
        )

        configs, coeffs = map(np.array, zip(*result))
        idx = tuple(configs.T)
        array[idx] = coeffs

        return EvalResult(
            Tensor(
                "Result",
                dom=self._discopy_tensor.dom,
                cod=self._discopy_tensor.cod,
                array=array
            )
        )