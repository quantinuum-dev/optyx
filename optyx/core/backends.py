from abc import ABC, abstractmethod
from optyx.core.channel import Diagram
from cotengra import (
    ReusableHyperOptimizer,
    HyperOptimizer,
    HyperCompressedOptimizer,
    ReusableHyperCompressedOptimizer,
)
from discopy import tensor
import numpy as np
import perceval as pcvl
from dataclasses import dataclass
from optyx.core.channel import Ty, mode, bit
from quimb.tensor import TensorNetwork

@dataclass(frozen=True)
class EvalResult:
    """
    Class to encapsulate the result of an evaluation.
    """
    result_tensor: tensor.Box
    output_types: Ty
    state_type: str # "amps" for amplitudes, "dm" for density matrix, "prob" for probability distribution

    @property
    def tensor(self) -> tensor.Box:
        """
        Get the resulting tensor.

        Returns:
            tensor.Box: The result tensor.
        """
        return self.result_tensor

    @property
    def density_matrix(self) -> tensor.Box:
        if len(self.result_tensor.dom) != 0:
            raise ValueError("Result tensor must represent a state without inputs.")
        if self.state_type not in ["dm", "amps"]:
            raise TypeError("Cannot get density matrix from probability distribution.")
        if self.state_type == "amps":
            density_matrix = self.result_tensor.dagger() >> self.result_tensor
            return density_matrix
        return self.result_tensor

    @property
    def amplitudes(self) -> dict:
        if self.state_type != "amps":
            raise TypeError("Cannot get amplitudes from density matrix or probability distribution.")
        return self._convert_array_to_dict(self.result_tensor.array)

    def prob_dist(self, round_digits: int = None) -> dict:
        """
        Get the probability distribution from the result tensor.

        Returns:
            dict: A dictionary mapping occupation configurations to probabilities.
        """
        if len(self.result_tensor.dom) != 0:
            raise ValueError("Result tensor must represent a state without inputs.")
        if self.state_type == "amps":
            return self._prob_dist_pure(round_digits)
        elif self.state_type == "dm":
            return self._prob_dist_mixed(round_digits)
        elif self.state_type == "prob":
            return self._convert_array_to_dict(self.result_tensor.array, round_digits=round_digits)
        else:
            raise ValueError("Unsupported state_type type. Must be 'amps', 'dm', or 'prob'.")

    def prob(self, occupation: tuple) -> float:
        """
        Get the probability of a specific occupation configuration.

        Args:
            occupation: The occupation configuration to query.

        Returns:
            float: The probability of the specified occupation configuration.
        """
        prob_dist = self.prob_dist()
        return prob_dist.get(occupation, 0.0)

    def _convert_array_to_dict(
            self,
            array,
            round_digits: int = None
        ) -> dict:
        """
        Convert a result array to a probability distribution.

        Args:
            result: The array to convert.

        Returns:
            A list of tuples of the form (occupation configuration, probability).
        """
        if round_digits is not None:
            array = np.round(array, decimals=round_digits)
        return {idx: val for idx, val in np.ndenumerate(array) if val != 0}

    def _prob_dist_pure(self, round_digits: int = None) -> dict:
        """
        Get the probability distribution for a pure state.

        Returns:
            dict: A dictionary mapping occupation configurations to probabilities.
        """

        values = self._convert_array_to_dict(self.result_tensor.array, round_digits=round_digits)
        return {key: abs(value) ** 2 for key, value in values.items()}

    def _prob_dist_mixed(self, round_digits: int = None) -> dict:
        if not any(t in {bit, mode} for t in self.output_types):
            raise ValueError("Types must contain at least one 'bit' or 'mode'.")

        values = self._convert_array_to_dict(self.result_tensor.array, round_digits=round_digits)

        # for all measured wires, get all the occupation configurations
        # for each of the above get the values for the unmeasured wires
        # for each of the above, sum up the values if the occs agree on doubled wires

        mask = [[1] if t in {bit, mode} else [0, 0] for t in self.output_types]
        mask_flat = np.concatenate([np.atleast_1d(m) for m in mask])

        occs_measured_wires = set()
        for key, _ in values.items():
            occ = tuple(
                i for i, m in zip(key, mask_flat) if m != 0
            )
            occs_measured_wires.add(occ)

        values_aggregated = {key: [] for key in occs_measured_wires}
        for key, _ in values.items():
            occ_measured = tuple(
                i for i, m in zip(key, mask_flat) if m != 0
            )
            occs_unmeasured = tuple(
                i for i, m in zip(key, mask_flat) if m == 0
            )
            if all(occs_unmeasured[i] == occs_unmeasured[i + 1] for i in range(0, len(occs_unmeasured) - 1, 2)):
                values_aggregated[occ_measured].append(values[key])

        prob_dist = {key: sum(values) for key, values in values_aggregated.items()}

        return prob_dist


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

    def _get_matrix(
        self,
        diagram: Diagram
    ) -> np.ndarray:
        """
        Get the matrix representation of the diagram.

        Args:
            diagram (Diagram): The diagram to convert.

        Returns:
            np.ndarray: The matrix representation of the diagram.
        """
        try:
            return diagram.to_path().array
        except NotImplementedError as e:
            raise NotImplementedError(
                "The diagram cannot be converted to a matrix. It is not linear optical."
            ) from e

    def _get_quimb_tensor(
        self,
        diagram: Diagram
    ) -> TensorNetwork:
        return self._get_discopy_tensor(diagram).to_quimb()

    def _umatrix_to_perceval_circuit(self, matrix) -> pcvl.Circuit:
        m = pcvl.Matrix(matrix.T)
        return pcvl.components.Unitary(U=m)

    def _get_discopy_tensor(
        self,
        diagram: Diagram
    ) -> tensor.Diagram:
        if diagram.is_pure:
            return diagram.get_kraus().to_tensor()
        else:
            return diagram.double().to_tensor()

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

    def __init__(
            self,
            hyperoptimiser: HyperOptimizer = None,
            contraction_params: dict = None
        ):
        """
        Initialize the Quimb backend.

        Args:
            hyperoptimiser: An optional hyperoptimiser for contraction optimization.
            contraction_params: Optional parameters for the contraction.
        """
        self.hyperoptimiser = hyperoptimiser
        self.contraction_params = contraction_params or {}

    def eval(self, diagram : Diagram) -> EvalResult:
        """
        Evaluate the diagram using Quimb.

        Args:
            diagram (Diagram): The diagram to evaluate.

        Returns:
            The result of the evaluation.
        """
        quimb_tn = self._get_quimb_tensor(diagram)
        discopy_tensor = self._get_discopy_tensor(diagram)

        is_approx = isinstance(
            self.hyperoptimiser,
            (ReusableHyperCompressedOptimizer, HyperCompressedOptimizer)
        )

        is_exact = isinstance(
            self.hyperoptimiser,
            (ReusableHyperOptimizer, HyperOptimizer)
        )

        if self.hyperoptimiser is None:
            results=quimb_tn^...
        else:
            if not is_approx and not is_exact:
                raise ValueError(
                    "Unsupported hyperoptimiser type. Use ReusableHyperOptimizer, HyperOptimizer, ReusableHyperCompressedOptimizer, or HyperCompressedOptimizer."
                )

            contract = quimb_tn.contract_compressed if is_approx else quimb_tn.contract
            results = contract(
                optimize=self.hyperoptimiser,
                output_inds=sorted(quimb_tn.outer_inds()),
                **self.contraction_params
            )

        if not isinstance(results, (complex, float, int)):
            results = results.data

        if diagram.is_pure:
            state_type = "amps"
        else:
            state_type = "dm"

        return EvalResult(
            tensor.Box(
                "Result",
                discopy_tensor.dom,
                discopy_tensor.cod,
                results
            ),
            output_types=diagram.cod,
            state_type=state_type
        )


class DiscopyBackend(AbstractBackend):
    """
    Backend implementation using Discopy.
    """

    def eval(self, diagram: Diagram) -> EvalResult:
        """
        Evaluate the diagram using Discopy.

        Args:
            diagram (Diagram): The diagram to evaluate.

        Returns:
            The result of the evaluation.
        """
        discopy_tensor = self._get_discopy_tensor(diagram).eval()

        if diagram.is_pure:
            state_type = "amps"
        else:
            state_type = "dm"

        return EvalResult(
            discopy_tensor,
            output_types=diagram.cod,
            state_type=state_type,
        )


class PercevalBackend(AbstractBackend):
    """
    Backend implementation using Perceval.
    """

    def __init__(self, perceval_backend : pcvl.ABackend = None):
        """
        Initialize the Perceval backend.

        Args:
            perceval_backend: An optional Perceval backend configuration.
        """
        if perceval_backend is None:
            self.perceval_backend = pcvl.BackendFactory.get_backend("SLOS")
        else:
            self.perceval_backend = perceval_backend

    def eval(self, diagram: Diagram, perceval_state: pcvl.StateVector) -> EvalResult:
        """
        Evaluate the diagram using Perceval. Works only for unitary operations.

        Args:
            diagram (Diagram): The diagram to evaluate.

        Returns:
            The result of the evaluation.
        """

        discopy_tensor = self._get_discopy_tensor(diagram)

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

        array = np.zeros(discopy_tensor.cod.inside)

        if result:
            configs = np.fromiter(
                (i for key in result for i in key),
                dtype=int,
                count=len(result) * array.ndim
            ).reshape(len(result), array.ndim)

            coeffs  = np.fromiter(result.values(), dtype=float, count=len(result))

            array[tuple(configs.T)] = coeffs

        return EvalResult(
            tensor.Box(
                "Result",
                discopy_tensor.dom**0,
                discopy_tensor.cod,
                array
            ),
            output_types=diagram.cod,
            state_type="prob"
        )