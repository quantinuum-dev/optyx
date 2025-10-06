"""

Overview
--------

Back-end abstraction layer for **Optyx** diagram evaluation.

This module gathers several numerical engines under a single
interface so that any :class:`optyx.core.channel.Diagram` can be reduced
to concrete data - amplitudes, density matrices or classical probability
distributions.

Three families of engines are provided:

* **Quimb** – tensor-network contraction with optional
  hyper-optimisers for exact or compressed algorithms.
* **DisCoPy** – direct evaluation of a DisCoPy :class:`tensor.Diagram`,
  useful for small circuits or examples.
* **Perceval** – simulation of *unitary* linear-optics
  circuits, returning classical output statistics.

Classes
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    EvalResult
    AbstractBackend
    QuimbBackend
    DiscopyBackend
    PercevalBackend

Examples of usage
-----------------

**Exact tensor-network contraction with Quimb**

>>> from optyx.photonic import Create, BS
>>> from optyx.core.backends import QuimbBackend
>>> diag = Create(1, 1) >> BS
>>> backend = QuimbBackend()
>>> result = diag.eval(backend)
>>> np.round(result.prob((2, 0)), 1)
0.5

**Compressed contraction (hyper-optimiser reused across calls)**

>>> from cotengra import ReusableHyperCompressedOptimizer
>>> opt = ReusableHyperCompressedOptimizer(max_repeats=32)
>>> backend = QuimbBackend(hyperoptimiser=opt)
>>> result = diag.eval(backend)
>>> np.round(result.prob((2, 0)), 1)
0.5

**Unitary circuit simulation with Perceval**

>>> from optyx.core.backends import PercevalBackend
>>> backend = PercevalBackend()
>>> result = BS.eval(backend)
>>> np.round(result.prob((2, 0)), 1)
0.5
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Union
from collections import defaultdict
from enum import Enum
from cotengra import (
    ReusableHyperOptimizer,
    HyperOptimizer,
    HyperCompressedOptimizer,
    ReusableHyperCompressedOptimizer,
)
from discopy import tensor as discopy_tensor
import numpy as np
import perceval as pcvl
from quimb.tensor import TensorNetwork
from optyx.core.channel import Diagram
from optyx.core.channel import Ty, mode, bit
from optyx.utils.utils import preprocess_quimb_tensors_safe


class StateType(Enum):
    """
    Enum to represent the type of state represented by the result tensor.
    """
    AMP = "amp"     # pure-state amplitudes
    DM = "dm"       # density matrix
    PROB = "prob"   # classical probability distribution


@dataclass(frozen=True)
class EvalResult:
    """
    Class to encapsulate the result of an evaluation of a diagram.
    """
    _tensor: discopy_tensor.Box
    output_types: Ty
    state_type: StateType

    @property
    def tensor(self) -> discopy_tensor.Box:
        """
        Get the resulting tensor.

        Returns:
            tensor.Box: The result tensor.
        """
        return self._tensor

    @property
    def density_matrix(self) -> discopy_tensor.Box:
        """
        Get the density matrix from the result tensor.

        Returns:
            tensor.Box: The density matrix.
        """
        if len(self.tensor.dom) != 0:
            raise ValueError(
                "Result tensor must represent a state without inputs."
            )
        if self.state_type not in {StateType.AMP, StateType.DM}:
            raise TypeError(
                "Cannot get density matrix from probability distribution."
            )
        if self.state_type is StateType.AMP:
            density_matrix = self.tensor.dagger() >> self.tensor
            return density_matrix
        return self.tensor.array

    def amplitudes(self, normalise=True) -> dict[tuple[int, ...], float]:
        """
        Get the amplitudes from the result tensor.
        Returns:
            dict: A dictionary mapping occupation
        configurations to amplitudes.
        """
        if self.state_type != StateType.AMP:
            raise TypeError(
                ("Cannot get amplitudes from density " +
                 "matrix or probability distribution.")
            )
        if len(self.tensor.dom) != 0:
            raise ValueError(
                "Result tensor must represent a state without inputs."
            )

        dic = self._convert_array_to_dict(self.tensor.array)
        if normalise:
            return {key: value /
                    np.sqrt(np.sum(np.abs(list(dic.values()))**2))
                    for key, value in dic.items()}
        return dic

    def prob_dist(self, round_digits: int = None) -> dict:
        """
        Get the probability distribution from the result tensor.

        Returns:
            dict: A dictionary mapping occupation
              configurations to probabilities.
        """
        if len(self.tensor.dom) != 0:
            raise ValueError(
                "Result tensor must represent a state without inputs."
            )
        if self.state_type is StateType.AMP:
            return self._prob_dist_pure(round_digits)
        if self.state_type is StateType.DM:
            return self._prob_dist_mixed(round_digits)
        if self.state_type is StateType.PROB:
            return self._convert_array_to_dict(
                self.tensor.array,
                round_digits=round_digits
            )
        raise ValueError("Unsupported state_type type. " +
                         "Must be StateType.AMP, StateType.DM, " +
                         "or StateType.PROB.")

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

    # pylint: disable=no-self-use
    def _convert_array_to_dict(
            self,
            array: np.ndarray,
            round_digits: int = None) -> dict:
        """
        Return a dict that maps multi-indices - values for all non-zero
        entries of an array.
        """

        if round_digits is not None:
            array = np.round(array, round_digits)

        nz_flat = np.flatnonzero(array)
        if nz_flat.size == 0:
            return {}

        nz_vals = array.flat[nz_flat]
        nz_multi = np.vstack(np.unravel_index(nz_flat, array.shape)).T
        return {tuple(idx): val for idx, val in zip(nz_multi, nz_vals)}

    def _prob_dist_pure(self, round_digits: int = None) -> dict:
        """
        Get the probability distribution for a pure state.

        Returns:
            dict: A dictionary mapping occupation
            configurations to probabilities.
        """

        values = self._convert_array_to_dict(
            self.tensor.array,
            round_digits=round_digits
        )
        sum_ = np.sum(np.abs(list(values.values())) ** 2)
        return {key: (abs(value) ** 2)/sum_ for key, value in values.items()}

    def _prob_dist_mixed(
            self,
            round_digits: int | None = None) -> dict[tuple[int, ...], float]:
        """
        Get the probability distribution from a mixed state.
        This method computes the probability distribution by aggregating
        occupation configurations based on the output types of the tensor.

        Args:
            round_digits: Optional number of
            digits to round the probabilities.
        Returns:
            dict: A dictionary mapping
            occupation configurations to probabilities.
        """

        if not any(t in {bit, mode} for t in self.output_types):
            raise ValueError(
                "Types must contain at least one 'bit' or 'mode'."
            )

        values = self._convert_array_to_dict(self.tensor.array, round_digits)
        mask_flat = np.concatenate(
            [[1] if t in {bit, mode} else [0, 0] for t in self.output_types]
        )

        probs = defaultdict(float)
        all_measured = set()

        for key, amp in values.items():
            occ_measured = tuple(i for i, m in zip(key, mask_flat) if m)
            all_measured.add(occ_measured)

            occs_unmeasured = tuple(i for i, m in
                                    zip(key, mask_flat) if not m)
            if all(occs_unmeasured[i] == occs_unmeasured[i + 1]
                   for i in range(0, len(occs_unmeasured) - 1, 2)):
                probs[occ_measured] += amp

        for occ in all_measured:
            probs.setdefault(occ, 0.0)
        sum_ = np.sum(list(probs.values()))
        prob = {
            key: value / sum_
            for key, value in probs.items()
        }
        return prob


# pylint: disable=too-few-public-methods
class AbstractBackend(ABC):
    """
    Abstract base class for backend implementations.
    All backends must implement the `eval` method.
    """

    # pylint: disable=no-self-use
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
        except NotImplementedError as error:
            raise NotImplementedError(
                "The diagram cannot be converted to a matrix. " +
                "It is not linear optical."
            ) from error

    def _get_quimb_tensor(
        self,
        diagram: Diagram
    ) -> TensorNetwork:
        """
        Get the Quimb tensor representation of the diagram.
        """
        return self._get_discopy_tensor(diagram).to_quimb()

    # pylint: disable=no-self-use
    def _umatrix_to_perceval_circuit(
            self,
            matrix: np.ndarray) -> pcvl.Circuit:
        """
        Convert a unitary matrix to a Perceval circuit.
        """
        # pylint: disable=abstract-class-instantiated
        perceval_matrix = pcvl.Matrix(matrix.T)
        return pcvl.components.Unitary(U=perceval_matrix)

    # pylint: disable=no-self-use
    def _get_discopy_tensor(
        self,
        diagram: Diagram
    ) -> discopy_tensor.Diagram:
        """
        Get the Discopy tensor representation of the diagram.
        """
        if diagram.is_pure:
            return diagram.get_kraus().to_tensor()
        return diagram.double().to_tensor()

    @abstractmethod
    def eval(self, diagram: Diagram, **extra: Any) -> EvalResult:
        """
        Evaluate the backend with the given arguments.

        Args:
            diagram (Diagram): The diagram to evaluate.
            **extra: Additional arguments for the evaluation.
        Returns:
            The result of the evaluation.
        """


# pylint: disable=too-few-public-methods
class QuimbBackend(AbstractBackend):
    """
    Backend implementation using Quimb.
    """

    def __init__(
            self,
            hyperoptimiser: Union[
                HyperOptimizer,
                ReusableHyperOptimizer,
                HyperCompressedOptimizer,
                ReusableHyperCompressedOptimizer
            ] = None,
            contraction_params: dict = None):
        """
        Initialize the Quimb backend.

        Args:
            hyperoptimiser: An optional hyperoptimiser
            for contraction optimization.
            contraction_params: Optional parameters for the contraction.
        """
        self.hyperoptimiser = hyperoptimiser
        self.contraction_params = contraction_params or {}

    def eval(
            self,
            diagram: Diagram,
            **extra: Any) -> EvalResult:
        """
        Evaluate the diagram using Quimb.

        Args:
            diagram (Diagram): The diagram to evaluate.

        Returns:
            The result of the evaluation.
        """
        quimb_tn = self._get_quimb_tensor(diagram)
        tensor_diagram = self._get_discopy_tensor(diagram)

        for t in quimb_tn:
            dt = t.data.dtype
            if dt.kind in {'i', 'u', 'b'}:
                t.modify(data=t.data.astype(np.complex128, copy=False))

        if self.hyperoptimiser is None:
            results = quimb_tn ^ ...
        else:
            is_approx = isinstance(
                self.hyperoptimiser,
                (ReusableHyperCompressedOptimizer, HyperCompressedOptimizer)
            )

            is_exact = isinstance(
                self.hyperoptimiser,
                (ReusableHyperOptimizer, HyperOptimizer)
            )

            if not is_approx and not is_exact:
                raise ValueError(
                    "Unsupported hyperoptimiser type. " +
                    "Use ReusableHyperOptimizer, HyperOptimizer, " +
                    "ReusableHyperCompressedOptimizer, or " +
                    "HyperCompressedOptimizer."
                )

            if is_approx:
                quimb_tn = preprocess_quimb_tensors_safe(quimb_tn)

            contract = quimb_tn.contract_compressed if \
                is_approx else quimb_tn.contract
            results = contract(
                optimize=self.hyperoptimiser,
                output_inds=sorted(quimb_tn.outer_inds()),
                **self.contraction_params
            )

        if not isinstance(results, (complex, float, int)):
            results = results.data

        if diagram.is_pure:
            state_type = StateType.AMP
        else:
            state_type = StateType.DM

        return EvalResult(
            discopy_tensor.Box(
                "Result",
                tensor_diagram.dom,
                tensor_diagram.cod,
                results
            ),
            output_types=diagram.cod,
            state_type=state_type
        )


# pylint: disable=too-few-public-methods
class DiscopyBackend(AbstractBackend):
    """
    Backend implementation using Discopy.
    """

    def eval(self, diagram: Diagram,  **extra: Any) -> EvalResult:
        """
        Evaluate the diagram using Discopy.

        Args:
            diagram (Diagram): The diagram to evaluate.

        Returns:
            The result of the evaluation.
        """
        tensor_diagram = self._get_discopy_tensor(diagram).eval()

        if diagram.is_pure:
            state_type = StateType.AMP
        else:
            state_type = StateType.DM

        return EvalResult(
            tensor_diagram,
            output_types=diagram.cod,
            state_type=state_type,
        )


# pylint: disable=too-few-public-methods
class PercevalBackend(AbstractBackend):
    """
    Backend implementation using Perceval.
    """

    def __init__(self, perceval_backend: pcvl.ABackend = None):
        """
        Initialize the Perceval backend.

        Args:
            perceval_backend: An optional Perceval backend configuration.
        """
        if perceval_backend is None:
            self.perceval_backend = pcvl.BackendFactory.get_backend("SLOS")
        else:
            self.perceval_backend = perceval_backend

    def eval(
            self,
            diagram: Diagram,
            **extra: Any) -> EvalResult:
        """
        Evaluate the diagram using Perceval.
        Works only for unitary operations.
        If no `perceval_state` is provided in `extra`,
        it defaults to a bosonic product state.

        Args:
            diagram (Diagram): The diagram to evaluate.
            **extra: Additional arguments for the evaluation,
            including 'perceval_state'.

        Returns:
            The result of the evaluation.
        """

        if extra:
            try:
                perceval_state: pcvl.StateVector = extra["perceval_state"]
            except KeyError as error:
                raise TypeError(
                    "PercevalBackend.eval requires " +
                    "a 'perceval_state=' keyword."
                ) from error
        else:
            perceval_state = pcvl.StateVector(
                [1] * len(diagram.dom)
            )
        tensor_diagram = self._get_discopy_tensor(diagram)

        sim = pcvl.Simulator(self.perceval_backend)
        matrix = self._get_matrix(diagram)

        if not np.allclose(
            np.eye(matrix.shape[0]),
            matrix.dot(matrix.conj().T)
        ):
            raise ValueError(
                "The provided diagram does not represent a unitary operation."
            )

        perceval_circuit = self._umatrix_to_perceval_circuit(matrix)
        sim.set_circuit(perceval_circuit)
        result = sim.probs(perceval_state)
        result = {tuple(k): v for k, v in result.items()}

        array = np.zeros(tensor_diagram.cod.inside)

        if result:
            configs = np.fromiter(
                (i for key in result for i in key),
                dtype=int,
                count=len(result) * array.ndim
            ).reshape(len(result), array.ndim)

            coeffs = np.fromiter(
                result.values(),
                dtype=float,
                count=len(result)
            )

            array[tuple(configs.T)] = coeffs

        return EvalResult(
            discopy_tensor.Box(
                "Result",
                tensor_diagram.dom**0,
                tensor_diagram.cod,
                array
            ),
            output_types=diagram.cod,
            state_type=StateType.PROB
        )
