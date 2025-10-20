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
>>> np.round(result.single_prob((2, 0)), 1)
0.5

**Compressed contraction (hyper-optimiser reused across calls)**

>>> from cotengra import ReusableHyperCompressedOptimizer
>>> opt = ReusableHyperCompressedOptimizer(max_repeats=32)
>>> backend = QuimbBackend(hyperoptimiser=opt)
>>> result = diag.eval(backend)
>>> np.round(result.single_prob((2, 0)), 1)
0.5

**Unitary circuit simulation with Perceval**

>>> from optyx.core.backends import PercevalBackend
>>> backend = PercevalBackend()
>>> result = (Create(1, 1) >> BS).eval(backend)
>>> np.round(result.single_prob((2, 0)), 1)
0.5
"""

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
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
from optyx.core.channel import Diagram, Ty, mode, bit
from optyx.core.path import Matrix
from optyx.utils.utils import preprocess_quimb_tensors_safe


class StateType(Enum):
    """
    Enum to represent the type of state represented by the result tensor.
    """
    AMP = "amp"  # pure-state amplitudes
    DM = "dm"  # density matrix
    PROB = "prob"  # classical probability distribution
    SINGLE_PROB = "SINGLE_PROB"  # single |<v|U|w>|^2 probability
    SINGLE_AMP = "SINGLE_AMP"  # single <v|U|w> amplitude


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
    def density_matrix(self) -> np.ndarray:
        """
        Get the density matrix from the result tensor.

        Returns:
            np.ndarray: The density matrix.
        """
        if len(self.tensor.dom) != 0:
            raise ValueError(
                "Result tensor must represent a state with no inputs."
            )
        if self.state_type not in {StateType.AMP, StateType.DM}:
            raise TypeError(
                f"Cannot get density matrix from {self.state_type}."
            )
        if self.state_type is StateType.AMP:
            density_matrix = self.tensor.dagger() >> self.tensor
            return density_matrix.array
        return self.tensor.array

    def amplitudes(self, normalise=True) -> dict[tuple[int, ...], complex]:
        """
        Get the amplitudes from the result tensor.
        Returns:
            dict: A dictionary mapping occupation
        configurations to amplitudes.
        """
        if self.state_type != StateType.AMP:
            raise TypeError(
                (f"Cannot get amplitudes from {self.state_type}.")
            )
        if len(self.tensor.dom) != 0:
            raise ValueError(
                "Result tensor must represent a state with no inputs."
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
                "Result tensor must represent a state with no inputs."
            )
        if self.state_type is StateType.AMP:
            return self._prob_dist_pure(round_digits)
        if self.state_type is StateType.DM:
            return self._prob_dist_mixed(round_digits)
        if self.state_type is StateType.PROB:
            values = self._convert_array_to_dict(
                self.tensor.array,
                round_digits
            )

            if np.allclose(np.sum(list(values.values())), 1):
                return values

            probs = {}
            for k, v in values.items():
                val = float(np.real_if_close(v))
                if val < 0 and abs(val) < 1e-12:
                    val = 0.0
                probs[k] = val

            total = float(np.sum(list(probs.values())))
            if total == 0:
                raise ValueError("The probability distribution sums to zero.")

            norm = {k: v / total for k, v in probs.items()}
            return norm

        raise ValueError("Unsupported state_type type. " +
                         "Must be StateType.AMP, StateType.DM, " +
                         "or StateType.PROB.")

    def single_prob(self, occupation: tuple) -> float:
        """
        Get the probability of a specific occupation configuration.

        Args:
            occupation: The occupation configuration to query.

        Returns:
            float: The probability of the specified occupation configuration.
        """
        if self.state_type == StateType.SINGLE_PROB:
            return float(self.tensor.array)

        prob_dist = self.prob_dist()
        return prob_dist.get(occupation, 0.0)

    def single_amplitude(self, occupation: tuple) -> complex:
        """
        Get the amplitude of a specific occupation configuration.

        Args:
            occupation: The occupation configuration to query.

        Returns:
            complex: The amplitude of the specified occupation configuration.
        """
        if self.state_type == StateType.SINGLE_AMP:
            return complex(self.tensor.array)

        dic = self.amplitudes(normalise=False)
        return dic.get(occupation, 0.0)

    def _convert_array_to_dict(
            self,
            array: np.ndarray,
            round_digits: int = None) -> dict:
        """
        Return a dict that maps multi-indices - values for all non-zero
        entries of an array.
        """

        nz_flat = np.flatnonzero(array)
        if nz_flat.size == 0:
            return {}

        nz_vals = array.flat[nz_flat]
        nz_multi = np.vstack(np.unravel_index(nz_flat, array.shape)).T

        if round_digits is not None:
            return {tuple(idx): np.round(val, round_digits) for
                    idx, val in zip(nz_multi, nz_vals)}

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
        if sum_ == 0:
            raise ValueError("The probability distribution sums to zero.")
        return {key: (abs(value) ** 2)/sum_ for key, value in values.items()}

    def _prob_dist_mixed(
            self,
            round_digits: int | None = None) -> dict[tuple[int, ...], float]:
        """
        Get the probability distribution from a mixed state.
        This method computes the probability distribution by aggregating
        occupation configurations based on the output types of the tensor.

        Assumes the output types contain at least one 'bit' or 'mode'
        These will be treated as measured registers while 'qubit' and 'qmode'
        are treated as unmeasured and traced out.

        Args:
            round_digits: Optional number of
            digits to round the probabilities.
        Returns:
            dict: A dictionary mapping
            occupation configurations to probabilities.
        """

        if not any(t in {bit, mode} for t in self.output_types):
            raise ValueError(
                "Output types must contain at least one 'bit' or 'mode'." +
                "These will be treated as measured registers."
            )

        values = self._convert_array_to_dict(self.tensor.array, None)
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

                val = float(np.real_if_close(amp))
                if val < 0 and abs(val) < 1e-12:
                    val = 0.0
                probs[occ_measured] += val

        for occ in all_measured:
            probs.setdefault(occ, 0.0)
        sum_ = np.sum(list(probs.values()))
        if sum_ == 0:
            raise ValueError("The probability distribution sums to zero.")
        prob = {
            key: value / sum_
            for key, value in probs.items()
        }

        if round_digits is not None:
            prob = {k: round(v, round_digits) for k, v in prob.items()}

        return prob


# pylint: disable=too-few-public-methods
class AbstractBackend(ABC):
    """
    Abstract base class for backend implementations.
    All backends must implement the `eval` method.
    """

    def _get_matrix(
        self,
        diagram: Diagram
    ) -> Matrix:
        """
        Get the matrix representation of the diagram.

        Args:
            diagram (Diagram): The diagram to convert.

        Returns:
            np.ndarray: The matrix representation of the diagram.
        """
        try:
            return diagram.to_path()
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

    def _umatrix_to_perceval_circuit(
            self,
            matrix: np.ndarray) -> pcvl.Circuit:
        """
        Convert a unitary matrix to a Perceval circuit.
        """
        # pylint: disable=abstract-class-instantiated
        perceval_matrix = pcvl.Matrix(matrix.T)
        return pcvl.components.Unitary(U=perceval_matrix)

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
            hyperoptimiser: (
                HyperOptimizer |
                ReusableHyperOptimizer |
                HyperCompressedOptimizer |
                ReusableHyperCompressedOptimizer |
                None
             ) = None,
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

        tensor_diagram = self._get_discopy_tensor(diagram)

        if hasattr(tensor_diagram, 'terms'):
            results = sum(
                self._process_term(term) for term in tensor_diagram.terms
            )
        else:
            results = self._process_term(tensor_diagram)

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

    def _process_term(self, term: discopy_tensor.Diagram) -> np.ndarray:
        """
        Process a term in a sum of diagrams.

        Args:
            term (discopy.tensor.Diagram): The term to process.

        Returns:
            np.ndarray: The processed term as a numpy array.
        """
        quimb_tn = term.to_quimb()

        for t in quimb_tn:
            dt = t.data.dtype
            if dt.kind in {'i', 'u', 'b'}:
                t.modify(data=t.data.astype(np.complex128, copy=False))

        if self.hyperoptimiser is None:
            result = quimb_tn ^ ...
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
            result = contract(
                optimize=self.hyperoptimiser,
                output_inds=sorted(quimb_tn.outer_inds()),
                **self.contraction_params
            )

        if not isinstance(result, (complex, float, int)):
            result = result.data

        return result


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
            **extra: Additional arguments for the evaluation:
                - perceval_state: A `perceval.BasicState` or a sequence of
                    non-negative integers (occupation numbers). Defaults
                    to a bosonic product state |11...1> if the
                    diagram does not include any photon creations.
                    Either the creations for all input ports are specified
                    by the diagram (`Create(...)`) or the user must provide
                    the `perceval_state` argument covering all input ports.
                - task: The Perceval task to perform. Allowed values are
                    "probs" (default), "amps", "single_amp", "single_prob".
                - out: Required if task is "single_amp" or "single_prob".
                    A sequence of non-negative integers (occupation numbers)
                    specifying the output configuration for which to compute
                    the amplitude or probability.
        Returns:
            The result of the evaluation (EvalResult).
        """

        if hasattr(
            diagram,
            "terms"
        ):
            array = 0
            for term in diagram.terms:
                arr, output_types, return_type = \
                    self._process_term(
                        term, extra
                    )
                array += arr
        else:
            array, output_types, return_type = \
                self._process_term(
                    diagram, extra
                )

        if array.shape == (1,):
            cod = discopy_tensor.Dim(1)
        else:
            cod = discopy_tensor.Dim(*array.shape)

        return EvalResult(
            discopy_tensor.Box(
                "Result",
                discopy_tensor.Dim(1),
                cod,
                array
            ),
            output_types=output_types,
            state_type=return_type
        )

    def _get_state_from_creations(
        self,
        creations,
        external_perceval_state
    ):
        return (
            external_perceval_state *
            pcvl.BasicState(creations)
        )

    def _get_effect_from_selections(
        self,
        selections,
        external_perceval_effect
    ):
        return (
            external_perceval_effect *
            pcvl.BasicState(selections)
        )

    def _post_select_vacuum(
        self,
        dist,
        m_orig,
        k_extra
    ):
        """Keep only entries where extra (ancilla)
        modes are all 0, then drop them."""
        if k_extra <= 0:
            return dist
        return {
            k[:m_orig]: v
            for k, v in dist.items()
            if all(x == 0 for x in k[m_orig:])
        }

    def _process_state(self, perceval_state):
        if not isinstance(perceval_state, pcvl.BasicState):
            try:
                perceval_state = pcvl.BasicState(list(perceval_state))
            except Exception as e:
                raise TypeError(
                    "perceval_state must be a perceval.BasicState"
                    " or a sequence of non-negative " +
                    "integers (occupation numbers)."
                ) from e
        return perceval_state

    def _process_effect(self, perceval_effect):
        if perceval_effect is None:
            return None
        if not isinstance(perceval_effect, pcvl.BasicState):
            try:
                perceval_effect = pcvl.BasicState(list(perceval_effect))
            except Exception as e:
                raise TypeError(
                    "perceval_effect must be a perceval.BasicState"
                    " or a sequence of non-negative " +
                    "integers (occupation numbers)."
                ) from e
        return perceval_effect

    def _dilate(
        self,
        matrix,
        perceval_state
    ):
        warnings.warn(
            "The provided matrix is not unitary. "
            "PercevalBackend expects a unitary matrix. "
            "Dilation will be used. "
            "This can impact performance.",
            UserWarning,
            stacklevel=2
        )
        current_n_create = len(matrix.creations)
        matrix = matrix.dilate()
        pad_zeros = len(matrix.creations) - current_n_create

        perceval_state = perceval_state * pcvl.BasicState(
            [0] * pad_zeros
        )

        return matrix, perceval_state

    def _process_term(
            self,
            term,
            extra
    ):
        """
        Process a term in a sum of diagrams.

        Args:
            term (discopy.tensor.Diagram): The term to process.
        """
        matrix = self._get_matrix(term)

        task = extra.get("task", "probs")
        state_provided = "perceval_state" in extra
        effect_provided = "perceval_effect" in extra
        is_dom_closed = len(term.dom) == 0

        if not state_provided and not is_dom_closed:
            raise ValueError(
                "External 'perceval_state' not provided but the diagram "
                "has open input modes. Provide a 'perceval_state' " +
                "or close all input modes with a state."
            )

        external_perceval_state = pcvl.BasicState([])
        if state_provided:
            external_perceval_state = self._process_state(
                extra["perceval_state"]
            )

            if external_perceval_state.m != matrix.dom:
                raise ValueError(
                    "The provided 'perceval_state' does not match "
                    "the number of input modes of the diagram."
                )

        perceval_state = self._process_state(
            self._get_state_from_creations(
                matrix.creations,
                external_perceval_state
            )
        )

        perceval_effect = None
        if effect_provided:
            perceval_effect = self._process_effect(
                extra["perceval_effect"]
            )

            if perceval_effect.m != matrix.cod:
                raise ValueError(
                    "The provided 'perceval_effect' does not match "
                    "the number of output modes of the diagram."
                )

        if matrix.cod == 0:
            perceval_effect = pcvl.BasicState([matrix.selections[0]])
            matrix.selections = matrix.selections[1:]
            matrix.cod = 1

        if (
            perceval_effect is not None
        ):
            if task == "amps":
                task = "single_amp"
            if task == "probs":
                task = "single_prob"

        single_output_task = task in ("single_amp", "single_prob")

        if single_output_task:
            if perceval_effect is None:
                raise ValueError(
                    "The 'perceval_effect' argument must be provided for " +
                    "task 'single_amp' or 'single_prob'."
                )

        # pylint: disable=protected-access
        if not matrix._umatrix_is_unitary():
            matrix, perceval_state = self._dilate(
                matrix, perceval_state
            )

        selections = matrix.selections

        sim = pcvl.Simulator(self.perceval_backend)
        postselect_conditions = [
            f"{str([i + matrix.cod])} == {s}"
            for i, s in enumerate(selections)
        ]
        sim.set_postselection(
            pcvl.PostSelect(str.join(" & ", postselect_conditions))
        )
        perceval_circuit = self._umatrix_to_perceval_circuit(matrix.array)
        sim.set_circuit(perceval_circuit)

        m_orig = len(term.dom)
        k_extra = matrix.dom - m_orig
        result = None
        p = None

        if not single_output_task:
            if task == "probs":
                result = sim.probs(perceval_state)
                result = {tuple(k): float(v) for k, v in result.items()}
                return_type = StateType.PROB
            else:
                sv = sim.evolve(perceval_state)
                result = {tuple(k): complex(v) for k, v in sv}
                return_type = StateType.AMP

            result = self._post_select_vacuum(result, m_orig, k_extra)
            output_types = term.cod
            output_shape = self._get_discopy_tensor(term).cod.inside

        elif single_output_task:
            if task == "single_prob":
                p = float(sim.probability(perceval_state, perceval_effect))
                return_type = StateType.SINGLE_PROB

            else:
                p = complex(
                    sim.prob_amplitude(perceval_state, perceval_effect)
                )
                return_type = StateType.SINGLE_AMP

            output_shape = (1,)
            output_types = None

        else:
            raise ValueError(
                "Invalid task. Allowed values are" +
                " 'probs', 'amps', 'single_amp', 'single_prob'."
            )

        array = np.zeros(
            output_shape,
            dtype=float if task in ("single_prob", "probs") else complex
        )

        if result and not single_output_task:
            configs = np.fromiter(
                (i for key in result for i in key),
                dtype=int,
                count=len(result) * array.ndim
            ).reshape(len(result), array.ndim)

            coeffs = np.fromiter(
                result.values(),
                dtype=float if task == "probs" else complex,
                count=len(result)
            )

            array[tuple(configs.T)] = coeffs

        elif single_output_task:
            array[0] = p
        else:
            pass
        return array, output_types, return_type


class PermanentBackend(AbstractBackend):
    """
    Backend implementation using optyx' Path module to compute matrix
    permanents.
    """

    def eval(
        self,
        diagram: Diagram,
        **extra: Any
    ):
        """
        Evaluate the diagram using the Permanent/Path backend.
        Works only for LO circuits.

        Args:
            diagram (Diagram): The diagram to evaluate.
            **extra: Additional arguments for the evaluation,
            including 'n_photons' for optyx.core.path.Matrix.eval.
        """

        n_photons = extra.get(
            "n_photons",
            0
        )

        def check_creations(matrix):
            if (
                len(matrix.creations) == 0 and
                n_photons == 0
            ):
                raise ValueError(
                    "The diagram does not include any photon creations. " +
                    "n_photons must be greater than 0."
                )

        if hasattr(
            diagram,
            "terms"
        ):
            result = 0
            dims = []
            for term in diagram.terms:
                matrix = self._get_matrix(term)
                check_creations(matrix)
                result_matrix = matrix.eval(
                    n_photons=n_photons,
                    as_tensor=True
                )
                result += result_matrix.array
                dims.append(
                    (
                        result_matrix.dom.inside,  # list
                        result_matrix.cod.inside  # list
                    )
                )
        else:
            matrix = self._get_matrix(diagram)
            check_creations(matrix)
            result_matrix = matrix.eval(n_photons=n_photons, as_tensor=True)
            result = result_matrix.array
            dims = [(result_matrix.dom.inside, result_matrix.cod.inside)]

        norm = lambda x: list(x) if isinstance(  # noqa: E731
            x, (list, tuple)
        ) else [x]
        dom_lists, cod_lists = zip(*((norm(d), norm(c)) for d, c in dims))
        max_dom_dims = [max(vals) for vals in zip(*dom_lists)]
        max_cod_dims = [max(vals) for vals in zip(*cod_lists)]

        return EvalResult(
            discopy_tensor.Box(
                "Result",
                discopy_tensor.Dim(*tuple(max_dom_dims)),
                discopy_tensor.Dim(*tuple(max_cod_dims)),
                result
            ),
            output_types=diagram.cod,
            state_type=StateType.AMP
        )
