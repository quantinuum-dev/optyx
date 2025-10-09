import pytest
from optyx import photonic, qubits, classical, mode, qubit, qmode, bit
from cotengra import ReusableHyperCompressedOptimizer
from optyx.core.backends import QuimbBackend, PercevalBackend, DiscopyBackend, EvalResult, StateType
import numpy as np
import math
from itertools import chain
import perceval as pcvl
import discopy.tensor as discopy_tensor

@pytest.mark.skip(reason="Helper function for testing")
def chip_mzi(w, l):
    ansatz = photonic.ansatz(w, l)
    symbs = list(ansatz.free_symbols)
    s = [(i, np.random.uniform(0, 1)) for i in symbs]
    return ansatz.subs(*s)

PURE_CIRCUITS_TO_TEST = [
    photonic.BS,
    photonic.Phase(0.2) @ photonic.Phase(0.3) >> photonic.TBS(0.3),
    photonic.MZI(0.2, 0.8),
    chip_mzi(4, 4)
]

MIXED_CIRCUITS_TO_TEST = [
    photonic.BS >> photonic.Discard(1) @ photonic.qmode,
    (
        qubits.Id(1) @ qubits.Z(0, 2) @ qubits.Scalar(2**0.5) >>
        qubits.Z(1, 2) @ qubits.Id(2) >>
        qubits.Id(1) @ qubits.Z(2, 1) @ qubits.Id(1) >>
        qubits.Id(1) @ qubits.H() @ qubits.Id(1) >>
        qubits.Measure(1)**2 @ qubits.Id(1) >>
        classical.Id(qubits.bit) @ classical.BitControlledGate(qubits.X(1, 1, 0.5)) >>
        classical.BitControlledGate(qubits.Z(1, 1, 0.5))
    ),
]

CIRCUITS_WITH_DISCARDS_TO_TEST = [
    photonic.BS @ photonic.BS >> photonic.Discard(1) @ photonic.qmode**3,
    chip_mzi(4, 4) >> photonic.Discard(1) @ photonic.qmode**3
]

@pytest.mark.skip(reason="Helper function for testing")
def get_state(circuit):
    if circuit.dom[0] == photonic.qmode:
        return (1,)*len(circuit.dom)
    elif circuit.dom[0] == qubits.qubit:
        return qubits.X(0, 1, 0.3)

@pytest.mark.skip(reason="Helper function for testing")
def dict_allclose(d1: dict, d2: dict, *, rel_tol=1e-05, abs_tol=1e-10) -> bool:
    for key in set(chain(d1, d2)):
        v1 = d1.get(key, 0.0)
        v2 = d2.get(key, 0.0)
        if not math.isclose(v1, v2, rel_tol=rel_tol, abs_tol=abs_tol):
            return False
    return True

class TestQuimbBackend:
    # compare exact and approx backends
    @pytest.mark.parametrize("circuit", PURE_CIRCUITS_TO_TEST)
    def test_pure_circuit(self, circuit):
        state = photonic.Create(*get_state(circuit))
        diagram = state >> circuit
        result_exact = diagram.eval()
        opt = ReusableHyperCompressedOptimizer(max_repeats=32)
        backend = QuimbBackend(hyperoptimiser=opt)
        result_approx = diagram.eval(backend)

        assert dict_allclose(
            result_exact.prob_dist(),
            result_approx.prob_dist(),
        )

        assert dict_allclose(
            result_exact.amplitudes(),
            result_approx.amplitudes(),
        )

        assert np.allclose(
            result_exact.tensor.array,
            result_approx.tensor.array,
        )

    @pytest.mark.parametrize("circuit", MIXED_CIRCUITS_TO_TEST)
    def test_mixed_circuit(self, circuit):
        s = get_state(circuit)
        state = photonic.Create(*s) if isinstance(s, tuple) else s
        cod_len = len(circuit.cod)
        measure = photonic.NumberResolvingMeasurement(cod_len) if circuit.cod[0] == photonic.qmode else qubits.Measure(cod_len)
        diagram = state >> circuit >> measure
        result_exact = diagram.eval()
        opt = ReusableHyperCompressedOptimizer(max_repeats=32)
        backend = QuimbBackend(hyperoptimiser=opt)
        result_approx = diagram.eval(backend)

        assert dict_allclose(
            result_exact.prob_dist(),
            result_approx.prob_dist(),
        )

        assert np.allclose(
            result_exact.tensor.array,
            result_approx.tensor.array,
        )

        assert np.allclose(
            result_exact.density_matrix,
            result_approx.density_matrix,
        )

class TestPercevalBackend:
    # compare with matrix.probs
    @pytest.mark.parametrize("circuit", PURE_CIRCUITS_TO_TEST)
    def test_pure_circuit(self, circuit):
        backend = PercevalBackend()
        perceval_state = pcvl.BasicState(get_state(circuit))
        result_perceval = circuit.eval(backend, perceval_state=perceval_state)

        state = photonic.Create(*get_state(circuit))
        diagram = state >> circuit
        result_quimb = diagram.eval()

        assert dict_allclose(
            result_quimb.prob_dist(),
            result_perceval.prob_dist(),
        )

class TestDiscopyBackend:
    # compare with matrix.probs
    @pytest.mark.parametrize("circuit", PURE_CIRCUITS_TO_TEST)
    def test_pure_circuit(self, circuit):
        state = photonic.Create(*get_state(circuit))
        diagram = state >> circuit

        backend = DiscopyBackend()
        result_discopy = diagram.eval(backend)

        result_quimb = diagram.eval()

        assert dict_allclose(
            result_quimb.prob_dist(),
            result_discopy.prob_dist(),
        )

    @pytest.mark.parametrize("circuit", MIXED_CIRCUITS_TO_TEST)
    def test_mixed_circuit(self, circuit):
        s = get_state(circuit)
        state = photonic.Create(*s) if isinstance(s, tuple) else s
        cod_len = len(circuit.cod)
        measure = photonic.NumberResolvingMeasurement(cod_len) if circuit.cod[0] == photonic.qmode else qubits.Measure(cod_len)
        diagram = state >> circuit >> measure
        result_exact = diagram.eval()
        backend = DiscopyBackend()
        result_discopy = diagram.eval(backend)

        assert dict_allclose(
            result_exact.prob_dist(),
            result_discopy.prob_dist(),
        )

        assert np.allclose(
            result_exact.tensor.array,
            result_discopy.tensor.array,
        )

        assert np.allclose(
            result_exact.density_matrix,
            result_discopy.density_matrix,
        )

class TestEvalResult:
    # compare circuits with measurements and dangling wires with
    # circuits without measurements and discards
    # for mixed (Discopy + Quimb)
        # discopy and quimb
    @pytest.mark.parametrize("circuit", CIRCUITS_WITH_DISCARDS_TO_TEST)
    def test_circuits_with_discards(self, circuit):
        s = get_state(circuit)
        state = photonic.Create(*s)
        measure = photonic.NumberResolvingMeasurement(1)
        diagram = state >> circuit >> measure @ photonic.qmode**2

        result_standard = diagram.eval()

        diagram_with_discards = state >> circuit >> measure @ photonic.Discard(2)
        result_with_discards = diagram_with_discards.eval()

        assert dict_allclose(
            result_with_discards.prob_dist(),
            result_standard.prob_dist(),
        )

    # check if prob_dist is square of amps for pure circuits
    @pytest.mark.parametrize("circuit", PURE_CIRCUITS_TO_TEST)
    def test_pure_prob_dist_amp(self, circuit):
        state = photonic.Create(*get_state(circuit))
        diagram = state >> circuit
        result = diagram.eval()

        prob_dist = result.prob_dist()
        amps = result.amplitudes()

        for key, amp in amps.items():
            assert math.isclose(prob_dist.get(key, 0.0), abs(amp)**2)

    @pytest.mark.parametrize(
            "circuit",
            CIRCUITS_WITH_DISCARDS_TO_TEST + PURE_CIRCUITS_TO_TEST + MIXED_CIRCUITS_TO_TEST
    )
    def test_probs_sum_to_one(self, circuit):
        s = get_state(circuit)
        state = photonic.Create(*s) if isinstance(s, tuple) else s
        cod_len = len(circuit.cod)
        if circuit.cod[0] != photonic.qmode:
            return
        measure = photonic.NumberResolvingMeasurement(cod_len)
        diagram = state >> circuit >> measure
        result = diagram.eval()

        prob_dist = result.prob_dist()
        total_prob = sum(prob_dist.values())
        assert math.isclose(total_prob, 1.0)

    def test_prob_lookup_and_missing_key(self):
        v = np.array([1/np.sqrt(2), 1j/np.sqrt(2)], dtype=complex)
        box = discopy_tensor.Tensor(v, discopy_tensor.Dim(1), discopy_tensor.Dim(2))
        ev = EvalResult(_tensor=box, output_types=(mode,), state_type=StateType.AMP)

        p = ev.prob_dist()
        assert ev.prob((0,)) == pytest.approx(p[(0,)], rel=1e-12)
        assert ev.prob((1,)) == pytest.approx(p[(1,)], rel=1e-12)
        assert ev.prob((2,)) == 0.0

    def test_density_matrix_from_amp_is_outer_product(self):
        v = np.array([1/np.sqrt(3), np.sqrt(2/3)], dtype=complex)
        box = discopy_tensor.Tensor(v, discopy_tensor.Dim(1), discopy_tensor.Dim(2))
        ev = EvalResult(_tensor=box, output_types=(mode,), state_type=StateType.AMP)
        dm = ev.density_matrix
        assert dm.shape == (2, 2)
        assert np.allclose(dm, np.outer(v.conj(), v), atol=1e-12)

    def test_mixed_interleaved_partial_trace_numeric_hygiene(self):
        dm = np.zeros((2, 2, 2), dtype=complex)
        dm[0, 0, 0] = 0.499999999999   + 1e-14j
        dm[0, 1, 1] = 0.100000000001   - 1e-14j
        dm[1, 0, 0] = 0.2000000000005  + 1e-14j
        dm[1, 1, 1] = 0.2000000000005  - 1e-14j
        #off-diagonals that should be ignored by the diagonal check
        dm[0, 0, 1] = -1e-13
        dm[1, 1, 0] = -1e-13

        box = discopy_tensor.Tensor(dm, discopy_tensor.Dim(1), discopy_tensor.Dim(*dm.shape))
        #1st axis measured (mode), 2ns is quantum (unmeasured, bra/ket)
        ev = EvalResult(_tensor=box, output_types=(mode, object()), state_type=StateType.DM)
        probs = ev.prob_dist()

        assert pytest.approx(probs[(0,)], rel=1e-12) == 0.6
        assert pytest.approx(probs[(1,)], rel=1e-12) == 0.4
        assert pytest.approx(sum(probs.values()), rel=1e-12) == 1.0

    def test_prob_branch_rounding_and_sum(self):
        P = np.array([[1/3, 1/6],
                      [1/6, 1/3]], dtype=float)
        box = discopy_tensor.Tensor(P, discopy_tensor.Dim(1), discopy_tensor.Dim(*P.shape))
        ev = EvalResult(_tensor=box, output_types=(mode, mode), state_type=StateType.PROB)

        d = ev.prob_dist(round_digits=4)
        assert pytest.approx(sum(d.values()), rel=1e-12) == 1.0
        print(d)
        assert d[(0, 0)] == round(float(P[0, 0]), 4)
        assert d[(1, 1)] == round(float(P[1, 1]), 4)

    def test_density_matrix_from_amp_complex_phase(self):
        v = np.array([np.exp(1j*0.2)/np.sqrt(3),
                      np.exp(-1j*0.3)*np.sqrt(2/3)], dtype=complex)
        box = discopy_tensor.Tensor(v, discopy_tensor.Dim(1), discopy_tensor.Dim(2))
        ev = EvalResult(_tensor=box, output_types=(mode,), state_type=StateType.AMP)
        dm = ev.density_matrix
        assert dm.shape == (2, 2)
        assert np.allclose(dm, dm.conj().T, atol=1e-12)
        assert np.allclose(dm, np.outer(v.conj(), v), atol=1e-12)
        eig = np.linalg.eigvalsh(dm)
        assert np.all(eig >= -1e-12)

    def test_density_matrix_passthrough_for_DM(self):
        dm = np.array([[0.7, 0.1j],
                       [-0.1j, 0.3]], dtype=complex)
        box = discopy_tensor.Tensor(dm, discopy_tensor.Dim(1), discopy_tensor.Dim(*dm.shape))
        ev = EvalResult(_tensor=box, output_types=(mode,), state_type=StateType.DM)
        out = ev.density_matrix
        assert out.shape == (2, 2)
        assert np.allclose(out, dm, atol=1e-12)

    def test_mixed_simple_single_measured_single_unmeasured(self):
        dm = np.zeros((2, 2, 2), dtype=complex)
        dm[0, 0, 0] = 0.6 + 1e-14j
        dm[1, 1, 1] = 0.4 - 1e-14j
        dm[0, 1, 0] = -1e-13
        dm[1, 0, 1] = -1e-13

        box = discopy_tensor.Tensor(dm, discopy_tensor.Dim(1), discopy_tensor.Dim(*dm.shape))
        ev = EvalResult(_tensor=box, output_types=(mode, qubit), state_type=StateType.DM)
        probs = ev.prob_dist()
        assert pytest.approx(probs[(0,)], rel=1e-12) == 0.6
        assert pytest.approx(probs[(1,)], rel=1e-12) == 0.4
        assert pytest.approx(sum(probs.values()), rel=1e-12) == 1.0

    def test_mixed_two_measured_one_unmeasured_matches_manual_trace(self):
        P = np.array([[0.1, 0.2],
                      [0.3, 0.4]], dtype=float)
        W = np.array([0.6, 0.4], dtype=float)
        dm = np.zeros((2, 2, 2, 2), dtype=complex)

        for k in range(2):
            for l in range(2):
                dm[k, l, 0, 0] = P[k, l] * W[0]
                dm[k, l, 1, 1] = P[k, l] * W[1]
        dm[0, 0, 0, 1] = 1e-13
        dm[1, 1, 1, 0] = -1e-13

        box = discopy_tensor.Tensor(dm, discopy_tensor.Dim(1), discopy_tensor.Dim(*dm.shape))
        ev = EvalResult(_tensor=box, output_types=(mode, bit, qubit), state_type=StateType.DM)

        probs = ev.prob_dist()
        manual = {(k, l): (dm[k, l, 0, 0].real + dm[k, l, 1, 1].real) for k in range(2) for l in range(2)}
        Z = sum(manual.values())
        manual = {k: v / Z for k, v in manual.items()}

        for k in range(2):
            for l in range(2):
                assert pytest.approx(probs[(k, l)], rel=1e-12) == manual[(k, l)]
        assert pytest.approx(sum(probs.values()), rel=1e-12) == 1.0

    def test_mixed_round_digits_applied_before_normalization(self):
        dm = np.zeros((2, 2, 2), dtype=complex)
        dm[0, 0, 0] = 0.333333333333
        dm[1, 1, 1] = 0.666666666667
        box = discopy_tensor.Tensor(dm, discopy_tensor.Dim(1), discopy_tensor.Dim(*dm.shape))
        ev = EvalResult(_tensor=box, output_types=(mode, qubit), state_type=StateType.DM)
        d = ev.prob_dist(round_digits=6)
        assert pytest.approx(sum(d.values()), rel=1e-12) == 1.0
        assert d[(0,)] == pytest.approx(round(0.333333333333, 6) / (round(0.333333333333, 6) + round(0.666666666667, 6)), rel=1e-12)
        assert d[(1,)] == pytest.approx(round(0.666666666667, 6) / (round(0.333333333333, 6) + round(0.666666666667, 6)), rel=1e-12)

    def test_mixed_requires_at_least_one_measured_type(self):
        dm = np.eye(4, dtype=complex).reshape(2, 2, 2, 2) / 4.0
        box = discopy_tensor.Tensor(dm, discopy_tensor.Dim(1), discopy_tensor.Dim(*dm.shape))
        ev = EvalResult(_tensor=box, output_types=(qubit, qmode), state_type=StateType.DM)
        with pytest.raises(ValueError):
            _ = ev.prob_dist()

    def test_mixed_two_unmeasured_pairs_partial_trace(self):

        dm = np.zeros((2, 2, 2, 2, 2), dtype=float)

        w_m = {0: 0.55, 1: 0.45}
        w_u1 = np.array([0.6, 0.4])
        w_u2 = np.array([0.7, 0.3])

        for m in (0, 1):
            for r1 in (0, 1):
                for r2 in (0, 1):
                    dm[m, r1, r1, r2, r2] = w_m[m] * w_u1[r1] * w_u2[r2]

        dm[0, 0, 1, 0, 0] = 1e-13
        dm[1, 1, 0, 1, 0] = -1e-13

        box = discopy_tensor.Tensor(dm, discopy_tensor.Dim(1), discopy_tensor.Dim(*dm.shape))
        ev = EvalResult(_tensor=box, output_types=(mode, object(), object()), state_type=StateType.DM)
        probs = ev.prob_dist()

        assert pytest.approx(sum(probs.values()), rel=1e-12) == 1.0
        assert pytest.approx(probs[(0,)], rel=1e-12) == 0.55
        assert pytest.approx(probs[(1,)], rel=1e-12) == 0.45


class TestExceptions:
    # EvalResult
    # density matrix/amps/prob from not a state
    @pytest.mark.parametrize("circuit", MIXED_CIRCUITS_TO_TEST)
    def test_eval_result_not_state(self, circuit):
        with pytest.raises(ValueError):
            result = circuit.eval()
            prob_dist = result.prob_dist()

        with pytest.raises(ValueError):
            result = circuit.eval()
            density_matrix = result.density_matrix

    # density matrix from perceval eval
    @pytest.mark.parametrize("circuit", PURE_CIRCUITS_TO_TEST)
    def test_eval_result_not_density_matrix(self, circuit):
        with pytest.raises(TypeError):
            backend = PercevalBackend()
            perceval_state = pcvl.BasicState(get_state(circuit))
            result_perceval = circuit.eval(backend, perceval_state=perceval_state)

            dm = result_perceval.density_matrix

        # amps from mixed circuits and prob dist
    @pytest.mark.parametrize("circuit", MIXED_CIRCUITS_TO_TEST)
    def test_eval_result_not_amps(self, circuit):
        with pytest.raises(TypeError):
            s = get_state(circuit)
            state = photonic.Create(*s) if isinstance(s, tuple) else s
            cod_len = len(circuit.cod)
            measure = photonic.NumberResolvingMeasurement(cod_len) if circuit.cod[0] == photonic.qmode else qubits.Measure(cod_len)

            diagram = state >> circuit >> measure
            result = diagram.eval()

            amps = result.amplitudes()

    # AbstractBackend
        # not a LO trying to get matrix
    @pytest.mark.parametrize("circuit", CIRCUITS_WITH_DISCARDS_TO_TEST)
    def test_abstract_backend_not_lo(self, circuit):
        with pytest.raises(AssertionError):
            backend = PercevalBackend()
            perceval_state = pcvl.BasicState(get_state(circuit))
            result_perceval = circuit.eval(backend, perceval_state=perceval_state)

    def test_mixed_all_off_diagonal_mass_raises_zero_total(self):
        dm = np.zeros((2, 2, 2), dtype=float)
        dm[0, 0, 1] = 0.6
        dm[1, 1, 0] = 0.4

        box = discopy_tensor.Tensor(dm, discopy_tensor.Dim(1), discopy_tensor.Dim(*dm.shape))
        ev = EvalResult(_tensor=box, output_types=(mode, object()), state_type=StateType.DM)

        with pytest.raises(ValueError):
            _ = ev.prob_dist()