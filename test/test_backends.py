import pytest
from optyx import photonic, qubits, classical
from cotengra import ReusableHyperCompressedOptimizer
from optyx.core.backends import QuimbBackend, PercevalBackend, DiscopyBackend
import numpy as np
import math
from itertools import chain
import perceval as pcvl

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