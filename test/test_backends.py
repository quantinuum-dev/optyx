import pytest
from optyx import photonic, qubits, classical
from cotengra import ReusableHyperCompressedOptimizer
from optyx.core.backends import QuimbBackend, PercevalBackend, DiscopyBackend
import numpy as np
import math

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

@pytest.mark.skip(reason="Helper function for testing")
def get_state(circuit):
    if isinstance(circuit.dom[0], photonic.qmode):
        return (1,)*len(circuit.dom)
    elif isinstance(circuit.dom[0], qubits.qubit):
        return qubits.X(0, 1, 0.3)

@pytest.mark.skip(reason="Helper function for testing")
def dict_allclose(d1: dict, d2: dict, *, rel_tol=1e-09, abs_tol=0.0) -> bool:
    if d1.keys() != d2.keys():
        return False
    return all(
        math.isclose(d1[k], d2[k], rel_tol=rel_tol, abs_tol=abs_tol)
        for k in d1
    )

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
            result_exact.amplitudes,
            result_approx.amplitudes,
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

    # check if exact pure gives amps
    @pytest.mark.parametrize("circuit", PURE_CIRCUITS_TO_TEST)
    def test_pure_gives_amps(self, circuit):
        pass

    # check if exact mixed gives density matrix
    @pytest.mark.parametrize("circuit", MIXED_CIRCUITS_TO_TEST)
    def test_mixed_gives_density_matrix(self, circuit):
        pass

class TestPercevalBackend:
    # compare with matrix.probs
    @pytest.mark.parametrize("circuit", PURE_CIRCUITS_TO_TEST)
    def test_pure_circuit(self, circuit):
        pass

class TestDiscopyBackend:
    # compare with matrix.probs
    @pytest.mark.parametrize("circuit", PURE_CIRCUITS_TO_TEST)
    def test_pure_circuit(self, circuit):
        pass

class TestEvalResult:
    # compare circuits with measurements and dangling wires with
    # circuits without measurements and discards
    # for mixed (Discopy + Quimb)
        # discopy and quimb
    @pytest.mark.parametrize("circuit", MIXED_CIRCUITS_TO_TEST)
    def test_mixed_circuit(self, circuit):
        pass

    # check if prob_dist is square of amps for pure circuits
    @pytest.mark.parametrize("circuit", PURE_CIRCUITS_TO_TEST)
    def test_pure_prob_dist_amp(self, circuit):
        pass

    #check three-way agreement
        # pure:
            # prob dist from all for different input states
            # amps from exact quimb and discopy
    @pytest.mark.parametrize("circuit", PURE_CIRCUITS_TO_TEST)
    def test_pure_agreement(self, circuit):
        pass

        # mixed:
            # density matrix, amps, prob dist from discopy and quimb
    @pytest.mark.parametrize("circuit", MIXED_CIRCUITS_TO_TEST)
    def test_mixed_agreement(self, circuit):
        pass


class TestExceptions:
    # EvalResult
    # density matrix/amps/prob from not a state
    @pytest.mark.parametrize("circuit", MIXED_CIRCUITS_TO_TEST)
    def test_eval_result_not_state(self, circuit):
        with pytest.raises(ValueError):
             pass

    # density matrix from perceval eval
    @pytest.mark.parametrize("circuit", PURE_CIRCUITS_TO_TEST)
    def test_eval_result_not_density_matrix(self, circuit):
        with pytest.raises(TypeError):
            pass

        # amps from mixed circuits and prob dist
    @pytest.mark.parametrize("circuit", MIXED_CIRCUITS_TO_TEST)
    def test_eval_result_not_amps(self, circuit):
        with pytest.raises(TypeError):
            pass

    # AbstractBackend
        # not a LO trying to get matrix
    @pytest.mark.parametrize("circuit", MIXED_CIRCUITS_TO_TEST)
    def test_abstract_backend_not_lo(self, circuit):
        with pytest.raises(NotImplementedError):
            pass

    # QuimbBackend
        # eval with some weird backend

#@pytest.mark.parametrize("action, default", CIRCUITS_TO_TEST)
#@pytest.mark.skip(reason="Helper function for testing")