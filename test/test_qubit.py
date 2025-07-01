import pyzx
from optyx import qubit
from pytket import Circuit
from pytket.extensions.qiskit import AerBackend
from pytket.utils import probs_from_counts
from optyx.core import channel
from optyx.core import zx
import numpy as np

def test_pyzx():
    c = pyzx.Circuit(3)
    c.add_gate("TOF", 0, 1, 2)
    g = c.to_basic_gates().to_graph()
    c1 = pyzx.extract_circuit(g.copy())
    c2 = pyzx.extract_circuit(qubit.Circuit(g).to_pyzx().copy())
    assert c1.verify_equality(c2)

    c1 = pyzx.extract_circuit(g.copy())
    c2 = pyzx.extract_circuit(channel.Diagram.from_pyzx(g).to_pyzx().copy())
    assert c1.verify_equality(c2)

def test_tket_discopy():
    ghz_circ = Circuit(3)
    ghz_circ.H(0)
    ghz_circ.CX(0, 1)
    ghz_circ.CX(1, 2)
    ghz_circ.measure_all()

    backend = AerBackend()
    compiled_circ = backend.get_compiled_circuit(ghz_circ)
    handle = backend.process_circuit(compiled_circ, n_shots=200000)
    counts = backend.get_result(handle).get_counts()
    tket_probs = probs_from_counts({key: np.round(v, 2) for key, v in probs_from_counts(counts).items()})

    res = ((qubit.Circuit(ghz_circ)).double().to_tensor().to_quimb()^...).data

    rounded_result = np.round(res, 6)

    non_zero_dict = {idx: val for idx, val in np.ndenumerate(rounded_result) if val != 0}

    assert tket_probs == non_zero_dict

    res = (channel.Diagram.from_tket(ghz_circ).double().to_tensor().to_quimb()^...).data

    rounded_result = np.round(res, 6)

    non_zero_dict = {idx: val for idx, val in np.ndenumerate(rounded_result) if val != 0}

    assert tket_probs == non_zero_dict

def test_zx():
    circuit = qubit.Z(1, 2) >> qubit.H() @ qubit.H()
    assert qubit.Circuit(circuit) == circuit

def test_pure_double_kraus():
    c = pyzx.Circuit(3)
    c.add_gate("TOF", 0, 1, 2)
    g = c.to_basic_gates().to_graph()
    assert qubit.Circuit(g).double() == qubit.Circuit(g).double()

    assert qubit.Circuit(g).is_pure == qubit.Circuit(g).is_pure

    assert qubit.Circuit(g).get_kraus() == qubit.Circuit(g).get_kraus()

def test_to_dual_rail():
    circuit = qubit.Z(1, 2) >> qubit.H() @ qubit.H()
    dr_1 = qubit.Circuit(circuit).to_dual_rail().get_kraus()
    dr_2 = zx.zx2path(circuit.get_kraus())
    assert dr_1 == dr_2

def test_discard_qubits():
    a = (qubit.Discard(2).double().to_tensor().to_quimb() ^ ...).data
    b = ((qubit.Z(1, 1)**2 >> qubit.Discard(2)).double().to_tensor().to_quimb() ^ ...).data
    assert np.allclose(a, b)

def test_bit_flip_error():
    prob = 0.43
    a = (qubit.BitFlipError(prob).get_kraus().to_tensor().to_quimb() ^ ...).data
    b = zx.X(1, 2) >> zx.Id(1) @ zx.ZBox(
            1, 1, np.sqrt((1 - prob) / prob)
        ) @ zx.scalar(np.sqrt(prob * 2))
    b = (b.to_tensor().to_quimb() ^ ...).data

    assert np.allclose(a, b)

def test_dephasingerror():
    prob = 0.43
    a = (qubit.DephasingError(prob).get_kraus().to_tensor().to_quimb() ^ ...).data
    b = (
            zx.H
            >> zx.X(1, 2)
            >> zx.H
            @ zx.ZBox(1, 1, np.sqrt((1 - prob) / prob))
            @ zx.scalar(np.sqrt(prob * 2))
        )
    b = (b.to_tensor().to_quimb() ^ ...).data

    assert np.allclose(a, b)

def test_ket():
    from optyx.core import diagram
    a = (qubit.Ket(1).get_kraus().to_tensor().to_quimb() ^ ...).data
    b = zx.X(0, 1, 0.5) @ diagram.Scalar(1 / np.sqrt(2))

    b = (b.to_tensor().to_quimb() ^ ...).data
    assert np.allclose(a, b)

def test_bra():
    from optyx.core import diagram
    a = (qubit.Bra(1).get_kraus().to_tensor().to_quimb() ^ ...).data
    b = zx.X(1, 0, 0.5) @ diagram.Scalar(1 / np.sqrt(2))
    b = (b.to_tensor().to_quimb() ^ ...).data
    assert np.allclose(a, b)

def test_to_tket():
    circ = qubit.X(1, 2) @ channel.qubit >> channel.qubit @ qubit.Z(2, 1) @ qubit.Scalar(2**0.5)

    tket_circ = circ.to_tket()
    tket_circ.measure_all()
    backend = AerBackend()
    compiled_circ = backend.get_compiled_circuit(tket_circ)
    handle = backend.process_circuit(compiled_circ, n_shots=200000)
    counts = backend.get_result(handle).get_counts()
    tket_probs = probs_from_counts({key: np.round(v, 2) for key, v in probs_from_counts(counts).items()})

    circ_meas_prep = qubit.Ket(0) @ qubit.Ket(0) >> circ >> qubit.Measure(2)

    res = ((circ_meas_prep.double().to_tensor().to_quimb()^...).data)

    rounded_result = np.round(res, 6)

    non_zero_dict = {idx: val for idx, val in np.ndenumerate(rounded_result) if val != 0}

    assert tket_probs == non_zero_dict