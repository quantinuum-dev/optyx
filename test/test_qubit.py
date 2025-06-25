import pyzx
from optyx import qubit
from pytket import Circuit
from pytket.extensions.qiskit import AerBackend
from pytket.utils import probs_from_counts
import numpy as np

def test_pyzx():
    c = pyzx.Circuit(3)
    c.add_gate("TOF", 0, 1, 2)
    g = c.to_basic_gates().to_graph()
    c1 = pyzx.extract_circuit(g.copy())
    c2 = pyzx.extract_circuit(qubit.Circuit(g)._to_optyx().to_pyzx().copy())
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

    res = ((qubit.Circuit(ghz_circ)._to_optyx()).double().to_tensor().to_quimb()^...).data

    rounded_result = np.round(res, 6)

    non_zero_dict = {idx: val for idx, val in np.ndenumerate(rounded_result) if val != 0}

    assert tket_probs == non_zero_dict

def test_zx():
    circuit = qubit.Z(1, 2) >> qubit.H() @ qubit.H()
    assert qubit.Circuit(circuit)._to_optyx() == circuit

def test_pure_double_kraus():
    c = pyzx.Circuit(3)
    c.add_gate("TOF", 0, 1, 2)
    g = c.to_basic_gates().to_graph()
    assert qubit.Circuit(g)._to_optyx().double() == qubit.Circuit(g).double()

    assert qubit.Circuit(g)._to_optyx().is_pure == qubit.Circuit(g).is_pure

    assert qubit.Circuit(g)._to_optyx().get_kraus() == qubit.Circuit(g).get_kraus()

def test_to_dual_rail():
    from optyx.core import zx
    circuit = qubit.Z(1, 2) >> qubit.H() @ qubit.H()
    dr_1 = qubit.Circuit(circuit).to_dual_rail().get_kraus()
    dr_2 = zx.zx2path(circuit.get_kraus())
    assert dr_1 == dr_2

def test_discard_qubits():
    a = (qubit.DiscardQubits(2).double().to_tensor().to_quimb() ^ ...).data
    b = ((qubit.Z(1, 1)**2 >> qubit.DiscardQubits(2)).double().to_tensor().to_quimb() ^ ...).data
    assert np.allclose(a, b)

def test_bit_flip_error():
    from optyx.core import zx
    prob = 0.43
    a = (qubit.BitFlipError(prob).get_kraus().to_tensor().to_quimb() ^ ...).data
    b = zx.X(1, 2) >> zx.Id(1) @ zx.ZBox(
            1, 1, np.sqrt((1 - prob) / prob)
        ) @ zx.scalar(np.sqrt(prob * 2))
    b = (b.to_tensor().to_quimb() ^ ...).data

    assert np.allclose(a, b)

def test_dephasingerror():
    from optyx.core import zx
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
    from optyx.core import zx, diagram
    a = (qubit.Ket(1).get_kraus().to_tensor().to_quimb() ^ ...).data
    b = zx.X(0, 1, 0.5) @ diagram.Scalar(1 / np.sqrt(2))

    b = (b.to_tensor().to_quimb() ^ ...).data
    assert np.allclose(a, b)

def test_bra():
    from optyx.core import zx, diagram
    a = (qubit.Bra(1).get_kraus().to_tensor().to_quimb() ^ ...).data
    b = zx.X(1, 0, 0.5) @ diagram.Scalar(1 / np.sqrt(2))
    b = (b.to_tensor().to_quimb() ^ ...).data
    assert np.allclose(a, b)