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