import random

import numpy as np
from pytest import raises, fixture

from discopy.quantum.gates import CRz, CRx, CU1, Ket, Rx, H
from discopy import quantum
from optyx.core.zx import *
from optyx import qubits
from optyx.core.diagram import Diagram, Bit

@fixture
def random_had_cnot_diagram():
    def _random_had_cnot_diagram(qubits_, depth, p_had=0.5):
        random.seed(0)
        c = Ket(*(qubits_ * [0]))
        for _ in range(depth):
            r = random.random()
            if r > p_had:
                c = c.H(random.randrange(qubits_))
            else:
                tgt = random.randrange(qubits_)
                while True:
                    ctrl = random.randrange(qubits_)
                    if ctrl != tgt:
                        break
                c = c.CX(tgt, ctrl)
        return c
    return _random_had_cnot_diagram


def test_Diagram():
    bialgebra = Z(1, 2) @ Z(1, 2) >> Bit(1) @ SWAP @ Bit(1) >> X(2, 1) @ X(2, 1)
    assert str(bialgebra) == "Z(1, 2) @ bit >> bit @ bit @ Z(1, 2) " \
                             ">> bit @ Swap(bit, bit) @ bit " \
                             ">> X(2, 1) @ bit @ bit >> bit @ X(2, 1)"


def test_Spider():
    assert str(Z(1, 2, 3)) == "Z(1, 2, 3)"
    assert str(X(1, 2, 3)) == "X(1, 2, 3)"
    for spider in [Z, X]:
        assert spider(1, 2, 3).phase == 3
        assert spider(1, 2, 3j).dagger() == spider(2, 1, -3j)


def test_H():
    assert str(H) == "H"
    assert np.allclose(H[::-1].to_tensor().eval().array, H.to_tensor().eval().array)


def test_Sum():
    assert Z(1, 1) + Z(1, 1) >> Z(1, 1) == sum(2 * [Z(1, 1) >> Z(1, 1)])


def test_scalar():
    assert scalar(1j)[::-1] == scalar(-1j)


def test_subs():
    from sympy.abc import phi, psi
    assert Z(3, 2, phi).subs(phi, 1) == Z(3, 2, 1)
    assert scalar(phi).subs(phi, psi) == scalar(psi)


def test_grad():
    from sympy.abc import phi, psi
    from math import pi
    assert not scalar(phi).grad(psi) and scalar(phi).grad(phi) == scalar(1)
    assert not Z(1, 1, phi).grad(psi)
    assert Z(1, 1, phi).grad(phi) == scalar(pi) @ Z(1, 1, phi + .5)
    assert (Z(1, 1, phi / 2) >> Z(1, 1, phi + 1)).grad(phi)\
        == (scalar(pi / 2) @ Z(1, 1, phi / 2 + .5) >> Z(1, 1, phi + 1))\
           + (Z(1, 1, phi / 2) >> scalar(pi) @ Bit(1) >> Z(1, 1, phi + 1.5))


def test_to_pyzx_errors():
    with raises(NotImplementedError):
        Diagram.to_pyzx(quantum.H)


def test_to_pyzx():
    assert ZXDiagram.from_pyzx(Z(0, 2).to_pyzx()) == Z(0, 2) >> SWAP


def test_to_pyzx_scalar():
    # Test that a scalar is translated to the corresponding pyzx object.
    k = np.exp(np.pi / 4 * 1j)
    m = (scalar(k) @ scalar(k) @ Bit(1)).to_pyzx().to_matrix()
    m = np.linalg.norm(m / 1j - np.eye(2))
    assert np.isclose(m, 0)


def test_from_pyzx_errors():
    bialgebra = Z(1, 2) @ Z(1, 2) >> Bit(1) @ SWAP @ Bit(1) >> X(2, 1) @ X(2, 1)
    graph = bialgebra.to_pyzx()
    graph.set_inputs(())
    graph.set_outputs(())
    with raises(ValueError):  # missing_boundary
        ZXDiagram.from_pyzx(graph)
    graph.auto_detect_io()
    graph.set_inputs(graph.inputs() + graph.outputs())
    with raises(ValueError):  # duplicate_boundary
        ZXDiagram.from_pyzx(graph)


def _std_basis_v(*c):
    v = np.zeros(2**len(c), dtype=complex)
    v[np.sum((np.array(c) != 0) * 2**np.arange(len(c)))] = 1
    return np.expand_dims(v, -1)


def test_circuit2zx():
    circuit = Ket(0, 0) >> quantum.H @ Rx(0) >> CRz(0) >> CRx(0) >> CU1(0)
    assert qubits.Circuit(circuit).get_kraus() == Diagram.decode(
        dom=Bit(0), boxes_and_offsets=zip([
            X(0, 1), X(0, 1), scalar(0.5), H, X(1, 1),
            Z(1, 2), Z(1, 2), X(2, 1), Z(1, 0), scalar(2 ** 0.5),
            X(1, 2), X(1, 2), Z(2, 1), X(1, 0), scalar(2 ** 0.5),
            Z(1, 2), Z(1, 2), X(2, 1), Z(1, 0)],
            [0, 1, 2, 0, 1, 0, 2, 1, 1, 2, 0, 2, 1, 1, 2, 0, 2, 1, 1]))

    # Verify XYZ=iI
    circuit = quantum.Z >> quantum.Y >> quantum.X
    t = qubits.Circuit(circuit).get_kraus()
    t = t.to_pyzx().to_matrix() - 1j * np.eye(2)
    assert np.isclose(np.linalg.norm(t), 0)

    # Check scalar translation
    circuit = quantum.X >> quantum.X @ quantum.scalar(1j)
    t = qubits.Circuit(circuit).to_pyzx().to_matrix()
    assert np.isclose(np.linalg.norm(t - 1j * np.eye(2)), 0)

    # with raises(NotImplementedError):
    #     circuit2zx(quantum.scalar(1, is_mixed=True))

    circuit = Ket(0)
    t = qubits.Circuit(circuit).to_pyzx().to_matrix() - _std_basis_v(0)
    assert np.isclose(np.linalg.norm(t), 0)
    circuit = Ket(0, 0)
    t = qubits.Circuit(circuit).to_pyzx().to_matrix() - _std_basis_v(0, 0)
    assert np.isclose(np.linalg.norm(t), 0)

    circuit = quantum.Id(3).CX(0, 2)
    assert (qubits.Circuit(circuit).get_kraus()
            == Diagram.decode(
                dom=Bit(3),
                boxes_and_offsets=zip(
                    [SWAP, Z(1, 2), X(2, 1), scalar(2 ** 0.5), SWAP],
                    [1, 0, 1, 2, 1])))