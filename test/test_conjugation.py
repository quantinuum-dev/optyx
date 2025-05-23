import math

from optyx.diagram.channel import *
from optyx.diagram import lo, zx
import pytest
import numpy as np


@pytest.mark.parametrize("box", [lo.Phase, lo.BBS, lo.TBS])
def test_conjugation_LO(box):
    gate = box(0.27)
    dom = Ty.from_optyx(gate.dom)
    lhs = Discard(dom)
    rhs = Channel(gate.name, gate) >> Discard(dom)
    lhs_tensor = lhs.double().to_zw().to_tensor().eval().array
    rhs_tensor = rhs.double().to_zw().to_tensor().eval().array
    assert np.allclose(lhs_tensor, rhs_tensor)

def test_conjugation_MZI():
    gate = lo.MZI(0.27, 0.76)
    dom = qmode ** 2
    lhs = Discard(dom)
    rhs = Channel(gate.name, gate) >> Discard(dom)
    lhs_tensor = lhs.double().to_zw().to_tensor().eval().array
    rhs_tensor = rhs.double().to_zw().to_tensor().eval().array
    assert np.allclose(lhs_tensor, rhs_tensor)


def test_conjugation_LO_Gate():
    hbs_array = (1 / 2) ** (1 / 2) * np.array([[1, 1j], [1j, 1]])
    gate = lo.Gate(hbs_array, 2, 2, "HBS")
    assert np.allclose(gate.conjugate().to_path().array,
                       hbs_array.conjugate())


@pytest.mark.parametrize("box", [zx.Z, zx.X])
def test_conjugation_ZX(box):
    gate = box(1, 1, 0.69)
    lhs = Discard(qubit)
    rhs = Channel(gate.name, gate) >> Discard(qubit)
    lhs_tensor = lhs.double().to_zw().to_tensor().eval().array
    rhs_tensor = rhs.double().to_zw().to_tensor().eval().array
    assert np.allclose(lhs_tensor, rhs_tensor)


def test_conjugation_ZW():
    diagram = lo.TBS(0.24, 0.86).to_zw()
    lhs = Discard(qmode ** 2)
    rhs = Channel('diagram', diagram) >> Discard(qmode ** 2)
    lhs_tensor = lhs.double().to_zw().to_tensor().eval().array
    rhs_tensor = rhs.double().to_zw().to_tensor().eval().array
    assert np.allclose(lhs_tensor, rhs_tensor)

