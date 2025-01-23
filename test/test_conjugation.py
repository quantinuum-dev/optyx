import math

from optyx.channel import *
from optyx import lo, zx, zw
import itertools
import pytest
import numpy as np


@pytest.mark.parametrize("box", [lo.Phase, lo.BBS])
def test_conjugation_LO_1(box):
    gate = box(0.27)
    dom = Ty.from_optyx(gate.dom)
    lhs = Discard(dom)
    rhs = Channel(gate.name, gate) >> Discard(dom)
    lhs_tensor = lhs.double().to_zw().to_tensor().eval().array
    rhs_tensor = rhs.double().to_zw().to_tensor().eval().array
    assert np.allclose(lhs_tensor, rhs_tensor)


@pytest.mark.parametrize("box", [lo.TBS, lo.MZI])
def test_conjugation_LO_2(box):
    gate = box(0.46, 0.54)
    lhs = Discard(qmode ** 2)
    rhs = Channel(gate.name, gate) >> Discard(qmode ** 2)
    lhs_tensor = lhs.double().to_zw().to_tensor().eval().array
    rhs_tensor = rhs.double().to_zw().to_tensor().eval().array
    assert np.allclose(lhs_tensor, rhs_tensor, rtol=1e-01, atol=1e-01)


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

