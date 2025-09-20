import math

from optyx.core.channel import *
from optyx.core import zx
from optyx import photonic
import pytest
import numpy as np


@pytest.mark.parametrize("box", [photonic.Phase, photonic.BBS, photonic.TBS])
def test_conjugation_LO(box):
    gate = box(0.27)
    dom = gate.dom
    lhs = Discard(dom)
    rhs = gate >> Discard(dom)
    lhs_tensor = lhs.double().to_tensor().eval().array
    rhs_tensor = rhs.double().to_tensor().eval().array
    assert np.allclose(lhs_tensor, rhs_tensor)

def test_conjugation_MZI():
    gate = photonic.MZI(0.27, 0.76)
    dom = qmode ** 2
    lhs = Discard(dom)
    rhs = gate >> Discard(dom)
    lhs_tensor = lhs.double().to_tensor().eval().array
    rhs_tensor = rhs.double().to_tensor().eval().array
    assert np.allclose(lhs_tensor, rhs_tensor)


def test_conjugation_LO_Gate():
    hbs_array = (1 / 2) ** (1 / 2) * np.array([[1, 1j], [1j, 1]])
    gate = photonic.Gate(hbs_array, 2, 2, "HBS").get_kraus()
    assert np.allclose(gate.conjugate().to_path().array,
                       hbs_array.conjugate())

@pytest.mark.parametrize("box", [photonic.Phase, photonic.BBS, photonic.TBS])
def test_conjugation_LO_channel(box):
    gate = box(0.27)
    hbs_array = gate.array
    assert np.allclose(gate.conjugate().to_path().array,
                       hbs_array.conjugate())


def test_conjugation_MZI_channel():
    gate = photonic.MZI(0.27, 0.76)
    hbs_array = gate.array
    assert np.allclose(gate.conjugate().to_path().array,
                       hbs_array.conjugate())


@pytest.mark.parametrize("box", [zx.Z, zx.X])
def test_conjugation_ZX(box):
    gate = box(1, 1, 0.69)
    lhs = Discard(qubit)
    rhs = Channel("Gate", gate) >> Discard(qubit)
    lhs_tensor = lhs.double().to_tensor().eval().array
    rhs_tensor = rhs.double().to_tensor().eval().array
    assert np.allclose(lhs_tensor, rhs_tensor)


def test_conjugation_ZW():
    diagram = photonic.TBS(0.24, 0.86).get_kraus()
    lhs = Discard(qmode ** 2)
    rhs = Channel('diagram', diagram) >> Discard(qmode ** 2)
    lhs_tensor = lhs.double().to_tensor().eval().array
    rhs_tensor = rhs.double().to_tensor().eval().array
    assert np.allclose(lhs_tensor, rhs_tensor)
