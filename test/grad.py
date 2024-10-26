from optyx.zw import Id, Endo, Scalar, Diagram, Mode
from optyx.circuit import Phase, BS, TBS, MZI
from sympy.abc import psi, phi, theta
import numpy as np

from itertools import product

import pytest

param_circuits = [
    Phase(psi),
    BS >> Phase(psi) @ Id(Mode(1)) >> BS,
    Phase(3 * psi ** 3),
    TBS(psi),
    MZI(psi, 0.123),
    MZI(0.123, psi),
]

values = [x * 0.123 for x in range(10)]


@pytest.mark.parametrize("circ, value", product(param_circuits, values))
def test_daggers_cancel(circ, value):
    d = circ >> circ.dagger()
    out = d.grad(psi).subs((psi, value)).eval(2).array
    assert np.allclose(out, np.zeros(shape=out.shape))


@pytest.mark.parametrize("circ", param_circuits)
def test_zero_grad(circ):
    assert circ.grad(theta) == Diagram.sum_factory((), circ.dom, circ.cod)
