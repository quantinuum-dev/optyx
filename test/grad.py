from optyx.qpath import *
from sympy.abc import psi
import numpy as np

from itertools import product

import pytest

param_circuits = [
    Phase(psi),
    Phase(1 / 3 * psi**2),
    BS >> Phase(psi) @ Id(1) >> BS,
]

values = [x * 0.123 for x in range(10)]


@pytest.mark.parametrize("circ, value", product(param_circuits, values))
def test_daggers_cancel(circ, value):
    d = circ >> circ.dagger()
    out = d.grad(psi).subs((psi, value)).eval(2).array
    assert np.allclose(out, np.array([0.0]))
