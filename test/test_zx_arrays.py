import math

from optyx.zx import *
import itertools
import pytest

n_legs_in = list(range(1, 5))
n_legs_out = list(range(0, 5))
phase = [0, 0.25, 0.5, 0.25, 1]

combinations = list(itertools.product(n_legs_in, n_legs_out, phase))


@pytest.mark.parametrize("n_legs_in, n_legs_out, phase", combinations)
def test_Z(n_legs_in, n_legs_out, phase):
    z = Z(n_legs_in, n_legs_out, phase)
    assert np.allclose(z.array.flatten(), z.to_pyzx().to_tensor().flatten())

@pytest.mark.parametrize("n_legs_in, n_legs_out, phase", combinations)
def test_Z_tensor(n_legs_in, n_legs_out, phase):
    z = Z(n_legs_in, n_legs_out, phase)
    assert np.allclose(z.to_tensor().eval().array.flatten(),
                       z.to_pyzx().to_tensor().flatten())

@pytest.mark.parametrize("n_legs_in, n_legs_out, phase", combinations)
def test_X(n_legs_in, n_legs_out, phase):
    x = X(n_legs_in, n_legs_out, phase)
    assert np.allclose(x.array.flatten(), x.to_pyzx().to_tensor().flatten())

@pytest.mark.parametrize("n_legs_in, n_legs_out, phase", combinations)
def test_X_tensor(n_legs_in, n_legs_out, phase):
    x = X(n_legs_in, n_legs_out, phase)
    assert np.allclose(x.to_tensor().eval().array.flatten(),
                       x.to_pyzx().to_tensor().flatten())

scalar = [0, 0.5, 1]


@pytest.mark.parametrize("scalar", scalar)
def test_scalar(scalar):
    s = Scalar(scalar)
    assert np.allclose(s.array.flatten(), s.to_pyzx().to_tensor().flatten())