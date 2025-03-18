import pytest

from optyx.channel import *
import numpy as np

bell_density_re = np.array([
    [0.012, 0.014, 0.014, 0.000],
    [0.014, 0.508, 0.475, 0.008],
    [0.014, 0.475, 0.479, 0.009],
    [0.000, 0.008, 0.009, 0.000]
])
bell_density_im = np.exp(1j * np.pi * np.array([
    [0.000, -1.850, -1.825, -0.985],
    [1.850, 0.000, -0.002, -0.902],
    [1.825, 0.002, 0.000, -0.931],
    [0.985, 0.902, 0.931, 0.000]
]))
bell_density = np.multiply(bell_density_re, bell_density_im)


def test_CQMap():
    X = Channel("X", zx.X(1, 1, 0.5))
    bell = optyx.Box(name="Bell", dom=optyx.bit ** 2, cod=optyx.bit ** 2, array=bell_density)
    bell = optyx.Spider(0, 2, typ=optyx.bit) >> optyx.Id(optyx.bit) @ optyx.Spider(0, 2, typ=optyx.bit) @ optyx.Id(optyx.bit) >> optyx.Diagram.permutation([0,1,3,2], optyx.bit**4) >> optyx.Id(optyx.bit ** 2) @ bell >> optyx.Diagram.permutation([0,2,1,3], optyx.bit**4)

    Noisy_bell = CQMap('Physical Bell', bell @ optyx.Scalar(1/0.999), dom=Ty(), cod=qubit ** 2)
    Perfect_Bell_Effect = Channel("Perfect Bell Effect", optyx.Spider(2,0,typ=optyx.bit) @ optyx.Scalar(1 / np.sqrt(2)))

    CALCULATED_FIDELITY = (Noisy_bell >> Circuit.id(qubit) @ X >> Perfect_Bell_Effect).double().to_tensor().eval().array.real
    REAL_FIDELITY = .96898

    assert np.isclose(CALCULATED_FIDELITY, REAL_FIDELITY, rtol=1e-3)
