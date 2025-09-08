import pytest

from optyx.core.channel import *
from optyx.core import diagram, zx
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
    bell = diagram.Box(name="Bell", dom=diagram.bit ** 2, cod=diagram.bit ** 2, array=bell_density)
    bell = diagram.Spider(0, 2, typ=diagram.bit) >> diagram.Id(diagram.bit) @ diagram.Spider(0, 2, typ=diagram.bit) @ diagram.Id(diagram.bit) >> diagram.Diagram.permutation([0,1,3,2], diagram.bit**4) >> diagram.Id(diagram.bit ** 2) @ bell >> diagram.Diagram.permutation([0,2,1,3], diagram.bit**4)

    Noisy_bell = CQMap('Physical Bell', bell @ diagram.Scalar(1/0.999), dom=Ty(), cod=qubit ** 2)
    Perfect_Bell_Effect = Channel("Perfect Bell Effect", diagram.Spider(2,0,typ=diagram.bit) @ diagram.Scalar(1 / np.sqrt(2)))

    CALCULATED_FIDELITY = (Noisy_bell >> Diagram.id(qubit) @ X >> Perfect_Bell_Effect).double().to_tensor().eval().array.real
    REAL_FIDELITY = .96898
    print(CALCULATED_FIDELITY)
    print(REAL_FIDELITY)
    assert np.isclose(CALCULATED_FIDELITY, REAL_FIDELITY, rtol=1e-3)


def test_from_bosonic_op():
    from optyx.core.channel import Diagram

    matrix = [
        [0, 1],
        [1, 0]
    ]

    n = len(matrix)
    assert len(matrix[0]) == n

    terms = []

    for i in range(n):
        for j in range(n):
            term = Diagram.from_bosonic_operator(
                n_modes=n,
                operators=[(i, False), (j, True)],
                scalar=matrix[i][j]
            )
            terms.append(term)

    hamiltonian = Diagram.sum_factory(terms)

    sum_1 = np.sign(hamiltonian.get_kraus().to_tensor().eval().array)
    sum_2 = np.sign(hamiltonian.eval().tensor.array) # bug in discopy.tensor.Tensor.eval() for Sums (?)

    assert np.allclose(sum_2, sum_1)