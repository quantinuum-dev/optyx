import pytest

from optyx.zw import *
from optyx.circuit import BS, BBS, Phase
import numpy as np

unitary_circuits = [
	# BS >> Phase(1 / 4) @ Id(Mode(1)) >> BS.dagger(),
	# BS @ Id(1) >> Id(1) @ BS,
]
non_unitary_circuits = [
	# BS >> Endo(2) @ Id(Mode(1)) >> BS.dagger(),
	# Create(1) @ Id(Mode(1)),
	# Create(1) @ Id(Mode(1)) >> BS,
	# Create(2) @ Id(Mode(1)) >> BS >> Select(2) @ Id(1),
	# Id(1) @ Create(1, 1) >> BS @ Id(Mode(1)) >> Id(Mode(2)) @ Select(1),
	Create(1) @ Id(Mode(2)) >> BS @ Id(Mode(1)) >> Id(Mode(1)) @ BS ,
	# Id(1) @ Create(2, 2, 1) >> BBS(0.3) @ BBS(0.3) >> Id(1) @ BBS(0.3) @ Id(1)
]


@pytest.mark.parametrize("circuit", unitary_circuits + non_unitary_circuits)
@pytest.mark.parametrize("n_photons", range(1,2))
def test_perceval_probs_equivalence(circuit: Diagram, n_photons: int):
	np.set_printoptions(3)
	print()
	qpath_probs = circuit.to_path().prob(n_photons).normalise()
	print(qpath_probs)
	perceval_probs = circuit.to_path().prob(n_photons, with_perceval=True)
	print()
	assert np.isclose(qpath_probs.array, perceval_probs.array).all()
