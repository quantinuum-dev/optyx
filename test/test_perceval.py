import pytest

from optyx.zw import *
from optyx.LO import BS, Phase

unitary_circuits = [
	BS >> Phase(1 / 4) @ Id(Mode(1)) >> BS.dagger(),
]
non_unitary_circuits = [
	BS >> Endo(2) @ Id(Mode(1)) >> BS.dagger(),
	Create(1) @ Id(Mode(1)),
	Create(1) @ Id(Mode(1)) >> BS,
	Create(2) @ Id(Mode(1)) >> BS >> Select(2) @ Id(1),
	Id(1) @ Create(1, 1) >> BS @ Id(Mode(1)) >> Id(Mode(2)) @ Select(1),
]


@pytest.mark.parametrize("circuit", unitary_circuits + non_unitary_circuits)
@pytest.mark.parametrize("n_photons", range(1, 4))
def test_perceval_probs_equivalence(circuit: Diagram, n_photons: int):
	qpath_probs = circuit.to_path().prob(n_photons).normalise()
	perceval_probs = circuit.to_path().prob(n_photons, with_perceval=True)
	assert np.isclose(qpath_probs.array, perceval_probs.array).all()
