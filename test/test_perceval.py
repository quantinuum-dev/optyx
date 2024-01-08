import pytest

from optyx.qpath import *

unitary_circuits = [
	BS >> Phase(1 / 4) @ Id(1) >> BS.dagger(),
]
non_unitary_circuits = [
	BS >> Endo(2) @ Id(1) >> BS.dagger(),
	Create(1) @ Id(1),
	Create(1) @ Id(1) >> BS,
	Create(2) @ Id(1) >> BS >> Select(2) @ Id(1),
]


@pytest.mark.parametrize("circuit", unitary_circuits + non_unitary_circuits)
@pytest.mark.parametrize("n_photons", range(1, 4))
def test_perceval_probs_equivalence(circuit: Diagram, n_photons: int):
	qpath_probs = circuit.prob(n_photons).normalise()
	perceval_probs = circuit.prob(n_photons, with_perceval=True)
	assert np.isclose(qpath_probs.array, perceval_probs.array).all()
