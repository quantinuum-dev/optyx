import pytest

from optyx.core.zw import *
from optyx.photonic import (
    BS as BS_,
	BBS as BBS_,
	Phase as Phase_,
)
from optyx.core.diagram import Mode, Diagram
import numpy as np

BS = BS_.get_kraus()
BBS = lambda theta: BBS_(theta).get_kraus()
Phase = lambda theta: Phase_(theta).get_kraus()

unitary_circuits = [
	BS >> Phase(1 / 4) @ Id(Mode(1)) >> BS.dagger(),
	BS @ Id(1) >> Id(1) @ BS,
]
non_unitary_circuits = [
	BS >> Endo(2) @ Id(Mode(1)) >> BS.dagger(),
	Create(1) @ Id(Mode(1)),
	Id(Mode(3)) >> Id(1) @ BS,
	Id(Mode(2)) @ Create(1) @ Id(Mode(2)),
	Id(Mode(1)) @ Create(1) @ Id(Mode(2)) >> Id(2) @ BBS(0.3),
	Create(2) @ Id(Mode(1)) >> BS >> Select(2) @ Id(1),
	Id(1) @ Create(1, 1) >> BS @ Id(Mode(1)) >> Id(Mode(2)) @ Select(1),
	Create(1) @ Id(Mode(2)) >> BS @ Id(Mode(1)) >> Id(Mode(1)) @ BS ,
	Id(1) @ Create(2, 2, 1) >> BBS(0.3) @ BBS(0.3) >> Id(1) @ BBS(0.3) @ Id(1)
]

@pytest.mark.parametrize("circuit", unitary_circuits + non_unitary_circuits)
@pytest.mark.parametrize("n_photons", range(1,2))
def test_perceval_probs_equivalence(circuit: Diagram, n_photons: int):
	qpath_probs = circuit.to_path().prob(n_photons).normalise()
	perceval_probs = circuit.to_path().prob(n_photons, with_perceval=True)
	assert np.isclose(qpath_probs.array, perceval_probs.array).all()
