
from __future__ import annotations

import itertools
from typing import Dict, Tuple

import numpy as np
import pytest
from optyx.core.channel import Channel, qmode, Measure
from optyx.photonic import BS
from optyx.core.zw import Create
from optyx.core.diagram import DualRail, Scalar
from optyx.core.zx import Z, X


RNG = np.random.default_rng(seed=2025)

def random_internal_state(dim: int = 2) -> np.ndarray:
    """Return a normalised random complex vector of shape (dim,)."""
    v = RNG.random(dim) + 1j * RNG.random(dim)
    return v / np.linalg.norm(v)


def non_zero_entries(array: np.ndarray) -> Dict[Tuple[int, ...], complex]:
    """Sparse view of the non‑zero entries of an array."""
    return {idx: val for idx, val in np.ndenumerate(array) if val != 0}

INTERNAL_STATES = [
    np.array([1, 1]) / np.sqrt(2),
    np.array([1]),
    random_internal_state(),
]

DISTINGUISHABLE_STATE_POOL = [random_internal_state() for _ in range(5)]
DISTINGUISHABLE_PAIRS = list(
    itertools.product(DISTINGUISHABLE_STATE_POOL, DISTINGUISHABLE_STATE_POOL)
)

QUBIT_STATES = [
    Z(0, 1) @ Scalar(1 / np.sqrt(2)),
    Z(0, 1, 0.3) @ Scalar(1 / np.sqrt(2)),
    X(0, 1) @ Scalar(1 / np.sqrt(2)),
    X(0, 1, 0.3) @ Scalar(1 / np.sqrt(2)),
]


@pytest.mark.parametrize("internal_state", INTERNAL_STATES)
def test_beamsplitter_non_distinguishable(internal_state: np.ndarray) -> None:
    """
    Hong–Ou–Mandel bunching for *indistinguishable* photons:
    probabilities P_{02}=P_{20}=0.5 and all others vanish.
    """
    channel_bs = (
        Channel(
            "BS",
            Create(1, 1, internal_states=(internal_state, internal_state)) >> BS.get_kraus(),
        )
        >> Measure(qmode**2)
    ).inflate(len(internal_state))

    probs = (
        channel_bs.double()
        .to_tensor(max_dim=3)
        .eval()
        .array
    )

    nz = non_zero_entries(np.round(probs, 6))
    assert nz == {(0, 2): 0.5, (2, 0): 0.5}


@pytest.mark.parametrize("state1,state2", DISTINGUISHABLE_PAIRS)
def test_beamsplitter_distinguishable(state1: np.ndarray, state2: np.ndarray) -> None:
    channel_bs = (
        Channel(
            "BS",
            Create(1, 1, internal_states=(state1, state2)) >> BS.get_kraus(),
        )
        >> Measure(qmode**2)
    ).inflate(len(state1))

    probs = (
        channel_bs.double()
        .to_tensor(max_dim=3)
        .eval()
        .array
    )

    nz = non_zero_entries(np.round(probs, 6))
    observed = nz.get((1, 1), 0)

    overlap = np.abs(state1 @ state2.conj()) ** 2
    expected = 0.5 - 0.5 * overlap
    assert np.isclose(observed, expected, atol=1e-4)


@pytest.mark.parametrize(
    "qubit_state,internal_state", list(itertools.product(QUBIT_STATES, INTERNAL_STATES))
)
def test_dualrail_identity(qubit_state, internal_state) -> None:
    """Dual‑rail Encode . Decode acts as identity on the logical qubit."""
    encoded = qubit_state >> DualRail(internal_state=internal_state).inflate(len(internal_state))

    dr_drdagger = (
        encoded
        >> DualRail(internal_state=internal_state).inflate(len(internal_state)).dagger()
    ).to_tensor(max_dim=2).eval().array

    expected = qubit_state.to_tensor().eval().array
    assert np.allclose(dr_drdagger, expected)


def test_encode_n_qubits() -> None:
    """
    Test for Encode: three logical qubits in two photonic modes.
    Ensures the overall probability mass remains 1.
    """
    from optyx.core.channel import CQMap, mode, Encode

    create = CQMap(
        "create",
        Create(1, 1),
        dom=mode ** 0,
        cod=mode ** 2,
    )
    create_channel = Channel("create", Create(1, internal_states=([1, 0],)))

    result = (
        create
        @ create_channel
        >> Encode(mode ** 2, ([1 / np.sqrt(2), 1 / np.sqrt(2)], [1, 0]))
        @ Encode(qmode)
        >> Measure(qmode ** 3)
    ).inflate(2).double().to_tensor().eval().array

    nz = non_zero_entries(np.round(result, 3))
    assert nz[(1, 1, 1)] == 1.0


def test_photon_threshold_detector():
    from optyx.core.diagram import PhotonThresholdDetector

    d_1 = Create(1, internal_states=([1, 0])) >> PhotonThresholdDetector()
    d_2 = X(1, 0, 0.5) @ Scalar(0.5**0.5)

    arr_1 = d_1.inflate(2).to_tensor(max_dim=3).eval().array
    arr_2 = d_2.to_tensor(max_dim=3).eval().array

    assert np.allclose(arr_1, arr_2)
