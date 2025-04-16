from optyx.channel import Channel
from optyx.lo import BS
from optyx.zw import W, Create, Endo
from optyx.channel import qmode, Measure
import numpy as np
import pytest
import itertools

internal_state_random = np.random.rand(2) + 1j*np.random.rand(2)
internal_state_random = internal_state_random / np.linalg.norm(internal_state_random)

internal_states = [
    [0.5**0.5, 0.5**0.5],
    [1],
    internal_state_random
]

@pytest.mark.parametrize("internal_state", internal_states)
def test_non_distinguishable_photons(internal_state):
    channel_BS = (Channel(
        "BS",
            (
                Create(1, 1, internal_states=(internal_state,
                                             internal_state)) >>
                BS
            )
        ) >> Measure(qmode**2)
    ).inflate(len(internal_state))

    result = channel_BS.double().to_zw().to_tensor(max_dim=3).eval().array

    rounded_result = np.round(result, 6)

    non_zero_dict = {idx: val for idx, val in np.ndenumerate(rounded_result) if val != 0}
    assert non_zero_dict == {(0, 2): (0.5),
                             (2, 0): (0.5)}

internal_states_1 = []
internal_states_2 = []

for i in range(5):
    internal_state_1 = np.random.rand(2) + 1j*np.random.rand(2)
    internal_state_1 = internal_state_1 / np.linalg.norm(internal_state_1)

    internal_states_1.append(internal_state_1)

    internal_state_2 = np.random.rand(2) + 1j*np.random.rand(2)
    internal_state_2 = internal_state_2 / np.linalg.norm(internal_state_2)

    internal_states_2.append(internal_state_2)

@pytest.mark.parametrize("internal_state_1, internal_state_2",
                         itertools.product(internal_states_1,
                                           internal_states_2))
def test_distinguishable_photons(internal_state_1, internal_state_2):
    channel_BS = (Channel(
        "BS",
            (
                Create(1, 1, internal_states=(internal_state_1,
                                             internal_state_2)) >>
                BS
            )
        ) >> Measure(qmode**2)
    ).inflate(len(internal_state_1))

    result = channel_BS.double().to_zw().to_tensor(max_dim=3).eval().array

    rounded_result = np.round(result, 6)

    non_zero_dict = {idx: val for idx, val in np.ndenumerate(rounded_result) if val != 0}
    one_one_prob = non_zero_dict.get((1, 1), 0)
    assert np.allclose(one_one_prob,
                       0.5 - 0.5*np.abs(np.array(internal_state_1).dot(np.array(internal_state_2).conjugate()))**2, 4)