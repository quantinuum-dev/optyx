from optyx.channel import Channel
from optyx.lo import BS
from optyx.zw import W, Create, Endo
from optyx.optyx import DualRail, Scalar
from optyx.zx import Z, X
from optyx.feed_forward.classical_arithmetic import Add
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

qubit_states = [
    Z(0, 1) @ Scalar(0.5**0.5),
    Z(0, 1, 0.3) @ Scalar(0.5**0.5),
    X(0, 1) @ Scalar(0.5**0.5),
    X(0, 1, 0.3) @ Scalar(0.5**0.5),
]

internal_states = [
    [0.5**0.5, 0.5**0.5],
    [1],
    internal_state_random
]

@pytest.mark.parametrize("qubit_state, internal_state",
                         itertools.product(qubit_states,
                                           internal_states))
def test_dual_rail(qubit_state, internal_state):
    d = qubit_state >> DualRail(internal_state=internal_state).inflate(len(internal_state))

    qubit_array = qubit_state.to_tensor().eval().array
    dual_rail_array = (d >> DualRail(internal_state=internal_state).inflate(len(internal_state)).dagger()).to_tensor().eval().array
    assert np.allclose(qubit_array, dual_rail_array, 4)

    dual_rail_array = (
        qubit_state >>
        DualRail(internal_state=internal_state) >>
        DualRail(internal_state=internal_state).dagger()
    )

    dual_rail_array_channel = Channel(
        "DualRail",
        dual_rail_array
    )

    qubit_state_channel = Channel(
        "Qubit",
        qubit_state
    )

    dual_rail_array = dual_rail_array.inflate(len(internal_state)).to_tensor(max_dim=2).eval().array
    assert np.allclose(dual_rail_array, qubit_array, 4)

    dual_rail_array_channel = dual_rail_array_channel.inflate(len(internal_state)).double().to_tensor(max_dim=2).eval().array
    qubit_state_channel = qubit_state_channel.inflate(len(internal_state)).double().to_tensor(max_dim=2).eval().array
    assert np.allclose(dual_rail_array_channel, qubit_state_channel, 4)

    result = (d >> Add(len(internal_state)) @ Add(len(internal_state))).to_zw().to_tensor(max_dim=2).eval().array
    rounded_result = np.round(result, 6)
    non_zero_dict = {idx: (val if val != 0 else 0) for idx, val in np.ndenumerate(rounded_result)}
    s_1 = np.sum(list(non_zero_dict.values()))
    s_2 = np.sum(qubit_array)
    assert np.allclose(non_zero_dict[(1, 0)]/s_1, qubit_array[0]/s_2) and np.allclose(non_zero_dict[(0, 1)]/s_1, qubit_array[1]/s_2)