import optyx.zw as zw
import optyx.lo as lo
from optyx.utils import tensor_2_amplitudes
import optyx.lo as qpath
import itertools
import pytest
import numpy as np

pairs = [(1, 2), (2, 1)]

@pytest.mark.parametrize("photons_1, photons_2", pairs)
def test_BS(photons_1, photons_2):
    BS = qpath.BBS(0)

    diagram_qpath = qpath.Create(photons_1, photons_2) >> BS
    diagram_zw = diagram_qpath.to_zw()
    tensor = diagram_zw.to_tensor()

    n_photons_out = zw.calculate_num_creations_selections(diagram_zw)
    n_photons_out = n_photons_out[1] - n_photons_out[0]

    prob_zw = np.abs(tensor_2_amplitudes(tensor, n_photons_out)) ** 2
    prob_perceval = diagram_qpath.to_path().prob_with_perceval().array

    assert np.allclose(prob_zw, prob_perceval)


pairs_bias = [(1, 2, 0), (2, 1, 0), (1, 2, 0.5), (2, 1, 0.5)]


@pytest.mark.parametrize("photons_1, photons_2, bias", pairs_bias)
def test_BBS(photons_1, photons_2, bias):
    BS = qpath.BBS(bias)

    diagram_qpath = qpath.Create(photons_1, photons_2) >> BS
    diagram_zw = diagram_qpath.to_zw()
    tensor = diagram_zw.to_tensor()

    n_photons_out = zw.calculate_num_creations_selections(diagram_zw)
    n_photons_out = n_photons_out[1] - n_photons_out[0]

    prob_zw = np.abs(tensor_2_amplitudes(tensor, n_photons_out)) ** 2
    prob_perceval = diagram_qpath.to_path().prob_with_perceval().array

    assert np.allclose(prob_zw, prob_perceval)


@pytest.mark.parametrize("photons_1, photons_2, theta", pairs_bias)
def test_TBS(photons_1, photons_2, theta):
    BS = qpath.TBS(theta)

    diagram_qpath = qpath.Create(photons_1, photons_2) >> BS
    diagram_zw = diagram_qpath.to_zw()
    tensor = diagram_zw.to_tensor()

    n_photons_out = zw.calculate_num_creations_selections(diagram_zw)
    n_photons_out = n_photons_out[1] - n_photons_out[0]

    prob_zw = np.abs(tensor_2_amplitudes(tensor, n_photons_out)) ** 2
    prob_perceval = diagram_qpath.to_path().prob_with_perceval().array

    assert np.allclose(prob_zw, prob_perceval)


pairs_theta_phi = list(
    itertools.product(
        range(1, 3), range(1, 3), [0, 1, 0.5], [0, 1, 0.5]
    )
)


@pytest.mark.parametrize("photons_1, photons_2, theta, phi", pairs_theta_phi)
def test_MZI(photons_1, photons_2, theta, phi):
    BS = qpath.MZI(theta, phi)

    diagram_qpath = qpath.Create(photons_1, photons_2) >> BS
    diagram_zw = diagram_qpath.to_zw()
    tensor = diagram_zw.to_tensor()

    n_photons_out = zw.calculate_num_creations_selections(diagram_zw)
    n_photons_out = n_photons_out[1] - n_photons_out[0]

    prob_zw = np.abs(tensor_2_amplitudes(tensor, n_photons_out)) ** 2
    prob_perceval = diagram_qpath.to_path().prob_with_perceval().array

    assert np.allclose(prob_zw, prob_perceval)


circs = [
    zw.Create(1, 1) >> lo.BBS(0.3),
    zw.Create(1, 1) >> lo.TBS(0.3),
    zw.Create(1, 1) >> lo.MZI(0.3, 0.5)
]


@pytest.mark.parametrize("circ", circs)
def test_conversion_from_amplitudes_to_tensor(circ):
    ts = [i for i in circ.to_path().eval(0, as_tensor=True).array.flatten()[::-1] if i > 1e-10]
    amps = [i for i in circ.to_path().eval(0).array.flatten() if i > 1e-10]
    assert np.allclose(ts, amps)

circs = [
    (lo.BBS(0.3), 2),
    (lo.BBS(0.3) >> lo.BBS(0.7), 3),
    (lo.TBS(0.3), 2),
    (lo.TBS(0.3) >> lo.TBS(0.2), 4),
    (zw.Create(1, 1) >> lo.MZI(0.3, 0.5) >> lo.MZI(0.3, 0.5), 0),
    (zw.Create(1, 1) >> lo.MZI(0.3, 0.5) >> lo.BBS(0.5), 0)
]

@pytest.mark.parametrize("circ, n_extra_photons", circs)
def test_eval_tensor_and_perceval_tensor(circ, n_extra_photons):
    ts = [i for i in circ.to_path().eval(n_extra_photons, as_tensor=True).array.flatten()[::-1] if i > 1e-10]
    amps = [i for i in circ.to_path().eval(n_extra_photons).array.flatten() if i > 1e-10]
    assert np.allclose(ts, amps)
