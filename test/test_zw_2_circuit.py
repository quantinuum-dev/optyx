import optyx.zw as zw
import optyx.circuit as qpath
import itertools
import pytest
import numpy as np

pairs = list(itertools.product(range(1, 10, 4), range(1, 10, 4)))


@pytest.mark.parametrize("photons_1, photons_2", pairs)
def test_BS(photons_1, photons_2):
    BS = qpath.BBS(0)

    diagram_qpath = qpath.Create(photons_1, photons_2) >> BS
    diagram_zw = diagram_qpath.to_zw()

    prob_zw = np.abs(diagram_zw.to_tensor().eval().array).flatten() ** 2
    prob_zw = zw.tn_output_2_perceval_output(prob_zw, diagram_zw)

    prob_perceval = diagram_qpath.to_path().prob_with_perceval().array

    assert np.allclose(prob_zw, prob_perceval)


pairs_bias = list(
    itertools.product(range(1, 10, 4), range(1, 10, 4), [0, 1, 0.56])
)


@pytest.mark.parametrize("photons_1, photons_2, bias", pairs_bias)
def test_BBS(photons_1, photons_2, bias):
    BS = qpath.BBS(bias)

    diagram_qpath = qpath.Create(photons_1, photons_2) >> BS
    diagram_zw = diagram_qpath.to_zw()

    prob_zw = np.abs(diagram_zw.to_tensor().eval().array).flatten() ** 2
    prob_zw = zw.tn_output_2_perceval_output(prob_zw, diagram_zw)

    prob_perceval = diagram_qpath.to_path().prob_with_perceval().array

    assert np.allclose(prob_zw, prob_perceval)


@pytest.mark.parametrize("photons_1, photons_2, theta", pairs_bias)
def test_TBS(photons_1, photons_2, theta):
    BS = qpath.TBS(theta)

    diagram_qpath = qpath.Create(photons_1, photons_2) >> BS
    diagram_zw = diagram_qpath.to_zw()

    prob_zw = np.abs(diagram_zw.to_tensor().eval().array).flatten() ** 2
    prob_zw = zw.tn_output_2_perceval_output(prob_zw, diagram_zw)

    prob_perceval = diagram_qpath.to_path().prob_with_perceval().array

    assert np.allclose(prob_zw, prob_perceval)


pairs_theta_phi = list(
    itertools.product(
        range(1, 10, 4), range(1, 10, 4), [0, 1, 0.56], [0, 1, 0.56]
    )
)


@pytest.mark.parametrize("photons_1, photons_2, theta, phi", pairs_theta_phi)
def test_MZI(photons_1, photons_2, theta, phi):
    BS = qpath.MZI(theta, phi)

    diagram_qpath = qpath.Create(photons_1, photons_2) >> BS
    diagram_zw = diagram_qpath.to_zw()

    prob_zw = np.abs(diagram_zw.to_tensor().eval().array).flatten() ** 2
    prob_zw = zw.tn_output_2_perceval_output(prob_zw, diagram_zw)

    prob_perceval = diagram_qpath.to_path().prob_with_perceval().array

    assert np.allclose(prob_zw, prob_perceval)
