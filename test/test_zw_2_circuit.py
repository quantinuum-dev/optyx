import optyx.zw as zw
import optyx.circuit as circuit
from optyx.utils import basis_vector_from_kets, occupation_numbers
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
    prob_zw = tn_output_2_amplitudes_output(prob_zw, diagram_zw)

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
    prob_zw = tn_output_2_amplitudes_output(prob_zw, diagram_zw)

    prob_perceval = diagram_qpath.to_path().prob_with_perceval().array

    assert np.allclose(prob_zw, prob_perceval)


@pytest.mark.parametrize("photons_1, photons_2, theta", pairs_bias)
def test_TBS(photons_1, photons_2, theta):
    BS = qpath.TBS(theta)

    diagram_qpath = qpath.Create(photons_1, photons_2) >> BS
    diagram_zw = diagram_qpath.to_zw()

    prob_zw = np.abs(diagram_zw.to_tensor().eval().array).flatten() ** 2
    prob_zw = tn_output_2_amplitudes_output(prob_zw, diagram_zw)

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
    prob_zw = tn_output_2_amplitudes_output(prob_zw, diagram_zw)

    prob_perceval = diagram_qpath.to_path().prob_with_perceval().array

    assert np.allclose(prob_zw, prob_perceval)


circs = [
    zw.Create(1, 3) >> circuit.BBS(0.3),
    zw.Create(2, 3) >> circuit.BBS(0.3) >> circuit.BBS(0.7),
    zw.Create(1, 1) >> circuit.TBS(0.3),
    zw.Create(1, 1) >> circuit.TBS(0.3) >> circuit.TBS(0.2),
    zw.Create(1, 1) >> circuit.MZI(0.3, 0.5) >> circuit.MZI(0.3, 0.5)
]


@pytest.mark.parametrize("circ", circs)
def test_conversion_from_amplitudes_to_tensor(circ):
    ts = circ.to_path().eval(0, as_tensor=True)
    amps = circ.to_path().eval(0)

    np.allclose(tn_output_2_amplitudes_output(tn=ts,
                                              n_extra_photons=sum(circ.to_path().selections) + sum(circ.to_path().creations)),
                                              amps.array)

circs = [
    (circuit.BBS(0.3), 2),
    (circuit.BBS(0.3) >> circuit.BBS(0.7), 3),
    (circuit.TBS(0.3), 2),
    (circuit.TBS(0.3) >> circuit.TBS(0.2), 4),
    (zw.Create(1, 1) >> circuit.MZI(0.3, 0.5) >> circuit.MZI(0.3, 0.5), 0),
    (zw.Create(1, 1) >> circuit.MZI(0.3, 0.5) >> circuit.BBS(0.5), 0)
]

@pytest.mark.parametrize("circ, n_extra_photons", circs)
def test_eval_tensor_and_perceval_tensor(circ, n_extra_photons):
    ts = circ.to_path().prob(n_extra_photons, as_tensor=True)
    amps = circ.to_path().prob_with_perceval(n_extra_photons, as_tensor=True)

    np.allclose(tn_output_2_amplitudes_output(tn=ts, n_extra_photons=n_extra_photons), amps.array)


@pytest.mark.skip(reason="Helper function")
def tn_output_2_amplitudes_output(
    tn,
    diagram=None,
    n_extra_photons=0,
) -> np.ndarray:
    """Convert the prob output of the tensor
    network to the perceval prob output"""

    if diagram is None:

        wires_out = len(tn.cod)

        cod = list(tn.cod.inside)

        idxs = list(occupation_numbers(n_extra_photons, wires_out))

        tn = np.array(tn.array).flatten()
    else:
        n_selections, n_creations = zw.calculate_num_creations_selections(diagram)

        wires_out = len(diagram.cod)

        n_photons_out = n_extra_photons - n_selections + n_creations

        cod = list(diagram.to_tensor().cod.inside)

        idxs = list(occupation_numbers(n_photons_out, wires_out))

    ix = [basis_vector_from_kets(i, cod) for i in idxs]
    res_ = []
    for i in ix:
        if i < len(tn):
            res_.append(tn[i])
        else:
            res_.append(0.0)

    return np.array(res_)