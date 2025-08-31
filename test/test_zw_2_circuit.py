import optyx.core.zw as zw
from optyx import photonic, classical
from optyx.utils.utils import tensor_2_amplitudes, calculate_num_creations_selections
import itertools
import pytest
import perceval as pcvl
import numpy as np

pairs = [(1, 2), (2, 1)]

@pytest.mark.parametrize("photons_1, photons_2", pairs)
def test_BS(photons_1, photons_2):
    BS = photonic.BBS(0).get_kraus()

    diagram_qpath = zw.Create(photons_1, photons_2) >> BS
    diagram_zw = diagram_qpath
    tensor = diagram_zw.to_tensor()

    m = pcvl.Matrix(diagram_qpath.to_path().array.T)
    circuit = pcvl.components.Unitary(U=m)
    backend = pcvl.BackendFactory.get_backend("SLOS")
    backend.set_circuit(circuit)
    backend.set_input_state(pcvl.BasicState([photons_1, photons_2]))

    prob_zw = tensor.eval().array
    rounded_result = np.round(prob_zw, 8)
    non_zero_dict = {idx: val for idx, val in np.ndenumerate(rounded_result) if val != 0}

    for idx, val in non_zero_dict.items():
        assert np.allclose(backend.prob_amplitude(pcvl.BasicState([idx[0], idx[1]])), val)


pairs_bias = [(1, 2, 0), (2, 1, 0), (1, 2, 0.5), (2, 1, 0.5)]


@pytest.mark.parametrize("photons_1, photons_2, bias", pairs_bias)
def test_BBS(photons_1, photons_2, bias):
    BS = photonic.BBS(bias).get_kraus()

    diagram_qpath = zw.Create(photons_1, photons_2) >> BS
    diagram_zw = diagram_qpath
    tensor = diagram_zw.to_tensor()

    n_photons_out = calculate_num_creations_selections(diagram_zw)
    n_photons_out = n_photons_out[1] - n_photons_out[0]

    prob_zw = np.abs(tensor_2_amplitudes(tensor, n_photons_out)) ** 2
    prob_perceval = diagram_qpath.to_path().prob_with_perceval().array

    assert np.allclose(prob_zw, prob_perceval)


@pytest.mark.parametrize("photons_1, photons_2, theta", pairs_bias)
def test_TBS(photons_1, photons_2, theta):
    BS = photonic.TBS(theta).get_kraus()

    diagram_qpath = zw.Create(photons_1, photons_2) >> BS
    diagram_zw = diagram_qpath
    tensor = diagram_zw.to_tensor()

    m = pcvl.Matrix(diagram_qpath.to_path().array.T)
    circuit = pcvl.components.Unitary(U=m)
    backend = pcvl.BackendFactory.get_backend("SLOS")
    backend.set_circuit(circuit)
    backend.set_input_state(pcvl.BasicState([photons_1, photons_2]))

    prob_zw = tensor.eval().array
    rounded_result = np.round(prob_zw, 8)
    non_zero_dict = {idx: val for idx, val in np.ndenumerate(rounded_result) if val != 0}

    for idx, val in non_zero_dict.items():
        assert np.allclose(backend.prob_amplitude(pcvl.BasicState([idx[0], idx[1]])), val)



pairs_theta_phi = list(
    itertools.product(
        range(1, 3), range(1, 3), [0, 1, 0.5], [0, 1, 0.5]
    )
)


@pytest.mark.parametrize("photons_1, photons_2, theta, phi", pairs_theta_phi)
def test_MZI(photons_1, photons_2, theta, phi):
    BS = photonic.MZI(theta, phi).get_kraus()

    diagram_qpath = zw.Create(photons_1, photons_2) >> BS
    diagram_zw = diagram_qpath
    tensor = diagram_zw.to_tensor()

    n_photons_out = calculate_num_creations_selections(diagram_zw)
    n_photons_out = n_photons_out[1] - n_photons_out[0]

    prob_zw = np.abs(tensor_2_amplitudes(tensor, n_photons_out)) ** 2
    prob_perceval = diagram_qpath.to_path().prob_with_perceval().array

    assert np.allclose(prob_zw, prob_perceval)


circs = [
    zw.Create(1, 1) >> photonic.BBS(0.3).get_kraus(),
    zw.Create(1, 1) >> photonic.TBS(0.3).get_kraus(),
    zw.Create(1, 1) >> photonic.MZI(0.3, 0.5).get_kraus()
]


@pytest.mark.parametrize("circ", circs)
def test_conversion_from_amplitudes_to_tensor(circ):
    ts = [i for i in circ.to_path().eval(0, as_tensor=True).array.flatten()[::-1] if i > 1e-10]
    amps = [i for i in circ.to_path().eval(0).array.flatten() if i > 1e-10]
    assert np.allclose(ts, amps)

circs = [
    (photonic.BBS(0.3).get_kraus(), 2),
    (photonic.BBS(0.3).get_kraus() >> photonic.BBS(0.7).get_kraus(), 3),
    (photonic.TBS(0.3).get_kraus(), 2),
    (photonic.TBS(0.3).get_kraus() >> photonic.TBS(0.2).get_kraus(), 4),
    (zw.Create(1, 1) >> photonic.MZI(0.3, 0.5).get_kraus() >> photonic.MZI(0.3, 0.5).get_kraus(), 0),
    (zw.Create(1, 1) >> photonic.MZI(0.3, 0.5).get_kraus() >> photonic.BBS(0.5).get_kraus(), 0)
]


@pytest.mark.parametrize("circ, n_extra_photons", circs)
def test_eval_tensor_and_perceval_tensor(circ, n_extra_photons):
    ts = [i for i in circ.to_path().eval(n_extra_photons, as_tensor=True).array.flatten()[::-1] if i > 1e-10]
    amps = [i for i in circ.to_path().eval(n_extra_photons).array.flatten() if i > 1e-10]
    assert np.allclose(ts, amps)


@pytest.mark.parametrize("photons_1, photons_2", pairs)
def test_BS_channel(photons_1, photons_2):
    BS = photonic.BBS(0)

    diagram_qpath = photonic.Create(photons_1, photons_2) >> BS
    diagram_zw = diagram_qpath >> classical.Select(photons_1, photons_2)
    tensor = diagram_zw.double().to_tensor()

    n_photons_out = calculate_num_creations_selections(diagram_zw)
    n_photons_out = n_photons_out[1] - n_photons_out[0]

    prob_zw = tensor.eval().array
    M = pcvl.Matrix(diagram_qpath.to_path().array)
    c1 = pcvl.components.Unitary(U=M)
    backend = pcvl.BackendFactory.get_backend("SLOS")
    backend.set_circuit(c1)
    backend.set_input_state(pcvl.BasicState([photons_1, photons_2]))
    prob_perceval = backend.probability(pcvl.BasicState([photons_1, photons_2]))
    assert np.allclose(prob_zw, prob_perceval)


@pytest.mark.parametrize("photons_1, photons_2, bias", pairs_bias)
def test_BBS_channel(photons_1, photons_2, bias):
    BS = photonic.BBS(bias)

    diagram_qpath = photonic.Create(photons_1, photons_2) >> BS
    diagram_zw = diagram_qpath >> classical.Select(photons_1, photons_2)
    tensor = diagram_zw.double().to_tensor()

    n_photons_out = calculate_num_creations_selections(diagram_zw)
    n_photons_out = n_photons_out[1] - n_photons_out[0]

    prob_zw = tensor.eval().array
    M = pcvl.Matrix(diagram_qpath.to_path().array)
    c1 = pcvl.components.Unitary(U=M)
    backend = pcvl.BackendFactory.get_backend("SLOS")
    backend.set_circuit(c1)
    backend.set_input_state(pcvl.BasicState([photons_1, photons_2]))
    prob_perceval = backend.probability(pcvl.BasicState([photons_1, photons_2]))
    assert np.allclose(prob_zw, prob_perceval)


@pytest.mark.parametrize("photons_1, photons_2, theta", pairs_bias)
def test_TBS_channel(photons_1, photons_2, theta):
    BS = photonic.TBS(theta)

    diagram_qpath = photonic.Create(photons_1, photons_2) >> BS
    diagram_zw = diagram_qpath >> classical.Select(photons_1, photons_2)
    tensor = diagram_zw.double().to_tensor()

    prob_zw = tensor.eval().array
    M = pcvl.Matrix(diagram_qpath.to_path().array)
    c1 = pcvl.components.Unitary(U=M)
    backend = pcvl.BackendFactory.get_backend("SLOS")
    backend.set_circuit(c1)
    backend.set_input_state(pcvl.BasicState([photons_1, photons_2]))
    prob_perceval = backend.probability(pcvl.BasicState([photons_1, photons_2]))
    assert np.allclose(prob_zw, prob_perceval)


@pytest.mark.parametrize("photons_1, photons_2, theta, phi", pairs_theta_phi)
def test_MZI_channel(photons_1, photons_2, theta, phi):
    BS = photonic.MZI(theta, phi)

    diagram_qpath = photonic.Create(photons_1, photons_2) >> BS
    diagram_zw = diagram_qpath >> classical.Select(photons_1, photons_2)
    tensor = diagram_zw.double().to_tensor()

    n_photons_out = calculate_num_creations_selections(diagram_zw)
    n_photons_out = n_photons_out[1] - n_photons_out[0]

    prob_zw = tensor.eval().array
    M = pcvl.Matrix(diagram_qpath.to_path().array)
    c1 = pcvl.components.Unitary(U=M)
    backend = pcvl.BackendFactory.get_backend("SLOS")
    backend.set_circuit(c1)
    backend.set_input_state(pcvl.BasicState([photons_1, photons_2]))
    prob_perceval = backend.probability(pcvl.BasicState([photons_1, photons_2]))
    assert np.allclose(prob_zw, prob_perceval)