import pytest
import math
from optyx.core.zw import *
from optyx.photonic import (
    BS as BS_,
	BBS as BBS_,
	Phase as Phase_,
	Create as Create_,
    ansatz
)
from perceval.components.unitary_components import BS as BS_p
from optyx.core.diagram import Mode, Diagram
import numpy as np
import perceval as pcvl
from perceval import catalog
from optyx.qubits import Z, Scalar, Ket
from optyx.photonic import DualRail
from optyx import Channel
import random
from perceval.backends import SLOSBackend
from perceval.simulators import Simulator
from perceval.components import Source

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

@pytest.mark.skip(reason="Helper function for testing")
def dict_allclose(d1: dict, d2: dict, *, rel_tol=1e-05, abs_tol=1e-10) -> bool:
    all_keys = set(d1.keys()) | set(d2.keys())
    for key in all_keys:
        v1 = d1.get(key, 0.0)
        v2 = d2.get(key, 0.0)
        if not math.isclose(v1, v2, rel_tol=rel_tol, abs_tol=abs_tol):
            return False
    return True

@pytest.mark.skip(reason="Helper function for testing")
def check_dict_agreement(d1, d2, rtol=1e-5, atol=1e-8):
    for key in d1.keys() - d2.keys():
        assert np.isclose(d1[key], 0, rtol=rtol, atol=atol)
    for key in d2.keys() - d1.keys():
        assert np.isclose(d2[key], 0, rtol=rtol, atol=atol)
    for key in d1.keys() & d2.keys():
        assert np.isclose(d1[key], d2[key], rtol=rtol, atol=atol)

def test_perceval_ff_teleportation():
	p = pcvl.Processor("SLOS", 6)
	p.add(0, catalog["postprocessed cnot"].build_processor())
	p.add(0, pcvl.BS.H())
	p.add(2, pcvl.Detector.pnr())
	p.add(3, pcvl.Detector.pnr())
	p.add(0, pcvl.Detector.pnr())
	p.add(1, pcvl.Detector.pnr())

	ff_X = pcvl.FFCircuitProvider(2, 0, pcvl.Circuit(2))
	ff_X.add_configuration([0, 1], pcvl.PERM([1, 0]))
	p.add(2, ff_X)
	phi = pcvl.P("phi")
	ff_Z = pcvl.FFConfigurator(2, 3, pcvl.PS(phi), {"phi": 0}).add_configuration([0, 1], {"phi": np.pi})
	p.add(0, ff_Z)

	to_transmit = (0.7071067811865476+0j)*pcvl.BasicState([1, 0]) + (-0.21850801222441052+0.6724985119639574j)*pcvl.BasicState([0, 1])
	to_transmit.normalize()

	sg = pcvl.StateGenerator(pcvl.Encoding.DUAL_RAIL)
	bell_state = sg.bell_state("phi+")

	input_state = to_transmit * bell_state
	p.min_detected_photons_filter(2)

	input_state *= pcvl.BasicState([0, 0])

	p.with_input(input_state)
	res = p.probs()

	bell_state = Z(0, 2) @ Scalar(0.5**0.5)
	transmit = Ket("+") >> Z(1, 1, 0.3)

	dist = (
		transmit @ bell_state >>
		DualRail(3) >>
		Channel.from_perceval(p)
	).eval().prob_dist()

	check_dict_agreement(
    	{tuple(k): v for k, v in dict(res["results"]).items()},
    	dist
	)


def test_timedelay():
	HOM = pcvl.Processor("SLOS", 2)

	HOM.add(0, pcvl.BS())
	HOM.add(1, pcvl.TD(1))
	HOM.add(0, pcvl.BS())

	with pytest.raises(ValueError):
		Channel.from_perceval(HOM)


def test_symbolic():
	N=5

	bs = pcvl.GenericInterferometer(N,
		lambda idx: pcvl.BS(theta=pcvl.P("theta_%d" % idx)) // (0, pcvl.PS(phi=np.pi * 2 * random.random())),
		shape=pcvl.InterferometerShape.RECTANGLE,
		depth=2 * N,
		phase_shifter_fun_gen=lambda idx: pcvl.PS(phi=np.pi*2*random.random()))

	with pytest.raises(TypeError):
		Channel.from_perceval(bs)


def test_polarisation():
	p = pcvl.Processor("SLOS", 2)

	p.add(0, pcvl.HWP(0.8))

	with pytest.raises(ValueError):
		Channel.from_perceval(p)


def test_walk():
	steps = 4
	n = 2*steps

	BS_array = [[[0]*2]*(i+1) for i in range(steps)]

	i_0 = n/2
	for s in range(steps):
		if s==0:
			BS_array[s][0] = [i_0, i_0-1]
		else:
			z = 0
			for i, j in BS_array[s-1]:
				if [i+1, i] not in BS_array[s]:
					BS_array[s][z] = [i+1, i]
					z += 1
				if [j, j-1] not in BS_array[s]:
					BS_array[s][z] = [j, j-1]
					z += 1

	circuit = pcvl.Circuit(n)
	for s in range(steps):
		for bs in BS_array[s]:
			circuit.add(int(bs[1]), BS_p())

	# define input state by inserting a photon in the first mode
	mode = 3
	in_list = [0]*n
	in_list[mode] = 1
	in_state = pcvl.BasicState(in_list)

	# select a backend and define the simulator on the circuit
	simulator = Simulator(SLOSBackend())
	simulator.set_circuit(circuit)

	#Define a source and input distribution due to source noise
	source = Source(losses=0, indistinguishability=1)
	input_distribution = source.generate_distribution(expected_input=in_state)

	prob_dist = simulator.probs_svd(input_distribution)
	p_optyx = Channel.from_perceval(circuit)

	prob_dist_optyx = (
		Create_(*in_list) >>
		p_optyx
	).eval().prob_dist()

	check_dict_agreement(
    	{tuple(k): v for k, v in dict(prob_dist["results"]).items()},
    	prob_dist_optyx
	)

	# two photons input state
	in_list = [0]*n
	in_list[3], in_list[4] = 1, 1
	in_state = pcvl.BasicState(in_list)

	# select a backend and define the simulator on the circuit
	simulator = Simulator(SLOSBackend())
	simulator.set_circuit(circuit)

	# define a source and input distribution due to source noise
	source = Source(losses=0, indistinguishability=1)
	input_distribution = source.generate_distribution(expected_input=in_state)

	prob_dist = simulator.probs_svd(input_distribution)

	optyx_res = (
		Create_(*in_list) >>
		p_optyx
	).eval().prob_dist()

	check_dict_agreement(
    	{tuple(k): v for k, v in dict(prob_dist["results"]).items()},
    	optyx_res
	)

def test_cnot():
	QPU = pcvl.Processor("SLOS", 4)
	QPU.add(0, pcvl.BS.H())
	p = pcvl.catalog['postprocessed cnot'].build_processor()
	QPU.add(0, p)

	a = pcvl.Parameter("a")
	b = pcvl.Parameter("b")

	QPU.add(0, pcvl.BS.H(theta=a))
	QPU.add(2, pcvl.BS.H(theta=b))

	a.set_value(0)
	b.set_value(0)
	QPU.min_detected_photons_filter(2)
	QPU.with_input(pcvl.BasicState([1, 0, 1, 0]))
	output_distribution=QPU.probs()["results"]
	QPU_optyx = Channel.from_perceval(QPU)
	dist_optyx = (Create_(1, 0, 1, 0) >> QPU_optyx).eval().prob_dist()

	check_dict_agreement(
    	{tuple(k): v for k, v in dict(output_distribution).items()},
    	dist_optyx
	)

def test_unitaries():
	from optyx import Diagram
	from optyx.photonic import Create

	def chip_mzi(w, l):
		ansatz_ = ansatz(w, l)
		symbs = list(ansatz_.free_symbols)
		s = [(i, np.random.uniform(0, 1)) for i in symbs]
		return ansatz_.subs(*s)

	ansatz_ = chip_mzi(6, 3)
	m = ansatz_.to_path().array
	c = pcvl.Matrix(m.T)
	c = pcvl.components.Unitary(U=c)
	state = pcvl.BasicState([1, 1, 1, 1, 1, 1])
	p = pcvl.Processor("SLOS", 6)
	p.add(0, c)
	p.with_input(state)
	p_res = p.probs()["results"]
	p_res = {tuple(key): val for key, val in p_res.items()}

	o_res = (Create(1, 1, 1, 1, 1, 1) >> Diagram.from_perceval(p)).eval().prob_dist()

	rounded_result_optyx = {idx: np.round(val, 6) for idx, val in o_res.items()}
	non_zero_dict_optyx = {idx: val for idx, val in rounded_result_optyx.items() if val != 0}

	rounded_result_perceval = {idx: np.round(val, 6) for idx, val in p_res.items()}
	non_zero_dict_perceval = {idx: val for idx, val in rounded_result_perceval.items() if val != 0}

	assert dict_allclose(non_zero_dict_optyx, non_zero_dict_perceval, rel_tol=1e-03)