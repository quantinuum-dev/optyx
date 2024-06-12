import pyzx as zx
import numpy as np
import networkx as nx

from optyx.compiler import OpenGraph, Measurement


import os


# Converts a graph to and from an optyx graph and then checks the resulting
# pyzx graph is equal to the original.
def assert_reconstructed_pyzx_graph_equal(circ: zx.Circuit):
    g = circ.to_graph()
    zx.simplify.to_graph_like(g)
    zx.simplify.full_reduce(g)

    g_copy = circ.to_graph()
    optyx_graph = OpenGraph.from_pyzx_graph(g_copy)
    reconstructed_pyzx_graph = optyx_graph.to_pyzx_graph()

    # The "tensorfy" function break if the rows aren't set for some reason
    for v in reconstructed_pyzx_graph.vertices():
        reconstructed_pyzx_graph.set_row(v, 2)

    ten = zx.tensorfy(g).flatten()
    ten_graph = zx.tensorfy(reconstructed_pyzx_graph).flatten()

    # Here we check their tensor representations instead of composing g with
    # the adjoint of reconstructed_pyzx_graph and checking it reduces to the
    # identity since there seems to be a bug where equal graphs don't produce
    # the identity
    i = np.argmax(ten)
    assert np.allclose(ten / ten[i], ten_graph / ten_graph[i])


def test_adder():
    circ = zx.Circuit.load("./test/circuits/adder_n4_debug.qasm")
    assert_reconstructed_pyzx_graph_equal(circ)


# Tests that the output of optyx and pyzx's simulators produce the same output
# for the same graph state and that we can compare them correctly.
def test_compare_pyzx_optyx_simulator():
    circ = zx.qasm(
        """
qreg q[2];

h q[1];
cx q[0], q[1];
h q[1];
"""
    )
    g = circ.to_graph()
    zx.simplify.to_graph_like(g)
    zx.simplify.full_reduce(g)
    g.apply_state("++")
    ten = zx.tensorfy(g)
    matrix_from_pyzx = zx.tensor_to_matrix(ten, 0, 2)
    state_vector_from_pyzx = matrix_from_pyzx[:, 0]

    inside = nx.Graph([(0, 1), (0, 2), (1, 3)])
    inputs = [0, 1]
    outputs = [2, 3]
    measurements = {0: Measurement(0, "XY"), 1: Measurement(0, "XY")}

    og = OpenGraph(inside, measurements, inputs, outputs)
    sv = og.simulate()
    state_vector_from_optyx = sv.flatten()

    assert np.allclose(state_vector_from_pyzx, state_vector_from_optyx)


# Tests that compiling from a pyzx graph to an OpenGraph returns the same
# graph. Only works with small circuits up to 4 qubits since PyZX's `tensorfy`
# function seems to consume huge amount of memory for larger qubit
def test_all_small_circuits():
    direc = "./test/circuits/"
    directory = os.fsencode(direc)

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if not filename.endswith(".qasm"):
            raise Exception(f"only '.qasm' files allowed: not {filename}")

        circ = zx.Circuit.load(direc + filename)
        assert_reconstructed_pyzx_graph_equal(circ)
