import os
import pytest
import pyzx as zx

import networkx as nx

from optyx.compiler.mbqc import OpenGraph, Measurement

from optyx.compiler.semm import (
    compile_to_semm,
    compile_to_semm_with_short_lines,
    decompile_from_semm,
    decompile_from_semm_multi,
)

from optyx.compiler.mbqc import Fusion, FusionNetwork, Measurement
from optyx.compiler.semm import compile_multi_piece
from optyx.compiler.semm_decompiler import (
    decompile_to_fusion_network_multi,
)


# Generate many random graphs and confirm all of them can be compiled and
# reconstructed correctly.
# NOTE: This fails when the graph only contains one node since our graph
# structure only stores a dictionary of edges so it doesn't contain any
# information in this case. This would require refactoring the graph
# datastructure
@pytest.mark.parametrize("num_vertices", range(2, 8))
def test_fuzz_semm_compiler(num_vertices: int):
    with open(f"test/graph_data/graph{num_vertices}c.g6", "rb") as f:
        lines = f.readlines()

    graphs = nx.read_graph6(lines)
    meas = {i: Measurement(i, "XY") for i in [0]}

    # This choice of inputs and outputs is completely arbitary.
    # Should write more tests with different inputs and output combinations
    inputs = [0]
    outputs = list(range(1, num_vertices))

    # For some reason nx.read_graph6 returns a list of graphs if there are many
    # graphs, and the actual graph if there is only one graph, so we need to
    # convert it back into a list here
    graphs = graphs if type(graphs) is list else [graphs]

    for graph in graphs:
        og = OpenGraph(graph, meas, inputs, outputs)

        # If the graph does not have gflow, then we omit this case as
        # compilation will fail.
        if og.find_gflow() is None:
            continue

        ins = compile_to_semm(og)
        og_reconstructed = decompile_from_semm(ins, inputs, outputs)

        assert og == og_reconstructed


def test_decompile_multi():
    m = {i: Measurement(0.5 * i, "XY") for i in range(2)}
    fn = FusionNetwork([[0, 1], [2]], m, [Fusion(0, 2, "X")])

    # We impose any partial order on the nodes for demonstrative purposes
    def order(n: int) -> list[int]:
        return list(range(n, 3))

    ins = compile_multi_piece(fn, order)
    fn_decompiled = decompile_to_fusion_network_multi(ins)
    assert fn == fn_decompiled


# Tests that compiling from a pyzx graph to an OpenGraph returns the same
# graph. Only works with small circuits up to 4 qubits since PyZX's `tensorfy`
# function seems to consume huge amount of memory for larger qubit
def test_all_small_circuits():
    direc = "./test/circuits/"
    directory = os.fsencode(direc)

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        print(filename)
        if not filename.endswith(".qasm"):
            raise Exception(
                f"only files with extension '.qasm' allowed: {filename}"
            )

        circ = zx.Circuit.load(direc + filename)
        g = circ.to_graph()
        og = OpenGraph.from_pyzx_graph(g)

        ins = compile_to_semm_with_short_lines(og, 4)
        og_reconstructed = decompile_from_semm_multi(
            ins, og.inputs, og.outputs
        )

        assert og == og_reconstructed
