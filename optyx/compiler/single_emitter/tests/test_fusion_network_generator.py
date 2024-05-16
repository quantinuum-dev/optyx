import pytest

from optyx.compiler.single_emitter.fusion_network import (
    compile_to_fusion_network,
)

import networkx as nx

from optyx.compiler.mbqc import OpenGraph, Measurement

from optyx.compiler.single_emitter.fusion_network import (
    sfn_to_open_graph,
)


# Generate many random graphs and confirm all of them can be compiled and
# reconstructed correctly.
# NOTE: This fails when the graph only contains one node since our graph
# structure only stores a dictionary of edges so it doesn't contain any
# information in this case. This would require refactoring the graph
# datastructure
@pytest.mark.parametrize("num_vertices", range(2, 8))
def test_compiler_fuzz(num_vertices: int):
    with open(f"test/graph_data/graph{num_vertices}c.g6", "rb") as f:
        lines = f.readlines()

    graphs = nx.read_graph6(lines)
    meas = {i: Measurement(i, "XY") for i in range(num_vertices)}

    # This choice of inputs and outputs is completely arbitary.
    # Should write more tests with different inputs and output combinations
    inputs = {0}
    outputs = {num_vertices - 1}

    # For some reason nx.read_graph6 returns a list of graphs if there are many
    # graphs, and the actual graph if there is only one graph, so we need to
    # convert it back into a list here
    graphs = graphs if type(graphs) is list else [graphs]

    for graph in graphs:
        og = OpenGraph(graph, meas, inputs, outputs)
        assert compile_and_decompile(og, inputs, outputs)


# Compiles an open graph into a fusion network, and converts it back into an
# open graph again to verify correctness.
def compile_and_decompile(
    g: OpenGraph, inputs: set[int], outputs: set[int]
) -> bool:
    fn = compile_to_fusion_network(g)

    g_reconstructed = sfn_to_open_graph(fn, inputs, outputs)

    return g == g_reconstructed
