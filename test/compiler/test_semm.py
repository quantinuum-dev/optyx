import networkx as nx
import pytest

from optyx.compiler.mbqc import Measurement, OpenGraph
from optyx.compiler.semm import compile_to_semm, decompile_from_semm


# Generate many random graphs and confirm all of them can be compiled and
# reconstructed correctly.
# NOTE: This fails when the graph only contains one node since our graph
# structure only stores a dictionary of edges so it doesn't contain any
# information in this case. This would require refactoring the graph
# datastructure
@pytest.mark.parametrize("num_vertices", range(2, 5))
def test_fuzz_semm_compiler(num_vertices: int):
    with open(f"test/graph_data/graph{num_vertices}c.g6", "rb") as f:
        lines = f.readlines()

    graphs = nx.read_graph6(lines)
    meas = {i: Measurement(i, "XY") for i in range(num_vertices - 1)}

    # This choice of inputs and outputs is completely arbitary.
    # Should write more tests with different inputs and output combinations
    inputs = {0}
    outputs = {num_vertices - 1}

    # For some reason nx.read_graph6 returns a list of graphs if there are many
    # graphs, and the actual graph if there is only one graph, so we need to
    # convert it back into a list here
    graphs = graphs if isinstance(graphs, list) else [graphs]

    for graph in graphs:
        og = OpenGraph(graph, meas, inputs, outputs)

        # If the graph does not have gflow, then we omit this case as
        # compilation will fail.
        if og.find_gflow() is None:
            continue

        ins = compile_to_semm(og)
        og_reconstructed = decompile_from_semm(ins, inputs, outputs)

        assert og == og_reconstructed
