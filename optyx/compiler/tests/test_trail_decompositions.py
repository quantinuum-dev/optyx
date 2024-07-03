import os
import pyzx as zx
import networkx as nx

from optyx.compiler import OpenGraph
from optyx.compiler.x_fusions import (
    min_trail_decomp,
    minimise_trail_decomp,
    reduce,
)


# Indicates whether the given list of a trails constitute a valid trail
# decomposition for the graph.
def is_trail_decomp(g: nx.Graph, trails: list[list[int]]) -> bool:
    for trail in trails:
        for i in range(len(trail) - 1):
            if g.has_edge(trail[i], trail[i + 1]):
                g.remove_edge(trail[i], trail[i + 1])
            else:
                return False

    return g.number_of_edges() == 0


def test_is_trail_decomp():
    g = nx.Graph([(1, 2), (2, 3)])
    trails = [[1, 2], [2, 3]]
    assert is_trail_decomp(g.copy(), trails)

    trails = [[1, 2, 3]]
    assert is_trail_decomp(g.copy(), trails)

    trails = [[1, 2]]
    assert not is_trail_decomp(g.copy(), trails)


# Returns a list of all the different kinds of connected graphs
def get_test_graphs(num: int, nodes: int) -> list[nx.Graph]:
    graphs = []
    for i in range(num):
        g = nx.generators.erdos_renyi_graph(nodes, 0.5, seed=0)
        graphs.append(g)

    return graphs


def test_random_trail_decomp():
    graphs = get_test_graphs(200, 10)

    for g in graphs:
        trails = min_trail_decomp(g.copy())
        assert is_trail_decomp(g, trails)


def test_minimise_trail_decomp():
    trails = [[0, 1, 2, 0], [0, 3]]
    mtd = minimise_trail_decomp(trails)

    assert len(mtd) == 1

    trails = [[0, 1, 2, 0], [0, 3, 4], [5, 2, 7, 5]]
    mtd = minimise_trail_decomp(trails)

    assert len(mtd) == 1


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
        pyzx_graph = circ.to_graph()
        og = OpenGraph.from_pyzx_graph(pyzx_graph)

        trails = min_trail_decomp(og.inside.copy())
        assert is_trail_decomp(og.inside.copy(), trails)

        g = og.inside
        num_photons = (
            2 * g.number_of_edges() - len(g.nodes()) + 2 * len(trails)
        )
        print(f"{filename}: lines={len(trails)} photons={num_photons}")


# Tests that compiling from a pyzx graph to an OpenGraph returns the same
# graph. Only works with small circuits up to 4 qubits since PyZX's `tensorfy`
# function seems to consume huge amount of memory for larger qubit
def test_all_small_circuits_with_reduce():
    direc = "./test/circuits/"
    directory = os.fsencode(direc)

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if not filename.endswith(".qasm"):
            raise Exception(f"only '.qasm' files allowed: not {filename}")

        circ = zx.Circuit.load(direc + filename)
        pyzx_graph = circ.to_graph()
        og = OpenGraph.from_pyzx_graph_sneaky(pyzx_graph)

        g = reduce(og.inside)
        trails = min_trail_decomp(g.copy())
        assert is_trail_decomp(g.copy(), trails)

        # TODO also only works for connected graphs
        num_photons = (
            2 * g.number_of_edges() - len(g.nodes()) + 2 * len(trails)
        )
        print(f"{filename}: lines={len(trails)} photons={num_photons}")
