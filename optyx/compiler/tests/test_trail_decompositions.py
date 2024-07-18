import pytest
import os
import pyzx as zx
import networkx as nx

from optyx.compiler import OpenGraph
from optyx.compiler.x_fusions import (
    min_trail_decomp,
    bounded_min_trail_decomp,
    min_number_trails,
    minimise_trail_decomp,
    is_trail_decomp,
)


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
        assert is_trail_decomp(g.copy(), trails)
        assert len(trails) == min_number_trails(g)


# Tests that the bounded trail decomposition function produces a valid trail
# decomposition and each trail has the correct maximum length
@pytest.mark.parametrize("trail_length", range(3, 8))
def test_bounded_trail_decomp(trail_length):
    graphs = get_test_graphs(200, 10)

    for g in graphs:
        trails = bounded_min_trail_decomp(g.copy(), trail_length)
        assert is_trail_decomp(g, trails)

        # We use trail_length+1 here because trail_length bounds the number of
        # edges, but len(t) is the number of vertices.
        assert all(len(t) <= trail_length + 1 for t in trails)


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
