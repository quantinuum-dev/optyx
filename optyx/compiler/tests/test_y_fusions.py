import networkx as nx
from optyx.compiler.mbqc import OpenGraph, Measurement
from optyx.compiler.semm import compute_linear_fn
from optyx.compiler.y_fusions import is_path_cover


def test_path_cover_finder():
    g = nx.Graph([(0, 1), (1, 2), (0, 3)])
    measurements = {i: Measurement(0.5 * i, "XY") for i in range(2)}
    inputs = [0, 1]
    outputs = [2, 3]
    og = OpenGraph(g, measurements, inputs, outputs)
    gflow = og.find_gflow()

    fn = compute_linear_fn(g, gflow.layers, measurements, 3)

    assert set(g.nodes) == set(fn.nodes())

    assert sorted(fn.resources[0]) == [0, 1, 2]
    assert sorted(fn.resources[1]) == [3]


# Find the path cover of a more complex structure
def test_pc_finder_complex():
    g = nx.Graph([(0, 1), (1, 2), (0, 3), (3, 4), (2, 4)])
    measurements = {i: Measurement(0.5 * i, "XY") for i in range(2)}
    measurements[4] = Measurement(0, "YZ")

    inputs = [0, 1]
    outputs = [2, 3]
    og = OpenGraph(g, measurements, inputs, outputs)
    gflow = og.find_gflow()

    fn = compute_linear_fn(g, gflow.layers, measurements, 3)

    assert set(g.nodes) == set(fn.nodes())

    assert sorted(fn.resources[0]) == [0, 1, 2]
    assert sorted(fn.resources[1]) == [3, 4]


def test_is_path_cover():
    g = nx.Graph([(0, 1), (1, 2)])
    pc = [[0, 1], [2]]
    assert is_path_cover(g, pc)

    # Doesn't cover all vertices
    g = nx.Graph([(0, 1), (1, 2)])
    pc = [[0, 1]]
    assert not is_path_cover(g, pc)

    # Paths are not mutually vertex disjoint
    g = nx.Graph([(0, 1), (1, 2)])
    pc = [[0, 1], [1, 2]]
    assert not is_path_cover(g, pc)

    # A path traverses the same vertex twice
    g = nx.Graph([(0, 1), (1, 2), (0, 2)])
    pc = [[0, 1, 2, 0]]
    assert not is_path_cover(g, pc)

    # Path cover has a vertex that doesn't exist
    g = nx.Graph([(0, 1), (1, 2), (0, 2)])
    pc = [[0, 1, 2], [3]]
    assert not is_path_cover(g, pc)
