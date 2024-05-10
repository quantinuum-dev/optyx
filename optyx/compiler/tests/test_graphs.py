from optyx.compiler.graphs import find_min_path_cover
import networkx as nx


def test_path_cover_star():
    # This is a star graph so it should find a path cover with 3 paths
    g = nx.Graph()
    g.add_edges_from([(0, 1), (0, 2), (0, 3), (0, 4)])

    paths = find_min_path_cover(g)
    assert len(paths) == 3


def test_complete():
    # Fully connected graph, so it only requires one path
    g = nx.Graph()
    g.add_edges_from([(i, j) for i in range(4) for j in range(i)])

    paths = find_min_path_cover(g)
    assert len(paths) == 1


def test_something_weird_looking():
    # 0         3
    #  \       /
    #   1 - - 4
    #  /       \
    # 2         5
    g = nx.Graph()
    g.add_edges_from([(0, 1), (1, 2), (1, 4), (4, 3), (4, 5)])

    paths = find_min_path_cover(g)
    assert len(paths) == 2
