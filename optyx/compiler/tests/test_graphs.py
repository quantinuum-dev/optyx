import networkx as nx
from optyx.compiler.graphs import find_triangles, complement_triangle


def test_find_triangle():
    g = nx.Graph([(0, 1), (1, 2), (2, 0)])
    triangles = find_triangles(g)
    assert len(triangles) == 1

    g = nx.Graph([(0, 1), (1, 2), (2, 0), (2, 3), (0, 3)])
    triangles = find_triangles(g)
    assert len(triangles) == 2


def test_complement_triangle():
    g = nx.Graph([(0, 1), (1, 2), (2, 0), (2, 3), (0, 3)])
    complement_triangle(g, (0, 1, 2))
    assert nx.is_isomorphic(
        g, nx.Graph([(4, 1), (4, 2), (4, 0), (2, 3), (0, 3)])
    )
