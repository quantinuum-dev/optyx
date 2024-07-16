import networkx as nx
from optyx.compiler.xy_fusions import remove_hedge_paths


def test_remove_hedge_paths():
    g = nx.Graph([(1, 3), (2, 3), (3, 4), (4, 5), (4, 6)])
    g_clone = g.copy()

    _ = remove_hedge_paths(g)
    g_clone.remove_edges_from([(3, 4)])
    assert nx.utils.graphs_equal(g, g_clone)

    g = nx.Graph(
        [
            (1, 3),
            (2, 3),
            (3, 6),
            (4, 6),
            (5, 6),
            (6, 7),
            (7, 8),
            (7, 9),
            (6, 10),
            (10, 11),
            (10, 12),
        ]
    )
    g_clone = g.copy()

    _ = remove_hedge_paths(g)
    g_clone.remove_edges_from([(3, 6), (6, 10), (6, 7)])

    assert nx.utils.graphs_equal(g, g_clone)
