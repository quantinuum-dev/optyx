import networkx as nx

from optyx.compiler.xy_fusions import (
    remove_hedge_paths,
    find_trail_cover,
    is_trail_cover,
)

from optyx.compiler.x_fusions import min_number_trails


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


def test_find_trail_cover():
    g = nx.Graph([(1, 3), (2, 3), (3, 4), (4, 5), (4, 6)])
    trails = find_trail_cover(g, 10)

    print(len(trails))


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
        hedge_paths = remove_hedge_paths(g.copy())
        num_trails = min_number_trails(g.copy())

        max_trail_size = g.number_of_edges()
        trail_cover = find_trail_cover(g.copy(), max_trail_size)
        assert is_trail_cover(g.copy(), trail_cover)
        assert len(trail_cover) == num_trails - len(hedge_paths)


def test_is_trail_cover():
    # Should fail since (0, 2) is not in the trail
    g = nx.Graph([(0, 1), (1, 2)])
    trails = [[0, 1], [0, 2]]
    assert not is_trail_cover(g, trails)

    # Should fail since the trail are not edge-disjoint
    g = nx.Graph([(0, 1), (1, 2)])
    trails = [[0, 1, 2], [0, 1]]
    assert not is_trail_cover(g, trails)

    # Should fail since the trail doesn't cover vertex 2
    g = nx.Graph([(0, 1), (1, 2)])
    trails = [[0, 1]]
    assert not is_trail_cover(g, trails)
