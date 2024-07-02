import networkx as nx
from optyx.compiler.x_fusions import min_trail_decomp


# Indicates whether the given list of a trails constitute a valid trail
# decomposition for the graph.
def is_trail_decomp(g: nx.Graph, trails: list[list[int]]) -> bool:
    for trail in trails:
        for i in range(len(trail) - 1):
            print(trail[i], trail[i + 1])
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
        print("----------")
        print(g.edges())
        trails = min_trail_decomp(g.copy())
        assert is_trail_decomp(g, trails)
