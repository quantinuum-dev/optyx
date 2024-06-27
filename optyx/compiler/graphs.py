"""Provides common graph functionality

Will most likely be replaced by networkx in future.
Wrote a custom library now for simplicity.
"""

import networkx as nx


# Chooses the path that contains the smallest value node.
# In the case of a tie, ignore the minimal nodes
# and compare again. If we exhaust all nodes, then return True.
def _is_better_path_l_infty(
    v: dict[int, int], p1: list[int], p2: list[int]
) -> bool:
    if len(p2) == 0:
        return True
    if len(p1) == 0:
        return False

    val1 = max(p1, key=lambda x: v[x])
    val2 = max(p2, key=lambda x: v[x])

    if v[val1] == v[val2]:
        new_p1 = [el for el in p1 if el != val1]
        new_p2 = [el for el in p2 if el != val2]
        return _is_better_path_l_infty(v, new_p1, new_p2)

    return v[val1] > v[val2]


def delay_based_path_cover(
    g: nx.Graph, v: dict[int, int], path_len: int, cmp=_is_better_path_l_infty
) -> list[list[int]]:
    """Finds a path cover of the graph using paths of a certain maximum length
    that seeks to minimise the implementation cost of the cover.

    :param g: the graph
    :param v: the valuation of each node
    :param path_len: maximum length of any path
    :param cmp: the function used to compare two paths. Returns true if the
        path in the first arguments is better than the second

    Example
    -------
    >>> import networkx as nx
    >>> from optyx.compiler.mbqc import OpenGraph, Measurement
    >>> g = nx.Graph([(0, 1), (1, 2), (0, 3)])
    >>> measurements = {i: Measurement(0.5 * i, "XY") for i in range(2)}
    >>> inputs = [0, 1]
    >>> outputs = [2, 3]
    >>> og = OpenGraph(g, measurements, inputs, outputs)
    >>> gflow = og.find_gflow()
    >>> from optyx.compiler.semm import compute_linear_fn
    >>> fn = compute_linear_fn(g, gflow.layers, measurements, 3)
    >>> sorted(fn.resources[0])
    [0, 1, 2]
    >>> sorted(fn.resources[1])
    [3]
    """

    # Chooses a vertex with minimal value
    def choose_maximal_vertex(nodes: list[int]) -> int:
        return max(nodes, key=lambda x: v[x])

    # DFS on all paths of length "path_len" starting from "start"
    # As we add paths, we remove the nodes from the H graph
    h = g.copy()
    paths: list[list[int]] = []

    while len(h.nodes) != 0:
        start = choose_maximal_vertex(h)
        best_path = [start]
        frontier = [[start]]
        while len(frontier) != 0:
            path = frontier.pop()

            if cmp(v, path, best_path):
                best_path = path

            if len(path) != path_len:
                tail = path[-1]
                for nbr in h.neighbors(tail):
                    if nbr in path:
                        continue

                    new_path = path.copy()
                    new_path.append(nbr)
                    frontier.append(new_path)

        h.remove_nodes_from(best_path)
        paths.append(best_path)

    return paths
