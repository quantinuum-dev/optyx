"""Provides common graph functionality

Will most likely be replaced by networkx in future.
Wrote a custom library now for simplicity.
"""

from typing import Optional
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
    >>> inside_graph = nx.Graph([(0, 1), (1, 2), (0, 3)])
    >>> measurements = {i: Measurement(0.5 * i, "XY") for i in range(2)}
    >>> inputs = [0, 1]
    >>> outputs = [2, 3]
    >>> og = OpenGraph(inside_graph, measurements, inputs, outputs)
    >>> from optyx.compiler.semm import fn_with_short_lines
    >>> fn = fn_with_short_lines(og, 3)
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


def find_min_path_cover(g: nx.Graph) -> list[list[int]]:
    """Returns a minimum path cover of the graph.
    This uses a brute force algorithm and so it only works on small graphs with
    approximately less than 15 vertices.


    Example
    -------
    # This is a star graph so it should find a path cover with 3 paths
    >>> g = nx.Graph([(0, 1), (0, 2), (0, 3), (0, 4)])
    >>> paths = find_min_path_cover(g)
    >>> assert len(paths) == 3

    # Fully connected graph, so it only requires one path
    >>> g = nx.Graph([(i, j) for i in range(4) for j in range(i)])
    >>> paths = find_min_path_cover(g)
    >>> assert len(paths) == 1

    # Test something weird looking
    >>> g = nx.Graph([(0, 1), (1, 2), (1, 4), (4, 3), (4, 5)])
    >>> paths = find_min_path_cover(g)
    >>> assert len(paths) == 2
    """
    for i in range(1, g.number_of_nodes()):
        paths: list[list[int]] = []

        path_cover = find_path_cover(g, paths, i)
        if path_cover is not None:
            return path_cover

    # This part is unreachable since in the worst case, we will return the
    # path cover where each path covers only a single vertex.
    raise ValueError("unreachable!")


PathCover = list[list[int]]


def find_path_cover(
    g: nx.Graph, paths: list[list[int]], max_paths: int
) -> Optional[PathCover]:
    """Returns whether there exists a path cover of the graph with numPaths
    number of paths. If it True, then the path cover is contained in the
    "paths" input variable"""

    # Checks whether the element exists in any of the paths in the slice
    def _in_paths(paths: list[list[int]], v: int) -> bool:
        return any(v in p for p in paths)

    # Recursive auxillary function. Which performs a depth first search for the
    # path cover.
    def _find_path_cover_aux(
        g: nx.Graph, paths: list[list[int]], num_paths: int
    ) -> Optional[PathCover]:

        num_nodes_in_paths = sum(len(path) for path in paths)
        if num_nodes_in_paths == g.number_of_nodes():
            return paths

        # The last node in the paths so far
        current_path = paths[-1]
        tail = current_path[-1]

        for nbr in g[tail]:
            if _in_paths(paths, nbr):
                continue

            current_path.append(nbr)
            paths[-1] = current_path

            path_cover = _find_path_cover_aux(g, paths, num_paths)
            if path_cover is not None:
                return paths

            current_path.pop()

        if num_paths > 1:
            return find_path_cover(g, paths, num_paths - 1)

        return None

    for vert in g.nodes:
        # The vertex already belongs to another path
        if _in_paths(paths, vert):
            continue

        paths.append([vert])
        if _find_path_cover_aux(g, paths, max_paths):
            return paths

        paths.pop()

    return None
