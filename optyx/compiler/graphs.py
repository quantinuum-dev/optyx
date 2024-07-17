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


def find_min_path_cover(g: nx.Graph) -> list[list[int]]:
    paths: list[list[int]] = []

    for i in range(len(g)):
        ok = find_min_path_cover_k(g, paths, i + 1)
        if ok:
            return paths

    # This part is unreachable since in the worst case, we will return the
    # path cover where each path covers only a single vertex.
    raise ValueError("unreachable!")


# Returns whether there exists a path cover of the graph with num_paths number
# of paths. If it true, then the path cover is contained in the "paths" input
# variable
def find_min_path_cover_k(
    g: nx.Graph, paths: list[list[int]], num_paths: int
) -> bool:
    for vert in g.nodes():
        # The vertex already belongs to another path
        if in_paths(paths, vert):
            continue

        paths.append([vert])
        if find_path_cover_aux(g, paths, num_paths):
            return True

        paths.pop()

    return False


# The recursive auxillary function to FindMin_pathCoverWithKPaths. Returns
# whether there exists a path covering with the given number of paths
def find_path_cover_aux(
    g: nx.Graph, paths: list[list[int]], num_paths: int
) -> bool:
    if total_length(paths) == len(g):
        return True

    # The last node in the paths so far
    current_path = paths[-1]
    tail = current_path[-1]

    for edge in g.neighbors(tail):
        if in_paths(paths, edge):
            continue

        current_path.append(edge)
        paths[-1] = current_path

        ok = find_path_cover_aux(g, paths, num_paths)
        if ok:
            return True

        current_path = current_path[:-1]
        paths[-1] = current_path

    if num_paths > 1:
        return find_min_path_cover_k(g, paths, num_paths - 1)

    return False


# Checked whether the element exists in any of the paths in the slice
def in_paths(paths: list[list[int]], v: int) -> bool:
    return any(v in p for p in paths)


def total_length(p: list[list[int]]) -> int:
    return sum(len(v) for v in p)


def find_path_cover(g: nx.Graph, search_size=5) -> list[list[int]]:
    """Finds a path cover of the graph using paths of a certain maximum length
    that seeks to minimise the implementation cost of the cover.

    The difficulty is trying to make a heuristic algorithm that is still
    performant. What I do that I start by searching if there exists a path of a
    certain length from a given vertex, if we find such a path, then we
    continue again from where it left off. This gives a polynomial algorithm
    which can still find long paths.

    :param g: the graph
    :param search_size: the size of the path we will exhaustively search
                        for at each step
    """

    # Chooses a vertex with minimal value
    def choose_random_vertex(nodes: list[int]) -> int:
        return max(nodes)

    # DFS on all paths of length "path_len" starting from "start"
    # As we add paths, we remove the nodes from the H graph
    h = g.copy()
    paths: list[list[int]] = []

    while len(h.nodes) != 0:
        start = choose_random_vertex(h)

        path = find_longest_path(h, start, search_size, [])
        total_path = path

        while len(path) == search_size:
            path = find_longest_path(h, start, search_size, total_path)
            total_path.extend(path)

        h.remove_nodes_from(total_path)
        paths.append(total_path)

    return paths


# Finds the longest path in the graph starting at "start" with length at most
# "path_len"
def find_longest_path(
    g: nx.Graph, start: int, path_len: int, exclude: list[int]
) -> list[int]:
    frontier = [[start]]

    while len(frontier) != 0:
        path = frontier.pop()

        if len(path) == path_len:
            return path

        tail = path[-1]
        for nbr in g.neighbors(tail):
            if nbr not in path and nbr not in exclude:
                new_path = path.copy()
                new_path.append(nbr)
                frontier.append(new_path)

    return path


def is_path(g: nx.Graph, path: list[int]) -> bool:
    """Indicates whether the sequence of integers constitutes a path in the
    graph"""

    # Vertices cannot be repeated
    if len(path) != len(set(path)):
        return False

    path_edges = vertices_to_edges(path)
    ordered_path_edges = order_edge_tuples(set(path_edges))
    graph_edges = order_edge_tuples(set(g.edges()))

    return ordered_path_edges.issubset(graph_edges)


def is_path_cover(g: nx.Graph, paths: list[list[int]]) -> bool:
    """Indicates whether the paths constitute a valid path cover"""
    if not all(is_path(g, path) for path in paths):
        return False

    all_verts: set[int] = set()
    for path in paths:
        # All paths must be vertex-disjoint
        if len(all_verts.intersection(path)) != 0:
            return False
        all_verts.update(set(path))

    return set(g.nodes()) == all_verts


# Converts a path [1, 4, 6, 3] to a list of the individual edges [1, 4], [4,
# 6], [6, 3]
def vertices_to_edges(path: list[int]) -> list[tuple[int, int]]:
    return [(path[i], path[i + 1]) for i in range(len(path) - 1)]


def order_edge_tuples(edges: set[tuple[int, int]]) -> set[tuple[int, int]]:
    return set((e[0], e[1]) if e[0] < e[1] else (e[1], e[0]) for e in edges)
