"""Provides common graph functionality

Will most likely be replaced by networkx in future.
Wrote a custom library now for simplicity.
"""

from typing import Callable
import networkx as nx


def lc(g: nx.Graph, v: int):
    """Locally complements a graph about the given node"""
    nbrs = list(g.neighbors(v))

    for i in range(len(nbrs)):
        for nbr in nbrs[i + 1 :]:
            toggle_edge(g, nbrs[i], nbr)


def toggle_edge(g: nx.Graph, v: int, u: int):
    """Toggles an edge between two nodes"""
    if g.has_edge(v, u):
        g.remove_edge(v, u)
    else:
        g.add_edge(v, u)


def local_comp_reduction(
    g: nx.Graph, loss: Callable[[nx.Graph], int]
) -> tuple[nx.Graph, list[int]]:
    """Reduces a graph by locally complemention until we reach a local minimum

    Returns the new graph and the sequence of local complementations that were
    performed to obtain it in order.

    :param g: the input graph
    :param loss: the loss function used to evaluate the graph
    """

    # Tries to locally complement each vertex in the graph once.
    # Returns a list of all vertices that where locally complemented in order.
    def reduce(g: nx.Graph) -> list[int]:
        lc_vertices = []
        vertices = list(g.nodes())

        for v in vertices:
            old_loss = loss(g)
            lc(g, v)
            new_loss = loss(g)

            if new_loss >= old_loss:
                lc(g, v)
            else:
                lc_vertices.append(v)

        return lc_vertices

    g = g.copy()

    # Keep trying to locally complement all vertices until nothing can be done.
    all_complemented_verts: list[int] = []
    while True:
        lc_vertices = reduce(g)
        if len(lc_vertices) == 0:
            return g, all_complemented_verts

        all_complemented_verts.extend(lc_vertices)


Triangle = tuple[int, int, int]


def complement_triangles(
    g: nx.Graph, should_complement: Callable[[nx.Graph, Triangle], bool]
) -> tuple[nx.Graph, list[Triangle]]:
    """Complement the triangles in the graph according to a decision
    function

    :param g: the graph
    :param should_complement: function that decides whether we should
        complement the triangle
    """
    g = g.copy()

    triangles = []
    tris = find_triangles(g)
    for tri in tris:
        # We have to add this check in because complementing a triangle
        # may remove an edge in an adjacent triangle and therefore we
        # cannot complement it.
        if not is_triangle(g, tri):
            continue

        if should_complement(g, tri):
            complement_triangle(g, tri)
            triangles.append(tri)

    return g, triangles


def is_triangle(g: nx.Graph, tri: Triangle) -> bool:
    """Indicates whether the triangle exists in the graph"""
    return (
        g.has_edge(tri[0], tri[1])
        and g.has_edge(tri[1], tri[2])
        and g.has_edge(tri[2], tri[0])
    )


def complement_triangle(g: nx.Graph, tri: Triangle):
    """Complements the triangle in the graph"""
    new_id = max(g.nodes()) + 1
    g.add_edges_from([(new_id, tri[0]), (new_id, tri[1]), (new_id, tri[2])])
    lc(g, new_id)


def find_triangles(g: nx.Graph) -> list[Triangle]:
    """Returns a list of all triangles in the graph"""
    tris = []
    nodes = list(g.nodes())

    for i in range(len(nodes)):
        for j in range(i):
            for k in range(j):
                if is_triangle(g, (nodes[i], nodes[j], nodes[k])):
                    tris.append((k, j, i))

    return tris


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


def vertices_to_edges(path: list[int]) -> list[tuple[int, int]]:
    """Converts a path [1, 4, 6, 3] to a list of the individual edges [1, 4],
    [4, 6], [6, 3]"""
    return [(path[i], path[i + 1]) for i in range(len(path) - 1)]


def order_edge_tuples(edges: set[tuple[int, int]]) -> set[tuple[int, int]]:
    """Orders the tuples in a list in numeric order"""
    return set((e[0], e[1]) if e[0] < e[1] else (e[1], e[0]) for e in edges)
