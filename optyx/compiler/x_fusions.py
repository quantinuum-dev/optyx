"""Functionality for reducing graphs with X fusions
"""

import networkx as nx


def reduce(g: nx.Graph) -> nx.Graph:
    """Optimises a graph based on the heuristic that we want to decrease both
    the number of odd vertices and the number of edges.
    """
    vertices = list(g.nodes())
    for v in vertices:
        old_loss = loss(g)
        lc(g, v)
        new_loss = loss(g)

        if new_loss >= old_loss:
            lc(g, v)

    return g


def loss(g: nx.Graph):
    """Returns a general loss function that guides the reduction of X
    fusions"""
    num_edges = len(g.edges())
    num_odd_verts = num_odd_vertices(g)

    return num_edges + num_odd_verts


def num_odd_vertices(g: nx.Graph) -> int:
    """Returns the number of odd vertices in the graph"""
    return sum(g.degree(v) % 2 for v in g.nodes())


def toggle_edge(g: nx.Graph, v: int, u: int):
    """Toggles an edge between two nodes"""
    if g.has_edge(v, u):
        g.remove_edge(v, u)
    else:
        g.add_edge(v, u)


def lc(g: nx.Graph, v: int):
    """Locally complements a graph about the given node"""
    nbrs = list(g.neighbors(v))

    for i in range(len(nbrs)):
        for nbr in nbrs[i:]:
            toggle_edge(g, nbrs[i], nbr)


def random_odd_vertex(g: nx.Graph) -> int:
    """Returns a random odd vertex in the graph.
    Returns -1 if none exist.
    """
    for v in g.nodes():
        if g.degree(v) % 2 == 1:
            return v

    raise ValueError("no odd vertices in graph")


def random_trail_odd_vertices(g: nx.Graph) -> list[int]:
    """Returns a random path between two odd vertices.
    Simply starts at a random odd vertex and creates a trail until it reaches a
    node that it can't exist, which naturally must be odd."""

    start = random_odd_vertex(g)
    trail = [start]

    reached_end = False

    while not reached_end:
        reached_end = True

        nbrs = g.neighbors(trail[-1])
        for nbr in nbrs:
            if nbr not in trail:
                trail.append(nbr)
                reached_end = False
                break

    return trail


def random_trail_decomp(g: nx.Graph) -> list[list[int]]:
    """Returns a random trail decomposition of the graph.
    Is not guarenteed to return the minimum trail decomposition.
    """

    # Converts a list of edges which form a trail into the corresponding
    # sequence of vertices.
    def edge_list_to_verts(edges: list[tuple[int, int]]) -> list[int]:
        return [e[0] for e in edges] + [edges[-1][1]]

    trails = []

    while g.number_of_edges() != 0:
        num_odd_verts = sum(g.degree(v) % 2 for v in g.nodes())

        if num_odd_verts > 2:
            trail = random_trail_odd_vertices(g)
        if num_odd_verts <= 2:
            euler_trail_edges = list(nx.eulerian_path(g))
            trail = edge_list_to_verts(euler_trail_edges)

        for i in range(len(trail) - 1):
            g.remove_edge(trail[i], trail[i + 1])

        # Remove disconnected nodes from the graph
        for v in set(trail):
            if g.degree(v) == 0:
                g.remove_node(v)

        trails.append(trail)

    return trails


def min_trail_decomp(g: nx.Graph) -> list[list[int]]:
    """Compute the minimum trail decomposition of graph where the length of
    trails is unbounded."""
    num_odd_verts = sum(g.degree(v) % 2 for v in g.nodes())

    trails = random_trail_decomp(g.copy())

    if len(trails) == 1 and num_odd_verts == 0:
        return trails

    if len(trails) == num_odd_verts // 2:
        return trails

    # TODO need to transform the trail decomposition into a minimum trail
    # decomposition.
    raise ValueError("unimplemented")
