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

def compute_mtd(g: nx.Graph) -> list[list[int]]:
    """Compute the minimum trail decomposition of graph where the length of
    trails is unbounded"""


    return []
