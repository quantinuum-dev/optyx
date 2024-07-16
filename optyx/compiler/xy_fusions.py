"""Provides functionality for finding trail covers of graphs"""

import networkx as nx

from optyx.compiler.x_fusions import min_trail_decomp


def search_for_odd(g: nx.Graph, v: int) -> list[int]:
    """Searches for a path between v and an odd vertex in the graph where the
    path only passes through nodes whose degree is greater than 2"""
    start_path = [v]
    seen = {v}
    frontier = [start_path]

    while len(frontier) > 0:
        path = frontier.pop()

        nbrs = g.neighbors(path[-1])
        for nbr in nbrs:
            if nbr in seen:
                continue

            if g.degree(nbr) <= 2:
                continue

            path.append(nbr)
            if g.degree(nbr) % 2 == 1:
                return path

            frontier.append(path)

    return []


# TODO I think this is duplicate
def remove_path_edges_from_graph(g: nx.Graph, path: list[int]):
    for i in range(len(path) - 1):
        g.remove_edge(path[i], path[i + 1])


def remove_hedge_paths(g: nx.Graph) -> list[list[int]]:
    """Use a greedy algorithm to find H-edge paths that will reduce the total
    number of odd vertices in the graph.

    The algorithm is essentially this. Go through all the odd vertices in the
    graph. At each one search for a path to another odd vertex where each
    vertex along the way has degree > 2
    """

    paths: list[list[int]] = []
    odd_verts = [
        v for v in g.nodes() if g.degree(v) > 1 and g.degree(v) % 2 == 1
    ]

    i = 0
    while i < len(odd_verts):
        path = search_for_odd(g, odd_verts[i])
        if len(path) == 0:
            i += 1
            continue

        remove_path_edges_from_graph(g, path)
        odd_verts.remove(path[-1])
        paths.append(path)

    return paths


def find_trail_cover(g: nx.Graph) -> list[list[int]]:
    _ = remove_hedge_paths(g)
    trails = min_trail_decomp(g)
    return trails


def is_trail_decomposition(g: nx.Graph, trails: list[list[int]]) -> bool:
    trail_cover_verts = set()
    for trail in trails:
        trail_cover_verts.update(trail)

    return set(g.nodes()) == trail_cover_verts
