"""Provides functionality for finding trail covers of graphs"""

import networkx as nx

from optyx.compiler.x_fusions import min_trail_decomp

from optyx.compiler.semm import _path_to_edges


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

            new_path = path + [nbr]
            if g.degree(nbr) % 2 == 1:
                return new_path

            frontier.append(new_path)
            seen.add(nbr)

    return []


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
        if len(path) != 0:
            path_edges = _path_to_edges(path)
            g.remove_edges_from(path_edges)
            odd_verts.remove(path[-1])
            paths.append(path)

        i += 1

    return paths


def find_trail_cover(g: nx.Graph) -> list[list[int]]:
    """Returns a trail cover of the graph.

    Uses a heuristic to attempt to identify opportunities to reduce the number
    of trails"""
    _ = remove_hedge_paths(g)
    trails = min_trail_decomp(g)
    return trails


def order_edge_tuples(edges: set[tuple[int, int]]) -> set[tuple[int, int]]:
    return set(tuple(sorted(e)) for e in edges)


def is_trail_cover(g: nx.Graph, trails: list[list[int]]) -> bool:
    """Indicates whether the trails form a trail cover of the graph."""

    # Check all vertices are covered
    trail_cover_verts = set()
    for trail in trails:
        trail_cover_verts.update(trail)

    if set(g.nodes()) != trail_cover_verts:
        return False

    graph_edges = order_edge_tuples(set(g.edges()))

    # TODO rename _path_to_edges to vertex sequence to edges or similar
    # Check all edges in the trails are distinct and exist in the graph.
    all_edges: set[tuple[int, int]] = set()
    for t in trails:
        trail_edges = order_edge_tuples(set(_path_to_edges(t)))

        # Trail has an edge which doesn't exist in the graph
        if not trail_edges.issubset(graph_edges):
            return False

        if len(all_edges.intersection(trail_edges)) != 0:
            # Edges are not distinct
            return False

        all_edges.update(trail_edges)

    return True
