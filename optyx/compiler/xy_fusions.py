"""Provides functionality for finding trail covers of graphs"""

import networkx as nx

from optyx.compiler.x_fusions import loss, min_trail_decomp
from optyx.compiler.mbqc import ProtoFusionNetwork
from optyx.compiler.graphs import (
    connected_components,
    vertices_to_edges,
    order_edge_tuples,
    local_comp_reduction,
    complement_triangles,
    Triangle,
)


def generate_xy_fusion_network(
    g: nx.Graph, max_len: int
) -> ProtoFusionNetwork:
    """Returns a fusion network comprised of X and Y fusions that implements
    the graph with bounded linear resource states.

    :param g: input graph
    :param max_len: maximum number of edges in the linear resource state
    """

    # Here we use the same reduction as in the X fusions case.
    g, lcs = local_comp_reduction(g, loss)
    g, _ = complement_triangles(g, triangle_complement_condition)
    trails = find_trail_cover(g.copy(), max_len)

    return ProtoFusionNetwork(g, trails, lcs)


def triangle_complement_condition(g: nx.Graph, tri: Triangle) -> bool:
    """Complement the triangle when at least two of the nodes are odd"""
    return len([v for v in tri if g.degree(v) % 2 == 1]) >= 2


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
    vertex along the way has degree > 2 and it doesn't disconnect the graph
    into another component with zero odd vertices
    """

    paths: list[list[int]] = []
    odd_verts = [
        v for v in g.nodes() if g.degree(v) > 1 and g.degree(v) % 2 == 1
    ]

    num_zero_cc = num_zero_components(g)
    i = 0
    while i < len(odd_verts):
        path = search_for_odd(g, odd_verts[i])
        if len(path) != 0:
            path_edges = vertices_to_edges(path)

            g.remove_edges_from(path_edges)
            new_num_zero_comp = num_zero_components(g)

            # Only accept the path if it doesn't create another zero component
            if new_num_zero_comp <= num_zero_cc:
                num_zero_cc = new_num_zero_comp
                odd_verts.remove(path[-1])
                paths.append(path)
            else:
                g.add_edges_from(path_edges)

        i += 1

    return paths


def num_zero_components(g: nx.Graph) -> int:
    cc = connected_components(g)
    return sum(num_odd_verts(c) == 0 for c in cc)


def num_odd_verts(g: nx.Graph) -> int:
    return sum(g.degree(v) % 2 for v in g.nodes())


def find_trail_cover(g: nx.Graph, max_len: int) -> list[list[int]]:
    """Returns a trail cover of the graph.

    Uses a heuristic to attempt to identify opportunities to reduce the number
    of trails"""
    _ = remove_hedge_paths(g)
    trails = min_trail_decomp(g)

    all_trails = []
    for trail in trails:
        t = segment_trail_with_space(trail, max_len)
        all_trails.extend(t)

    return all_trails


def segment_trail_with_space(trail: list[int], length: int) -> list[list[int]]:
    """Subdivides a trail into a list of smaller trails, each having a bounded
    number of edges"""
    return [
        trail[i : i + length + 1] for i in range(0, len(trail), length + 1)
    ]


def is_trail_cover(g: nx.Graph, trails: list[list[int]]) -> bool:
    """Indicates whether the trails form a trail cover of the graph."""

    # Check all vertices are covered
    trail_cover_verts = set()
    for trail in trails:
        trail_cover_verts.update(trail)

    if set(g.nodes()) != trail_cover_verts:
        return False

    graph_edges = order_edge_tuples(set(g.edges()))

    # Check all edges in the trails are distinct and exist in the graph.
    all_edges: set[tuple[int, int]] = set()
    for t in trails:
        trail_edges = order_edge_tuples(set(vertices_to_edges(t)))

        # Trail has an edge which doesn't exist in the graph
        if not trail_edges.issubset(graph_edges):
            return False

        if len(all_edges.intersection(trail_edges)) != 0:
            # Edges are not distinct
            return False

        all_edges.update(trail_edges)

    return True
