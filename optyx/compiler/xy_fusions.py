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
    g: nx.Graph, max_len: int, maximal_y_fusions=False
) -> ProtoFusionNetwork:
    """Returns a fusion network comprised of X and Y fusions that implements
    the graph with bounded linear resource states.

    :param g: input graph
    :param max_len: maximum number of edges in the linear resource state
    """

    # Here we use the same reduction as in the X fusions case.
    g, lcs = local_comp_reduction(g, loss)
    g, _ = complement_triangles(g, triangle_complement_condition)
    trails = find_trail_cover(g.copy(), max_len, maximal_y_fusions)

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


def remove_potentially_unnecessary_hedges(
    g: nx.Graph,
) -> list[tuple[int, int]]:
    """Removes edges to make Hedges that don't strictly decrease the number of
    X fusions"""
    total_edges: list[tuple[int, int]] = []
    while True:
        edges = potentially_unnecessary_hedges_one_cycle(g)

        if len(edges) == 0:
            return total_edges

        total_edges.extend(edges)
        g.remove_edges_from(edges)


def potentially_unnecessary_hedges_one_cycle(
    g: nx.Graph,
) -> list[tuple[int, int]]:
    """Removes edges to make Hedges that don't strictly decrease the number of
    X fusions"""
    odd = num_odd_verts(g)
    if odd == 0:
        e1 = zero_odd_verts_case(g)
        edges = odd_verts_case(g)
        return [e1] + edges

    return odd_verts_case(g)


def zero_odd_verts_case(g: nx.Graph) -> tuple[int, int]:
    """In this case we can take any edge, but it is better to take edges
    between two nodes whose degree > 2 so that we can apply the heuristic again
    to take more edges"""
    best = None
    for e in g.edges():
        if g.degree(e[0]) > 2 and g.degree(e[1]) > 2:
            return e

        if g.degree(e[0]) > 2 or g.degree(e[1]) > 2:
            best = e

    if best is None:
        # In this case it is a cycle, so any edge is fine
        return list(g.edges())[0]

    return best


def odd_verts_case(g: nx.Graph) -> list[tuple[int, int]]:
    """In this case we have the following rules. If both of the vertices is odd
    and doesn't produce two discconected zero odd vertex components, then take
    it.
    If only one of the vertices is odd, then it cannot produce any zero odd
    vertex disconnect components."""
    edges = []
    num_zero_cc = num_zero_components(g)

    for e in g.edges():
        if g.degree(e[0]) % 2 == 1 and g.degree(e[1]) % 2 == 1:
            g.remove_edge(e[0], e[1])
            new_num_zero_cc = num_zero_components(g)

            # Only accept if at least one of the connected components has
            # non-zero odd vertices
            if new_num_zero_cc > num_zero_cc + 1:
                g.add_edge(e[0], e[1])
            else:
                num_zero_cc = new_num_zero_cc
                edges.append(e)

        elif g.degree(e[0]) % 2 == 1 or g.degree(e[1]) % 2 == 1:
            g.remove_edge(e[0], e[1])
            new_num_zero_cc = num_zero_components(g)

            # Only accept if both connected components (if it split them) has
            # non-zero odd vertices
            if new_num_zero_cc > num_zero_cc:
                g.add_edge(e[0], e[1])
            else:
                num_zero_cc = new_num_zero_cc
                edges.append(e)

    return edges


def num_zero_components(g: nx.Graph) -> int:
    """Returns the number of connected components of the graph with zero odd
    vertices"""
    cc = connected_components(g)
    return sum(num_odd_verts(c) == 0 for c in cc)


def num_odd_verts(g: nx.Graph) -> int:
    """Returns the number of odd vertices in a graph"""
    return sum(g.degree(v) % 2 for v in g.nodes())


# TODO the example with the square with a diagonal doesn't compile to a
# reasonable trail cover unecessarily covers an example vertex

def find_trail_cover(g: nx.Graph, max_len: int, maximal_y_fusions=False) -> list[list[int]]:
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
