"""Module for reducing Y fusions in MBQC patterns"""

import math
import networkx as nx

from optyx.compiler.mbqc import ProtoFusionNetwork
from optyx.compiler.graphs import (
    local_comp_reduction,
    vertices_to_edges,
    order_edge_tuples,
    Triangle,
    complement_triangles,
)


def generate_y_fusion_network(
    g: nx.Graph, max_len: int, search_size=5
) -> ProtoFusionNetwork:
    """Returns a fusion network comprised of only X fusions that implements the
    graph with bounded linear resource states.

    :param g: input graph
    :param max_len: maximum number of edges in the linear resource state
    :param search_size: increasing makes it slower but more accurate. See
        description in `find_path_cover`
    """

    g, lcs = local_comp_reduction(g, loss)
    g, _ = complement_triangles(g, triangle_complement_condition)
    paths = find_path_cover(g, max_len, search_size)

    return ProtoFusionNetwork(g, paths, lcs)

def sort_edges(edge_list: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return [(a, b) if a < b else (b, a) for a, b in edge_list]

def compute_y_fusions(g: nx.Graph, tc: list[list[int]]) -> list[tuple[int, int]]:
    edges = list(g.edges())

    edges = sort_edges(edges)

    tc_edges: list[tuple[int, int]] = []
    for trail in tc:
        tc_edges.extend(vertices_to_edges(trail))
    tc_edges = sort_edges(tc_edges)

    edge_set = set(edges)
    tc_edge_set = set(tc_edges)

    return list(edge_set - tc_edge_set)

def count_paths_in_photon_restricted_network(g: nx.Graph, paths: list[list[int]], photon_length: int, r: int) -> int:
    assert photon_length > 2*r
    y_fusions = compute_y_fusions(g, paths)

    # Contains the number of Y fusions a node in involved in
    fusion_dict: dict[int, int] = {}
    for fusion in y_fusions:
        fusion_dict[fusion[0]] = fusion_dict.get(fusion[0], 0) + 1
        fusion_dict[fusion[1]] = fusion_dict.get(fusion[1], 0) + 1

    num_paths = 0
    for i, path in enumerate(paths):
        num_photons = len(path)

        # Add all the Y fusion photons
        for node in path:
            num_photons += r*fusion_dict.get(node, 0)

        if num_photons <= 2*r:
            num_paths += 1
        else:
            num_paths += math.ceil(float(num_photons - 2*r)/float(photon_length - 2*r))

    return num_paths


def lc_y(g: nx.Graph) -> nx.Graph:
    g, lcs = local_comp_reduction(g, loss)
    return g

def complement_triangles_y(g: nx.Graph) -> nx.Graph:
    g, _ = complement_triangles(g, triangle_complement_condition)
    return g

def triangle_complement_condition(g: nx.Graph, tri: Triangle) -> bool:
    """Only complement when all vertices has degree greater than three"""
    return all(g.degree(v) > 3 for v in tri)


def loss(g: nx.Graph):
    """Returns a general loss function that guides the reduction of Y
    fusions"""
    return len(g.edges())


def find_path_cover(
    g: nx.Graph, max_len: int, search_size=5
) -> list[list[int]]:
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

        total_path: list[int] = []

        while True:
            real_search_size = min(max_len - len(total_path), search_size)
            path = find_longest_path(h, start, real_search_size)

            # Don't remove the last element since we need to search the graph
            # starting at that point
            h.remove_nodes_from(path[:-1])
            total_path.extend(path[:-1])
            start = path[-1]

            if len(path) - 1 < search_size:
                break

        h.remove_node(path[-1])
        total_path.append(path[-1])
        paths.append(total_path)

    return paths


def find_longest_path(g: nx.Graph, start: int, max_len: int) -> list[int]:
    """Finds the longest path starting at a given vertex

    :param g: the graph
    :param start: the initial vertex to start from
    :param path_len: maximum length of the path
    """

    frontier = [[start]]

    while len(frontier) != 0:
        path = frontier.pop()

        # +1 because path length is the number of edges not nodes
        if len(path) - 1 == max_len:
            return path

        tail = path[-1]
        for nbr in g.neighbors(tail):
            if nbr not in path:
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
