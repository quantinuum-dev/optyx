"""Functionality for reducing graphs with X fusions
"""

import math
import networkx as nx
from dataclasses import dataclass

from optyx.compiler.mbqc import ProtoFusionNetwork
from optyx.compiler.graphs import (
    local_comp_reduction,
    Triangle,
    complement_triangles,
    vertices_to_edges,
    connected_components,
)


def generate_x_fusion_network(g: nx.Graph, max_len: int) -> ProtoFusionNetwork:
    """Returns a fusion network comprised of only X fusions that implements the
    graph with bounded linear resource states.

    :param g: input graph
    :param max_len: maximum number of edges in the linear resource state
    """

    g, lcs = local_comp_reduction(g, loss)
    g, _ = complement_triangles(g, triangle_complement_condition)
    trails = bounded_min_trail_decomp(g, max_len)

    return ProtoFusionNetwork(g, trails, lcs)


def triangle_complement_condition(g: nx.Graph, tri: Triangle) -> bool:
    """Complement the triangle only when at least two vertices are odd"""
    return len([v for v in tri if g.degree(v) % 2 == 1]) >= 2


def min_number_trails(g: nx.Graph) -> int:
    """Returns the minimum number of unbounded trails needed to decompose the
    graph"""
    num_trails = 0
    for cc in connected_components(g):
        num_odd_verts = sum(cc.degree(v) % 2 for v in cc.nodes())
        if num_odd_verts == 0:
            num_trails += 1
        else:
            num_trails += num_odd_verts // 2

    return num_trails


def loss(g: nx.Graph):
    """Returns a general loss function that guides the reduction of X
    fusions"""
    num_edges = len(g.edges())
    min_trails = min_number_trails(g)

    return num_edges + min_trails


def random_odd_vertex(g: nx.Graph) -> int:
    """Returns a random odd vertex in the graph or -1 if none exist"""
    for v in g.nodes():
        if g.degree(v) % 2 == 1:
            return v

    return -1


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

    trails = []

    while g.number_of_edges() != 0:
        for cc in connected_components(g):
            num_odd_verts = sum(cc.degree(v) % 2 == 1 for v in cc.nodes())

            if num_odd_verts > 2:
                trail = random_trail_odd_vertices(cc)
            else:
                if cc.number_of_nodes() == 1:
                    trail = [list(cc.nodes())[0]]
                else:
                    # Convert the edge list to a vertex sequence
                    edges = list(nx.eulerian_path(cc))
                    trail = [u for u, _ in edges] + [edges[-1][1]]

            for i in range(len(trail) - 1):
                g.remove_edge(trail[i], trail[i + 1])

            # Remove disconnected nodes from the graph
            for v in set(trail):
                if g.degree(v) == 0:
                    g.remove_node(v)

            trails.append(trail)

    return trails


def edges_to_vertices(edges: list[tuple[int, int]]) -> list[int]:
    if len(edges) == 1:
        return list(edges[0])

    final_edges = []
    for i in range(len(edges) - 1):
        edge1 = edges[i]
        edge2 = edges[i + 1]

        if edge1[0] not in edge2:
            final_edges.append(edge1[0])
        else:
            final_edges.append(edge1[1])

    if edges[-1][0] not in edges[-2]:
        final_edges.append(edges[-1][1])
        final_edges.append(edges[-1][0])
    else:
        final_edges.append(edges[-1][0])
        final_edges.append(edges[-1][1])

    return final_edges


def is_trail_decomp(g: nx.Graph, trails: list[list[int]]) -> bool:
    """Indicates whether the given list of a trails constitute a valid trail
    decomposition for the graph."""
    for trail in trails:
        for i in range(len(trail) - 1):
            if g.has_edge(trail[i], trail[i + 1]):
                g.remove_edge(trail[i], trail[i + 1])
            else:
                return False

    return g.number_of_edges() == 0


def rotate_closed_trail(trail: list[int], v: int) -> list[int]:
    """Rotates a closed trail to change its start and end point to be a
    different vertex in the trail

    Example
    -------
    >>> from optyx.compiler.x_fusions import rotate_closed_trail
    >>> rotate_closed_trail([0, 1, 2, 3, 0], 2)
    [2, 3, 0, 1, 2]
    >>> rotate_closed_trail([0, 1, 2, 3, 0], 0)
    [0, 1, 2, 3, 0]
    >>> rotate_closed_trail([0, 1, 0], 1)
    [1, 0, 1]
    """
    if v == trail[0]:
        # trail already starts at v
        return trail

    ind = trail.index(v)

    new_trail = trail[ind:] + trail[1 : ind + 1]
    return new_trail


def join_adjacent_trails(trails: list[list[int]]) -> list[list[int]]:
    """Joins adjacent trails in the trail decomposition"""
    changed = True
    while changed:
        changed = False

        i = 0
        while i < len(trails):
            t1 = trails[i]

            j = i + 1
            while j < len(trails):
                t2 = trails[j]
                if t1[0] == t2[0]:
                    new_trail = list(reversed(t1)) + t2[1:]
                    trails[i] = new_trail
                    t1 = trails[i]
                    trails.pop(j)
                    changed = True
                elif t1[-1] == t2[0]:
                    new_trail = t1 + t2[1:]
                    trails[i] = new_trail
                    t1 = trails[i]
                    trails.pop(j)
                    changed = True
                elif t1[0] == t2[-1]:
                    new_trail = t2 + t1[1:]
                    trails[i] = new_trail
                    t1 = trails[i]
                    trails.pop(j)
                    changed = True
                elif t1[-1] == t2[-1]:
                    new_trail = t1 + list(reversed(t2))[1:]
                    trails[i] = new_trail
                    t1 = trails[i]
                    trails.pop(j)
                    changed = True
                else:
                    j += 1
            i += 1

    return trails


def minimise_trail_decomp(trails: list[list[int]]) -> list[list[int]]:
    """Converts a trail decomposition into a minimum trail decomposition
    TODO need to handle case where after joining two trails they may become a
    cycle again
    """

    closed_trails = [trail for trail in trails if trail[0] == trail[-1]]
    open_trails = [trail for trail in trails if trail[0] != trail[-1]]

    for i in range(len(open_trails)):
        ot = open_trails[i]
        ot_vert_set = set(ot)

        j = 0
        while j < len(closed_trails):
            ct = closed_trails[j]
            ct_vert_set = set(ct)

            intersection = ot_vert_set.intersection(ct_vert_set)
            if len(intersection) == 0:
                j += 1
            else:
                v = intersection.pop()
                new_trail = rotate_closed_trail(ct, v)

                ind = ot.index(v)
                new_open_trail = ot[:ind] + new_trail + ot[ind + 1 :]
                open_trails[i] = new_open_trail
                ot = open_trails[i]
                ot_vert_set = set(ot)

                del closed_trails[j]

    trails = join_adjacent_trails(open_trails + closed_trails)

    return trails


def min_trail_decomp(g: nx.Graph) -> list[list[int]]:
    """Compute the minimum trail decomposition of graph where the length of
    trails is unbounded.
    """
    trails = random_trail_decomp(g.copy())
    trails = minimise_trail_decomp(trails)
    return trails


def segment_trail(trail: list[int], length: int) -> list[list[int]]:
    """Subdivides a trail into a list of smaller trails, each having a bounded
    number of edges"""
    trail_edges = vertices_to_edges(trail)
    subtrail_edges = [
        trail_edges[i : i + length] for i in range(0, len(trail_edges), length)
    ]
    return [edges_to_vertices(t) for t in subtrail_edges]


def bounded_min_trail_decomp(g: nx.Graph, length: int) -> list[list[int]]:
    """Compute the minimum trail decomposition of graph where the number of
    edges in each trail is at most a given length.
    """
    trails = min_trail_decomp(g)

    bounded_trails = []
    for trail in trails:
        bounded_trails.extend(segment_trail(trail, length))

    return bounded_trails

def photon_bounded_min_trail_decomp_count(g: nx.Graph, photon_length: int, r: int) -> int:
    """Compute the number of trails in a trail decomposition of graph where the number of
    photons in each trail is at most some amount
    """
    assert photon_length > 2*r

    trails = min_trail_decomp(g)

    num_trails = 0
    for i, trail in enumerate(trails):
        num_photons = compute_photons_with_x_fusions(trails, i, r)
        if num_photons <= 2*r:
            num_trails += 1
        else:
            num_trails += math.ceil(float(num_photons - 2*r)/float(photon_length - 2*r))

    return num_trails

def photon_bounded_min_trail_decomp(g: nx.Graph, photon_length: int, r: int) -> list[list[int]]:
    """Compute the number of trails in a trail decomposition of graph where the number of
    photons in each trail is at most some amount
    """
    assert photon_length > 2*r

    trails = min_trail_decomp(g)

    num_trails = 0
    new_trail_cover: list[list[int]] = []

    photon_map = compute_photons_with_x_fusions_single_node(trails, r)

    for i, trail in enumerate(trails):
        photon_count = 0
        residual_trail = []

        for node_index, node in enumerate(trail):
            photon_count += photon_map[i][node_index]

            while photon_count > photon_length - r:
                remaining_photons = sum(photon_map[i][node_index+1:])
                if photon_count + remaining_photons <= photon_length:
                    # The remaining nodes can all fit in this resource state
                    break

                # We will need to split the trail some time, now we check whether this photon is our last chance to do so
                # if photon_count + next_nodes_photons > photon_length - r:
                residual_trail.append(node)
                # We broke off "photon_length" photons, but we had to introduce 2*r as well
                photon_count -= photon_length - 2*r

                new_trail_cover.append(residual_trail)
                residual_trail = []

            residual_trail.append(node)

        new_trail_cover.append(residual_trail)

    return new_trail_cover


def reduce_x_fusions(g: nx.Graph) -> nx.Graph:
    """Returns a graph which has been reduced into an equivalent graph which
    hopefully can be implemented with an X fusion network with fewer fusions

    :param g: input graph
    """

    h, lcs = local_comp_reduction(g.copy(), loss)
    h, _ = complement_triangles(h, triangle_complement_condition)

    return h

def lc_x(g: nx.Graph) -> nx.Graph:
    h, lcs = local_comp_reduction(g.copy(), loss)
    return h

def compliment_triangles_x(g: nx.Graph) -> nx.Graph:
    h, _ = complement_triangles(g.copy(), triangle_complement_condition)
    return h

def compute_photons_with_x_fusions(trails: list[list[int]], trail_index: int, r: int) -> int:
    """Computes the number of photons in a trail.
    There is some ambiguity as to which trails the measurement photons in a
    fusion should live and where the extra fusion photons should live as
    well.

    Here we have put the measurement photons in the first trail in the
    decomposition that has the node, and spread the fusion photons evenly. So
    in an N-way fusion, the first trail has one fusion photon, the next has
    two, and the next next has two until the last which has one.
    """
    # Key is the node and value is the indices of the trails which this node
    # appears in in ascending order
    occurances: dict[int, list[int]] = {}
    for i, t in enumerate(trails):
        for node in t:
            trail_order = occurances.get(node, [])
            trail_order.append(i)
            occurances[node] = trail_order

    # Find index of trail in the decomposition

    # Start with all the measurement photons
    num_photons = 0
    for node in trails[trail_index]:
        trail_order = occurances[node]

        if trail_order[0] == trail_index:
            # Add the measurement photon
            num_photons += 1

        if len(trail_order) == 1:
            # No fusions here
            continue

        # Spread the fusion photons evenly across the trails
        if trail_order[0] == trail_index or trail_order[-1] == trail_index:
            num_photons += r
        else:
            num_photons += 2*r

    return num_photons


def compute_photons_with_x_fusions_single_node(trails: list[list[int]], r: int) -> list[list[int]]:
    """Computes the number of photons in a single node in a trail

    There is some ambiguity as to which trails the measurement photons in a
    fusion should live and where the extra fusion photons should live as
    well.

    Here we have put the measurement photons in the first trail in the
    decomposition that has the node, and spread the fusion photons evenly. So
    in an N-way fusion, the first trail has one fusion photon, the next has
    two, and the next next has two until the last which has one.
    """

    # Key is the node and value is the indices of the trails which this node
    # appears in in ascending order
    occurances: dict[int, list[int]] = {}
    for i, t in enumerate(trails):
        for node in t:
            trail_order = occurances.get(node, [])
            trail_order.append(i)
            occurances[node] = trail_order

    # The number of photons in each node
    photon_map: list[list[int]] = []

    for trail_index, trail in enumerate(trails):
        trail_photons = []

        for node in trail:
            # Start with all the measurement photons
            num_photons = 0
            trail_order = occurances[node]

            if trail_order[0] == trail_index:
                # Add the measurement photon
                num_photons += 1

            if len(trail_order) == 1:
                # No fusions here
                trail_photons.append(num_photons)
                continue

            # Spread the fusion photons evenly across the trails
            if trail_order[0] == trail_index or trail_order[-1] == trail_index:
                num_photons += r
            else:
                num_photons += 2*r

            trail_photons.append(num_photons)
        photon_map.append(trail_photons)

    return photon_map
