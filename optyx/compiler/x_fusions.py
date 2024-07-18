"""Functionality for reducing graphs with X fusions
"""

import networkx as nx

from optyx.compiler.mbqc import ProtoFusionNetwork
from optyx.compiler.graphs import local_comp_reduction


def generate_x_fusion_network(g: nx.Graph, max_len: int) -> ProtoFusionNetwork:
    """Returns a fusion network comprised of only X fusions that implements the
    graph with bounded linear resource states.

    :param g: input graph
    :param max_len: maximum number of edges in the linear resource state
    """

    g, lcs = local_comp_reduction(g, loss)
    trails = bounded_min_trail_decomp(g, max_len)

    return ProtoFusionNetwork(g, trails, lcs)


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


def connected_components(g: nx.Graph) -> list[nx.Graph]:
    """Returns a list of all the connected components"""
    components = []
    for conn in nx.connected_components(g):
        connected_component = g.subgraph(conn)
        components.append(connected_component)
    return components


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
        for cc in connected_components(g):
            num_odd_verts = sum(cc.degree(v) % 2 for v in cc.nodes())

            if num_odd_verts > 2:
                trail = random_trail_odd_vertices(cc)
            else:
                euler_trail_edges = list(nx.eulerian_path(cc))
                trail = edge_list_to_verts(euler_trail_edges)

            for i in range(len(trail) - 1):
                g.remove_edge(trail[i], trail[i + 1])

            # Remove disconnected nodes from the graph
            for v in set(trail):
                if g.degree(v) == 0:
                    g.remove_node(v)

            trails.append(trail)

    return trails


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
    return [trail[i : i + length + 1] for i in range(0, len(trail), length)]


def bounded_min_trail_decomp(g: nx.Graph, length: int) -> list[list[int]]:
    """Compute the minimum trail decomposition of graph where the number of
    edges in each trail is at most a given length.
    """
    trails = min_trail_decomp(g)

    bounded_trails = []
    for trail in trails:
        bounded_trails.extend(segment_trail(trail, length))

    return bounded_trails
