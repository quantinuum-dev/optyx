"""Compiles an open graph into a fusion network

Specifically, it compiles it into a fusion network assumes access to linear
resource states"""

from copy import deepcopy

from optyx.graphs import Graph, find_min_path_cover

from optyx.compiler import (
    OpenGraph,
    PartialOrder,
    zero_measurement,
)

from optyx.compiler.single_emitter.many_measure import (
    SingleFusionNetwork,
)


def compiler_to_fusion_network(og: OpenGraph) -> SingleFusionNetwork:
    """Compiles an open graph into a fusion network for FBQC with LRS"""

    pc = find_min_path_cover(og.g)
    path, new_vertices = _join_paths(pc)

    fusions = _calculate_fusions(og.g, pc)

    meas = deepcopy(og.m)

    # Add in all the breaks in the linear resource state
    for _ in new_vertices:
        meas.append(zero_measurement())

    return SingleFusionNetwork(path, meas, fusions, og.inputs, og.outputs)


# Calculates the fusions required to implement the graph given the path cover
def _calculate_fusions(
    g: Graph, paths: list[list[int]]
) -> list[tuple[int, int]]:
    path_cover_edges: set[tuple[int, int]] = set()

    for path in paths:
        edges = _path_to_edges(path)
        path_cover_edges = path_cover_edges.union(edges)

    graph_edges = g.edges()

    # The edges are all bidirectional so lets keep them in sorted order
    # so they correct remove the edges in the line before
    graph_edges_sorted = {_sorted_tuple(a, b) for a, b in graph_edges}
    path_cover_edges_sorted = {
        _sorted_tuple(a, b) for a, b in path_cover_edges
    }

    return list(graph_edges_sorted - path_cover_edges_sorted)


# Returns a tuple sorted in ascending order
def _sorted_tuple(a: int, b: int) -> tuple[int, int]:
    return (min(a, b), max(a, b))


# Converts a path [1, 4, 6, 3] to a list of the individual edges [1, 4], [4,
# 6], [6, 3]
def _path_to_edges(path: list[int]) -> list[tuple[int, int]]:
    return [(path[i], path[i + 1]) for i in range(len(path) - 1)]


def find_gflow(g: OpenGraph) -> PartialOrder:
    """Finds gflow of the open graph"""

    def no_order(_v):
        return g.inputs

    return no_order


# Joins paths together with an additional vertex.
# Returns the fully joined path along with a list containing the IDs of the
# newly added vertices.
# For example: if we were given [[0, 1], [2, 3]], it would use a new node 4 to
# join the two paths and return [[0, 1, 4, 2, 3], [4]]
def _join_paths(paths: list[list[int]]) -> tuple[list[int], list[int]]:
    total_path = paths[0]
    next_vertex = max(max(p) for p in paths) + 1
    new_vertices = []

    for p in paths[1:]:
        total_path.append(next_vertex)
        total_path.extend(p)
        new_vertices.append(next_vertex)
        next_vertex += 1

    return (total_path, new_vertices)


def _path_to_graph(path: list[int]) -> Graph:
    g = Graph({})

    for i, v in enumerate(path):
        if i != 0:
            g.add_edge(path[i - 1], v)
        if i != len(path) - 1:
            g.add_edge(path[i + 1], v)

    return g


def sfn_to_open_graph(sfn: SingleFusionNetwork) -> OpenGraph:
    """Converts a fusion network into an open graph"""

    g = _path_to_graph(sfn.path)

    for v1, v2 in sfn.fusions:
        g.add_edge(v1, v2)

    return OpenGraph(g, sfn.measurements, sfn.inputs, sfn.outputs)
