"""Functions for compiling a fusion network into instructions executable on a
single emitter multiple measure (SEMM) machine

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    compile_to_semm
    decompile_from_semm

"""

from copy import deepcopy
import networkx as nx

from optyx.compiler.mbqc import (
    OpenGraph,
    PartialOrder,
    get_fused_neighbours,
    add_fusion_order,
    fn_to_open_graph,
    FusionNetwork,
    Measurement,
    Fusion,
)

from optyx.compiler.path_cover import find_min_path_cover

from optyx.compiler.patterns import (
    Instruction,
    FusionOp,
    MeasureOp,
    NextNodeOp,
)

from optyx.compiler.semm_decompiler import (
    decompile_to_fusion_network,
)


def compile_to_semm(
    g: OpenGraph,
) -> list[Instruction]:
    """Compiles a graph to a single emitter many measurement device

    Example
    -------
    >>> import networkx as nx
    >>> g = nx.Graph([(0, 1), (1, 2)])
    >>> from optyx.compiler.mbqc import OpenGraph, Measurement, FusionNetwork
    >>>
    >>> meas = {i: Measurement(i, 'XY') for i in range(3)}
    >>> inputs = {0}
    >>> outputs = {2}
    >>>
    >>> og = OpenGraph(g, meas, inputs, outputs)
    >>> from optyx.compiler.semm import compile_to_semm
    >>> from optyx.compiler.patterns import (
    ...    FusionOp,
    ...    MeasureOp,
    ...    NextNodeOp,
    ...    Instruction,
    ... )
    >>> instructions = compile_to_semm(og)
    >>> assert instructions == [
    ...     NextNodeOp(node_id=0),
    ...     MeasureOp(delay=0, measurement=meas[0]),
    ...     NextNodeOp(node_id=1),
    ...     MeasureOp(delay=0, measurement=meas[1]),
    ...     NextNodeOp(node_id=2),
    ...     MeasureOp(delay=0, measurement=meas[2]),
    ... ]
    >>> assert decompile_from_semm(instructions, inputs, outputs) == og
    """
    sfn = compile_to_fusion_network(g)
    gflow = g.find_gflow()
    if gflow is None:
        raise ValueError("Graph does not have gflow")

    # Add the fusion ordering (induced by the corrections needing to be
    # performed from the fusions) to the partial order induced by gflow
    order_with_fusions = add_fusion_order(sfn.fusions, gflow.partial_order())

    ins = fn_to_semm(sfn, order_with_fusions)
    return ins


def decompile_from_semm(
    ins: list[Instruction],
    inputs: set[int],
    outputs: set[int],
) -> OpenGraph:
    """Decompiles from instructions on an SEMM device back into an open
    graph

    Example
    -------
    >>> from optyx.compiler.mbqc import OpenGraph, Measurement, FusionNetwork
    >>> from optyx.compiler.semm import decompile_from_semm
    >>> from optyx.compiler.patterns import (
    ...    FusionOp,
    ...    MeasureOp,
    ...    NextNodeOp,
    ...    Instruction,
    ... )
    >>> meas = {i: Measurement(0.5*i, "XY") for i in range(3)}
    >>> ins = [
    ...     NextNodeOp(node_id=0),
    ...     MeasureOp(delay=0, measurement=meas[0]),
    ...     NextNodeOp(node_id=1),
    ...     MeasureOp(delay=0, measurement=meas[1]),
    ...     NextNodeOp(node_id=2),
    ...     MeasureOp(delay=0, measurement=meas[2])
    ... ]
    >>>
    >>> import networkx as nx
    >>> g = nx.Graph([(0, 1), (1, 2)])
    >>>
    >>> inputs = {0}
    >>> outputs = {2}
    >>>
    >>> og = OpenGraph(g, meas, inputs, outputs)
    >>> assert decompile_from_semm(ins, inputs, outputs) == og
    """
    sfn = decompile_to_fusion_network(ins)
    g = fn_to_open_graph(sfn, inputs, outputs)
    return g


def fn_to_semm(fn: FusionNetwork, order: PartialOrder) -> list[Instruction]:
    """Compiles the fusion network into a series of instructions that can be
    executed on a single emitter/multi measurement machine.

    Assumes any additional correction induces by the fusions have already been
    incorporated into the given partial order.
    """

    c = get_creation_times(fn)
    m = get_measurement_times(fn, order, c)
    f = _get_fusion_photons(fn.fusions, fn.path, c)

    ins: list[Instruction] = []

    photon = 0
    for v in fn.path:
        ins.append(NextNodeOp(v))

        # Number of photons in a given node
        num_fusions = len(get_fused_neighbours(fn.fusions, v))

        for _ in range(num_fusions):
            photon += 1
            pair = f[photon]

            delay = max(0, pair - photon)
            # NOTE: the X fusion is hard coded into here
            ins.append(FusionOp(delay, "X"))

        # Calculate measurement delay
        photon += 1
        measurement = fn.measurements[v]
        delay = max(0, m[v] - c[v])
        ins.append(MeasureOp(delay, measurement))

    return ins


# Convert the fusions between nodes into fusions beteen specific photons
# There are more optimisations I could perform here to reduce the delay, but
# for now we will choose an arbitrary order for simplicity
#
# If photon 1 fuses with photon 5, then there will be two entries in the
# returned dictionary. 1: 5, and 5: 1
def _get_fusion_photons(
    fusions: list[Fusion], path: list[int], c: list[int]
) -> dict[int, int]:
    seen: dict[int, int] = {}
    fusion_photons: dict[int, int] = {}

    reverse_list = {v: i for i, v in enumerate(path)}

    for fusion in fusions:
        photon_num1 = seen.get(fusion.node1, 0) + 1
        photon_num2 = seen.get(fusion.node2, 0) + 1

        seen[fusion.node1] = photon_num1
        seen[fusion.node2] = photon_num2

        photon_index1 = c[reverse_list[fusion.node1]] - photon_num1
        photon_index2 = c[reverse_list[fusion.node2]] - photon_num2

        fusion_photons[photon_index1] = photon_index2
        fusion_photons[photon_index2] = photon_index1

    return fusion_photons


# Returns the number of fusion edges a node has
def _num_fusions(fusions: list[Fusion], node: int) -> int:
    return sum(fusion.contains(node) for fusion in fusions)


def get_creation_times(fn: FusionNetwork) -> list[int]:
    """Returns a list containing the creation times of the measurement photon
    of every node"""
    acc = 0
    c = []
    for node in fn.path:
        # One photon for each fusion, and one measurement photon
        acc += _num_fusions(fn.fusions, node) + 1
        c.append(acc)

    return c


def get_measurement_times(
    fn: FusionNetwork, order: PartialOrder, c: list[int]
) -> list[int]:
    """Returns a list containing the time the measurement photon of a given
    node can be measured"""

    m = [-1] * len(fn.path)

    # Recursively evaluate all the measurement times
    def get_measurement(node: int) -> int:
        if m[node] != -1:
            return m[node]

        past = order(node)
        # Don't want to recurse forever
        past.remove(node)

        if len(past) == 0:
            m[node] = c[node]
            return m[node]

        latest_past_measurement = 0
        if len(past) != 0:
            latest_past_measurement = max(get_measurement(u) for u in past) + 1

        m[node] = max(c[node], latest_past_measurement)
        return m[node]

    for v in fn.path:
        get_measurement(v)

    return m


def compile_to_fusion_network(og: OpenGraph) -> FusionNetwork:
    """Compiles an open graph into a fusion network for FBQC with LRS"""

    pc = find_min_path_cover(og.inside)
    path, new_vertices = _join_paths(pc)

    fusions = _calculate_x_fusions(og.inside, pc)

    meas = deepcopy(og.measurements)

    # Add in all the breaks in the linear resource state
    for i in new_vertices:
        meas[i] = Measurement(0, "XY")

    return FusionNetwork(path, meas, fusions)


# Calculates the X fusions required to implement the graph given the path cover
def _calculate_x_fusions(g: nx.Graph, paths: list[list[int]]) -> list[Fusion]:
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

    remaining_edges = list(graph_edges_sorted - path_cover_edges_sorted)

    fusions = [Fusion(e[0], e[1], "X") for e in remaining_edges]
    return fusions


# Returns a tuple sorted in ascending order
def _sorted_tuple(a: int, b: int) -> tuple[int, int]:
    return (min(a, b), max(a, b))


# Converts a path [1, 4, 6, 3] to a list of the individual edges [1, 4], [4,
# 6], [6, 3]
def _path_to_edges(path: list[int]) -> list[tuple[int, int]]:
    return [(path[i], path[i + 1]) for i in range(len(path) - 1)]


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
