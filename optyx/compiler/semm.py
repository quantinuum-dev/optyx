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
import graphix

from optyx.compiler.mbqc import (
    OpenGraph,
    Measurement,
    PartialOrder,
    get_fused_neighbours,
    FusionNetwork,
    Fusion,
)

from optyx.compiler.graphs import (
    delay_based_path_cover,
)

from optyx.compiler.protocols import (
    Instruction,
    FusionOp,
    MeasureOp,
    NextNodeOp,
    NextResourceStateOp,
    UnmeasuredOp,
)

from optyx.compiler.semm_decompiler import (
    decompile_to_fusion_network_multi,
)


def num_photons(ins: list[Instruction]) -> int:
    """Returns the number of photons used"""
    return sum(isinstance(i, (MeasureOp, FusionOp, UnmeasuredOp)) for i in ins)


def num_fusions(ins: list[Instruction]) -> int:
    """Returns the number of fusions used"""
    return sum(isinstance(i, FusionOp) for i in ins) // 2


def num_resource_states(ins: list[Instruction]) -> int:
    """Returns the number of resource states used"""
    return sum(isinstance(i, NextResourceStateOp) for i in ins)


def compile_to_resource_graphs(fn: FusionNetwork):
    """Compiles a fusion network into a graphix resource graph"""
    rgs: list[graphix.extraction.ResourceGraph] = []
    for resource in fn.resources:
        rg = graphix.extraction.create_resource_graph(resource)
        rgs.append(rg)

    return rg


def compile_to_semm(g: OpenGraph, line_length: int) -> list[Instruction]:
    """Compiles a graph to instructions on single emitter many measurement
    device which creates linear resource states.

    :param g: the open graph to be compiled
    :param line_length: the maximum length any linear resource state can be

    Example
    -------
    >>> import networkx as nx
    >>> g = nx.Graph([(0, 1), (1, 2)])
    >>> from optyx.compiler.mbqc import OpenGraph, Measurement, ULFusionNetwork
    >>>
    >>> meas = {i: Measurement(i, 'XY') for i in range(2)}
    >>> inputs = [0]
    >>> outputs = [2]
    >>>
    >>> og = OpenGraph(g, meas, inputs, outputs)
    >>> from optyx.compiler.semm import compile_to_semm
    >>> from optyx.compiler.protocols import (
    ...    FusionOp,
    ...    MeasureOp,
    ...    NextNodeOp,
    ...    NextResourceStateOp,
    ...    UnmeasuredOp,
    ... )
    >>> instructions = compile_to_semm(og, 3)
    >>> assert instructions == [
    ...     NextResourceStateOp(),
    ...     NextNodeOp(node_id=0),
    ...     MeasureOp(delay=0, measurement=meas[0]),
    ...     NextNodeOp(node_id=1),
    ...     MeasureOp(delay=0, measurement=meas[1]),
    ...     NextNodeOp(node_id=2),
    ...     UnmeasuredOp(),
    ... ]
    """
    gflow = g.find_gflow()
    if gflow is None:
        raise ValueError("ahhhhhhhhhhhhhhh")

    fn = compute_linear_fn(g.inside, gflow.layers, g.measurements, line_length)

    ins = compile_linear_fn(fn, gflow.partial_order())
    return ins


def simplify_graph(g: OpenGraph):
    """Simplifies the open graph by removing redundant input and output nodes.
    These are good for computing flow. But in practice they are unnecessary
    when actually compiling the graph state into instructions.
    """

    g_nx = g.inside.copy()
    meas = deepcopy(g.measurements)

    inputs = g.inputs
    outputs = g.outputs

    changed = True
    while changed:
        changed = False
        for inp in inputs:
            nbrs = list(g_nx.neighbors(inp))
            if inp in outputs:
                continue

            if len(nbrs) == 1 and meas[inp].is_z_measurement():
                changed = True
                g_nx.remove_node(inp)
                del meas[inp]
                inputs = [i if i != inp else nbrs[0] for i in inputs]

    changed = True
    while changed:
        for out in outputs:
            changed = False
            nbrs = list(g_nx.neighbors(out))
            if len(nbrs) == 1 and (
                out not in meas or meas[out].is_z_measurement()
            ):
                changed = True
                g_nx.remove_node(out)
                if out in meas:
                    del meas[out]
                outputs = [o if o != out else nbrs[0] for o in outputs]

    return (g_nx, meas, inputs, outputs)


def compile_linear_fn(
    fn: FusionNetwork, partial_order: PartialOrder
) -> list[Instruction]:
    """Compiles the fusion network into a series of instructions
    that can be executed on a single emitter/multi measurement machine.

    Assumes any additional correction induces by the fusions have already been
    incorporated into the given partial order.
    """

    c = compute_creation_times(fn)
    m = compute_completion_times(fn, partial_order, c)

    f = _get_fusion_photons_pairs(fn.fusions, c)

    ins: list[Instruction] = []

    photon = 0
    for r in fn.resources:
        ins.append(NextResourceStateOp())
        for v in r:
            ins.append(NextNodeOp(v))

            # Number of photons in a given node
            number_fusions = len(get_fused_neighbours(fn.fusions, v))

            for _ in range(number_fusions):
                photon += 1
                pair = f[photon]

                delay = max(0, pair - photon)
                # NOTE: the X fusion is hard coded into here
                ins.append(FusionOp(delay, "X"))

            photon += 1

            # Node "v" is an output, and therefore isn't measured
            if v not in fn.measurements:
                ins.append(UnmeasuredOp())
                continue

            # Calculate measurement delay
            measurement = fn.measurements[v]
            delay = max(0, m[v] - c[v])
            ins.append(MeasureOp(delay, measurement))

    return ins


def decompile_from_semm(
    ins: list[Instruction],
    inputs: list[int],
    outputs: list[int],
) -> OpenGraph:
    """Decompiles from instructions on an SEMM device back into an open
    graph

    Example
    -------
    >>> from optyx.compiler.mbqc import OpenGraph, Measurement, ULFusionNetwork
    >>> from optyx.compiler.semm import decompile_from_semm
    >>> from optyx.compiler.protocols import (
    ...    FusionOp,
    ...    MeasureOp,
    ...    NextNodeOp,
    ...    UnmeasuredOp,
    ...    NextResourceStateOp,
    ... )
    >>> meas = {i: Measurement(0.5*i, "XY") for i in range(2)}
    >>> ins = [
    ...     NextResourceStateOp(),
    ...     NextNodeOp(node_id=0),
    ...     MeasureOp(delay=0, measurement=meas[0]),
    ...     NextNodeOp(node_id=1),
    ...     MeasureOp(delay=0, measurement=meas[1]),
    ...     NextNodeOp(node_id=2),
    ...     UnmeasuredOp(),
    ... ]
    >>>
    >>> import networkx as nx
    >>> g = nx.Graph([(0, 1), (1, 2)])
    >>> inputs = [0]
    >>> outputs = [2]
    >>> og = OpenGraph(g, meas, inputs, outputs)
    >>>
    >>> assert decompile_from_semm(ins, inputs, outputs) == og
    """
    sfn = decompile_to_fusion_network_multi(ins)
    g = fn_to_open_graph_multi(sfn, inputs, outputs)
    return g


def compute_creation_times(fn: FusionNetwork) -> dict[int, int]:
    """Returns a list containing the creation times of the measurement photon
    of every node"""

    # Returns the number of fusion edges a node has
    def compute_fusions(fusions: list[Fusion], node: int) -> int:
        return sum(fusion.contains(node) for fusion in fusions)

    acc = 0
    c = {}
    for resource in fn.resources:
        for node in resource:
            # One photon for each fusion, and one measurement photon
            acc += compute_fusions(fn.fusions, node) + 1
            c[node] = acc

    return c


# Convert the fusions between nodes into fusions beteen specific photons
# There are more optimisations I could perform here to reduce the delay, but
# for now we will choose an arbitrary order for simplicity
#
# If photon 1 fuses with photon 5, then there will be two entries in the
# returned dictionary. 1: 5, and 5: 1
def _get_fusion_photons_pairs(
    fusions: list[Fusion], c: dict[int, int]
) -> dict[int, int]:
    seen: dict[int, int] = {}
    fusion_photons: dict[int, int] = {}

    for fusion in fusions:
        photon_num1 = seen.get(fusion.node1, 0) + 1
        photon_num2 = seen.get(fusion.node2, 0) + 1

        seen[fusion.node1] = photon_num1
        seen[fusion.node2] = photon_num2

        photon_index1 = c[fusion.node1] - photon_num1
        photon_index2 = c[fusion.node2] - photon_num2

        fusion_photons[photon_index1] = photon_index2
        fusion_photons[photon_index2] = photon_index1

    return fusion_photons


def compute_completion_times(
    fn: FusionNetwork, order: PartialOrder, c: dict[int, int]
) -> dict[int, int]:
    """Returns a dictionary where the key is a node ID and the value is it's
    "completion time".

    For non-output nodes, this is the time at which it can be measured, and for
    output nodes this is the time when the node could be used in another
    calculation safely. That is, all of the nodes that have the output node in
    it's correction set have been measured, and all fusions acting on the
    output node have been performed.

    :param fn: the fusion network
    :param order: the partial order on the nodes
    :param c: a dictionary with the node IDs as keys and the creation time of
        the nodes as values
    """

    # The dictionary holding the completion times
    m: dict[int, int] = {}

    # Recursively evaluate all the measurement times
    def get_measurement(node: int) -> int:
        if node in m:
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

    for v in fn.nodes():
        get_measurement(v)

    return m


def compute_linear_fn(
    g: nx.Graph,
    order_layers: dict[int, int],
    meas: dict[int, Measurement],
    k: int,
) -> FusionNetwork:
    """Compiles an open graph into a fusion network using short lines of length
    k assuming Hadamard fusions"""
    paths = delay_based_path_cover(g, order_layers, k)
    meas = deepcopy(meas)

    # Converts a path [1, 4, 6] to a list of edges [[1, 4], [4, 6]]
    def _path_to_edges(path: list[int]) -> list[tuple[int, int]]:
        return [(path[i], path[i + 1]) for i in range(len(path) - 1)]

    # Calculates the list of H-edge fusions requires to finish constructing the
    # graph given the path cover.
    def calculate_fusions(g: nx.Graph, paths: list[list[int]]) -> list[Fusion]:
        edges = g.edges()

        path_edges: list[tuple[int, int]] = []
        for path in paths:
            path_edges.extend(_path_to_edges(path))

        edges_set = {(min(e), max(e)) for e in edges}
        r_edges_set = {(min(e), max(e)) for e in path_edges}

        fusion_edges = edges_set - r_edges_set

        fusions = [Fusion(e[0], e[1], "X") for e in fusion_edges]
        return fusions

    fusions = calculate_fusions(g, paths)

    return FusionNetwork(paths, meas, fusions)


# Converts a path [1, 4, 6, 3] to a list of the individual edges [1, 4], [4,
# 6], [6, 3]
def _path_to_edges(path: list[int]) -> list[tuple[int, int]]:
    return [(path[i], path[i + 1]) for i in range(len(path) - 1)]


def _path_to_graph(path: list[int]) -> nx.Graph:
    g = nx.Graph({})

    for i, v in enumerate(path):
        if i != 0:
            g.add_edge(path[i - 1], v)
        if i != len(path) - 1:
            g.add_edge(path[i + 1], v)

    return g


def fn_to_open_graph_multi(
    fn: FusionNetwork, inputs: list[int], outputs: list[int]
) -> OpenGraph:
    """Converts a fusion network into an open graph"""

    subgraphs = [_path_to_graph(path) for path in fn.resources]

    g = nx.Graph()
    for subgraph in subgraphs:
        g = nx.union(g, subgraph)

    for fusion in fn.fusions:
        g.add_edge(fusion.node1, fusion.node2)

    return OpenGraph(g, fn.measurements, inputs, outputs)
