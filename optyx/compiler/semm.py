"""Full stack compiler functions"""

from optyx.compiler import OpenGraph
from optyx.compiler.mbqc import PSMInstruction

from optyx.compiler.single_emitter.fusion_network import (
    compile_to_fusion_network,
    find_gflow,
    sfn_to_open_graph,
)
from optyx.compiler.single_emitter.many_measure import (
    compile_single_emitter_multi_measurement,
)

from optyx.compiler.single_emitter.simulator import (
    decompile_to_fusion_pattern,
    fusion_pattern_to_network,
)


def compile_to_semm(
    g: OpenGraph,
) -> list[PSMInstruction]:
    """Compiles a graph to a single emitter many measurement device

    Example
    -------
    >>> import networkx as nx
    >>> g = nx.Graph()
    >>> g.add_edges_from([(0, 1), (0, 2), (2, 1)])
    >>> from optyx.compiler import OpenGraph, Measurement

    >>> meas = [Measurement(i, 'XY') for i in range (3)]
    >>> inputs = {0}
    >>> outputs = {2}

    >>> og = OpenGraph(g, meas, inputs, outputs)
    >>> from optyx.compiler.semm import compile_to_semm
    >>> from optyx.compiler.single_emitter import FusionNetworkSE
    >>> from optyx.compiler.mbqc import (
    ...    FusionOp,
    ...    MeasureOp,
    ...    NextNodeOp,
    ...    PSMInstruction,
    ... )
    >>> assert compile_to_semm(og) == [
    ...     NextNodeOp(node_id=0),
    ...     FusionOp(delay=3),
    ...     MeasureOp(delay=0, measurement=meas[0]),
    ...     NextNodeOp(node_id=1),
    ...     MeasureOp(delay=0, measurement=meas[1]),
    ...     NextNodeOp(node_id=2),
    ...     FusionOp(delay=0),
    ...     MeasureOp(delay=0, measurement=meas[2])
    ... ]
    """
    sfn = compile_to_fusion_network(g)
    gflow = find_gflow(g)

    ins = compile_single_emitter_multi_measurement(sfn, gflow)
    return ins


def decompile_from_semm(
    ins: list[PSMInstruction],
    inputs: set[int],
    outputs: set[int],
) -> OpenGraph:
    """Decompiles from instructions on an SEMM device back into an open
    graph

    Example
    -------
    >>> from optyx.compiler import OpenGraph, Measurement
    >>> from optyx.compiler.semm import decompile_from_semm
    >>> from optyx.compiler.single_emitter import FusionNetworkSE
    >>> from optyx.compiler.mbqc import (
    ...    FusionOp,
    ...    MeasureOp,
    ...    NextNodeOp,
    ...    PSMInstruction,
    ... )
    >>> meas = [Measurement(0.5*i, "XY") for i in range (3)]
    >>> ins = [
    ...     NextNodeOp(node_id=0),
    ...     FusionOp(delay=3),
    ...     MeasureOp(delay=0, measurement=meas[0]),
    ...     NextNodeOp(node_id=1),
    ...     MeasureOp(delay=0, measurement=meas[1]),
    ...     NextNodeOp(node_id=2),
    ...     FusionOp(delay=0),
    ...     MeasureOp(delay=0, measurement=meas[2])
    ... ]
    >>>
    >>> import networkx as nx
    >>> g = nx.Graph()
    >>> g.add_edges_from([(0, 1), (0, 2), (2, 1)])

    >>> inputs = {0}
    >>> outputs = {2}

    >>> og = OpenGraph(g, meas, inputs, outputs)
    >>> assert decompile_from_semm(ins, inputs, outputs) == og
    """
    sfp = decompile_to_fusion_pattern(ins)
    sfn = fusion_pattern_to_network(sfp)
    g = sfn_to_open_graph(sfn, inputs, outputs)
    return g
