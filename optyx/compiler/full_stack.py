"""Full stack compiler functions"""

from optyx.compiler import OpenGraph
from optyx.compiler.single_emitter import PSMInstruction

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
    """Compiles a graph to a single emitter many measurement device"""
    sfn = compile_to_fusion_network(g)
    gflow = find_gflow(g)

    ins = compile_single_emitter_multi_measurement(sfn, gflow)
    return ins


def decompile_from_semm(
    ins: list[PSMInstruction],
    inputs: list[int],
    outputs: list[int],
) -> OpenGraph:
    """Decompiles from instructions on an SEMM device back into an open
    graph"""
    sfp = decompile_to_fusion_pattern(ins)
    sfn = fusion_pattern_to_network(sfp)
    g = sfn_to_open_graph(sfn, inputs, outputs)
    g.perform_z_deletions()
    return g
