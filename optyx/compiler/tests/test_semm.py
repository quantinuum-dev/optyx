import os
import pytest
import networkx as nx
import pyzx as zx

from optyx.compiler.semm import (
    compile_to_semm,
    decompile_from_semm,
    compile_linear_fn,
    num_photons,
    num_fusions,
    num_resource_states,
)

from optyx.compiler.mbqc import (
    OpenGraph,
    Measurement,
    Fusion,
    FusionNetwork,
    Measurement,
)

from optyx.compiler.semm_decompiler import (
    decompile_to_fusion_network_multi,
)


# Tests an exception is raised when the graph doesn't have gflow
def test_no_gflow():
    g = nx.Graph([(0, 1), (1, 2), (2, 0)])
    inputs = [0]
    outputs = [2]
    meas = {i: Measurement(0.5 * i, "XY") for i in range(2)}
    og = OpenGraph(g, meas, inputs, outputs)

    with pytest.raises(ValueError):
        ins = compile_to_semm(og, 4)


# Tests we can decompile and recompile a fusion network with multiple lines.
def test_decompile_multi():
    m = {i: Measurement(0.5 * i, "XY") for i in range(2)}
    fn = FusionNetwork([[0, 1], [2]], m, [Fusion(0, 2, "X")])

    # We impose any partial order on the nodes for demonstrative purposes
    def order(n: int) -> list[int]:
        return list(range(n, 3))

    ins = compile_linear_fn(fn, order)
    fn_decompiled = decompile_to_fusion_network_multi(ins)
    assert fn == fn_decompiled


# Tests compiling and decompiling an open graph to the SEMM architecture and
# back returns the same graph.
@pytest.mark.parametrize("resource_len", range(3, 8))
def test_all_small_circuits(resource_len: int):
    direc = "./test/circuits/"
    directory = os.fsencode(direc)

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if not filename.endswith(".qasm"):
            raise Exception(f"only '.qasm' files allowed, not: {filename}")

        circ = zx.Circuit.load(direc + filename)
        g = circ.to_graph()
        og = OpenGraph.from_pyzx_graph(g)

        ins = compile_to_semm(og, resource_len)
        print(
            f"len={resource_len} {filename}: photons={num_photons(ins)} fusions={num_fusions(ins)} resource_states={num_resource_states(ins)}"
        )
        og_reconstructed = decompile_from_semm(ins, og.inputs, og.outputs)

        assert og == og_reconstructed
