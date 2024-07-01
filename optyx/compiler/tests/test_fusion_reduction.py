import pyzx as zx

from optyx.compiler import OpenGraph
from optyx.compiler.x_fusions import reduce, loss


def test_x_fusion_reduction():
    circ = zx.qasm("qreg q[2]; h q[1]; cx q[0], q[1]; h q[1];")
    pyzx_graph = circ.to_graph()
    og = OpenGraph.from_pyzx_graph(pyzx_graph)

    print(loss(og.inside))
    og.graph = reduce(og.inside)
    print(loss(og.inside))
