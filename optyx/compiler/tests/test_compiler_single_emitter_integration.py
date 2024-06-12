import pytest


from optyx.compiler.semm import (
    compile_linear_fn,
)

from optyx.compiler.mbqc import (
    PartialOrder,
    add_fusions_to_partial_order,
    FusionNetwork,
    Fusion,
)

from optyx.compiler.protocols import (
    FusionOp,
    MeasureOp,
    NextNodeOp,
    NextResourceStateOp,
    UnmeasuredOp,
)

from optyx.compiler.semm_decompiler import (
    decompile_to_fusion_network_multi,
)

from optyx.compiler.tests.common import (
    create_unique_measurements,
    numeric_order,
)


def test_linear_graph_compilation():
    m = create_unique_measurements(3)
    fp = FusionNetwork([[0, 1, 2]], m, [])

    compile_and_verify(fp, numeric_order)


def test_triangle_compilation():
    m = create_unique_measurements(3)
    fp = FusionNetwork([[0, 1, 2]], m, [Fusion(0, 2, "X")])

    compile_and_verify(fp, numeric_order)


def test_triangle_reverse_compilation():
    m = create_unique_measurements(3)
    fp = FusionNetwork([[0, 1, 2]], m, [Fusion(0, 2, "X")])

    def reverse_order(n: int) -> list[int]:
        return list(range(n, 3))

    compile_and_verify(fp, reverse_order)


def compile_and_verify(fn: FusionNetwork, order: PartialOrder):
    ins = compile_linear_fn(fn, order)

    fn_decompiled = decompile_to_fusion_network_multi(ins)
    assert fn == fn_decompiled


# Tests that the fusion order is being respected when we add it
#
# Here we are constructing a triangle but using the line 0 - 1 - 2 and
# fusing 0 and 2 together.
# The partial order specifies that 1 is in the future of 0, and nothing else.
#
# If we want to respect the fusion order, we would require that both 0 and 2
# are measured before 1 so that they can be first fused, and corrected on both
# sides. Therefore we must delay measuring node 1 for three time steps until we
# measure node 2.
# Whereas if we don't respect the fusion order, then we can simply measure as
# soon as they arrive.
def test_fusion_ordering():
    m = create_unique_measurements(2)
    fn = FusionNetwork([[0, 1, 2]], m, [Fusion(0, 2, "X")])

    def order(n: int) -> list[int]:
        if n == 1:
            return [1, 0]
        return [n]

    ins = compile_linear_fn(fn, order)

    assert ins == [
        NextResourceStateOp(),
        NextNodeOp(0),
        FusionOp(3, "X"),
        MeasureOp(0, m[0]),
        NextNodeOp(1),
        MeasureOp(0, m[1]),  # Here node 1 can be measured immediately
        NextNodeOp(2),
        FusionOp(0, "X"),
        UnmeasuredOp(),
    ]

    # Recompile but with the fusion order enhanced partial order
    order_with_fusions = add_fusions_to_partial_order(fn.fusions, order)
    ins = compile_linear_fn(fn, order_with_fusions)

    assert ins == [
        NextResourceStateOp(),
        NextNodeOp(0),
        FusionOp(3, "X"),
        MeasureOp(0, m[0]),
        NextNodeOp(1),
        MeasureOp(3, m[1]),  # With the fusion order it needs wait for node 2
        NextNodeOp(2),
        FusionOp(0, "X"),
        UnmeasuredOp(),
    ]
