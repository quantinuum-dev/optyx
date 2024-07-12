import pytest
import math

from optyx.compiler.semm import fn_to_semm

from optyx.compiler.mbqc import (
    PartialOrder,
    Measurement,
    add_fusion_order,
    FusionNetwork,
    Fusion,
    pattern_satisfies_order,
)

from optyx.compiler.patterns import (
    FusionOp,
    MeasureOp,
    NextNodeOp,
)

from optyx.compiler.semm_decompiler import (
    decompile_to_fusion_network,
)


def numeric_order(n: int) -> list[int]:
    return list(range(n + 1))


# Returns a list of unique measurements, all with different angles.
def create_unique_measurements(n: int) -> dict[int, Measurement]:
    small_angle = 2 * math.pi / float(n)
    return {i: Measurement(i * small_angle, "XY") for i in range(n)}


# Compiles the fusion network to SEMM instructions and then decompiles the
# instructions back to a fusion network to test correctness.
def compile_and_verify(fn: FusionNetwork, order: PartialOrder):
    ins = fn_to_semm(fn, order)

    fn_decompiled = decompile_to_fusion_network(ins)
    assert fn == fn_decompiled


def test_linear_graph_compilation():
    m = create_unique_measurements(3)
    fn = FusionNetwork([0, 1, 2], m, [])

    compile_and_verify(fn, numeric_order)


def test_triangle_compilation():
    m = create_unique_measurements(3)
    fn = FusionNetwork([0, 1, 2], m, [Fusion(0, 2, "X")])

    compile_and_verify(fn, numeric_order)


def test_triangle_reverse_compilation():
    m = create_unique_measurements(3)
    fn = FusionNetwork([0, 1, 2], m, [Fusion(0, 2, "X")])

    def reverse_order(n: int) -> list[int]:
        return list(range(n, 3))

    compile_and_verify(fn, reverse_order)


# Tests that some cases compiled without fusion ordering, will fail to satisfy
# the partial order that takes the fusion order into account
def test_triangle_reverse_compilation_fails():
    m = create_unique_measurements(4)
    fn = FusionNetwork([0, 1, 2, 3], m, [Fusion(0, 2, "X")])

    ins = fn_to_semm(fn, numeric_order)

    fn_decompiled = decompile_to_fusion_network(ins)
    assert fn == fn_decompiled

    order_with_fusions = add_fusion_order(fn.fusions, numeric_order)
    with pytest.raises(Exception):
        pattern_satisfies_order(fn_decompiled.measurements, order_with_fusions)


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
    m = create_unique_measurements(3)
    fn = FusionNetwork([0, 1, 2], m, [Fusion(0, 2, "X")])

    def order(n: int) -> list[int]:
        if n == 1:
            return [1, 0]
        return [n]

    ins = fn_to_semm(fn, order)

    assert ins == [
        NextNodeOp(0),
        FusionOp(3, "X"),
        MeasureOp(0, m[0]),
        NextNodeOp(1),
        MeasureOp(0, m[1]),  # Here node 1 can be measured immediately
        NextNodeOp(2),
        FusionOp(0, "X"),
        MeasureOp(0, m[2]),
    ]

    # Recompile but with the fusion order enhanced partial order
    order_with_fusions = add_fusion_order(fn.fusions, order)
    ins = fn_to_semm(fn, order_with_fusions)

    assert ins == [
        NextNodeOp(0),
        FusionOp(3, "X"),
        MeasureOp(0, m[0]),
        NextNodeOp(1),
        MeasureOp(3, m[1]),  # With the fusion order it needs wait for node 2
        NextNodeOp(2),
        FusionOp(0, "X"),
        MeasureOp(0, m[2]),
    ]
