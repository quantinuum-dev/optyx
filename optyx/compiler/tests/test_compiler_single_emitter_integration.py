import pytest


from optyx.compiler.semm import (
    compile_single_emitter_multi_measurement,
)

from optyx.compiler.mbqc import (
    PartialOrder,
    Measurement,
    add_fusions_to_partial_order,
    ULFusionNetwork,
    Fusion,
)

from optyx.compiler.protocols import (
    FusionOp,
    MeasureOp,
    NextNodeOp,
)

from optyx.compiler.semm_decompiler import (
    decompile_to_fusion_network,
)

from optyx.compiler.tests.common import (
    create_unique_measurements,
    numeric_order,
)


def test_linear_graph_compilation():
    m = create_unique_measurements(3)
    fp = ULFusionNetwork([0, 1, 2], m, [])

    compile_and_verify(fp, numeric_order)


def test_triangle_compilation():
    m = create_unique_measurements(3)
    fp = ULFusionNetwork([0, 1, 2], m, [Fusion(0, 2, "X")])

    compile_and_verify(fp, numeric_order)


def test_triangle_reverse_compilation():
    m = create_unique_measurements(3)
    fp = ULFusionNetwork([0, 1, 2], m, [Fusion(0, 2, "X")])

    def reverse_order(n: int) -> list[int]:
        return list(range(n, 3))

    compile_and_verify(fp, reverse_order)


def compile_and_verify(fn: ULFusionNetwork, order: PartialOrder):
    ins = compile_single_emitter_multi_measurement(fn, order)

    fn_decompiled = decompile_to_fusion_network(ins)
    assert fn == fn_decompiled


# Tests that some cases compiled without fusion ordering, will fail to satisfy
# the partial order that takes the fusion order into account
def test_triangle_reverse_compilation_fails():
    m = create_unique_measurements(4)
    fn = ULFusionNetwork([0, 1, 2, 3], m, [Fusion(0, 2, "X")])

    ins = compile_single_emitter_multi_measurement(fn, numeric_order)

    fn_decompiled = decompile_to_fusion_network(ins)
    assert fn == fn_decompiled

    order_with_fusions = add_fusions_to_partial_order(
        fn.fusions, numeric_order
    )
    with pytest.raises(Exception):
        _pattern_satisfies_order(
            fn_decompiled.measurements, order_with_fusions
        )


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
    fn = ULFusionNetwork([0, 1, 2], m, [Fusion(0, 2, "X")])

    def order(n: int) -> list[int]:
        if n == 1:
            return [1, 0]
        return [n]

    ins = compile_single_emitter_multi_measurement(fn, order)

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
    order_with_fusions = add_fusions_to_partial_order(fn.fusions, order)
    ins = compile_single_emitter_multi_measurement(fn, order_with_fusions)

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


# Checks every measurement happens only after everything in its past has been
# measured.
def _pattern_satisfies_order(
    measurements: list[tuple[int, Measurement]], order: PartialOrder
):
    seen: set[int] = set()

    for v, _ in measurements:
        past = order(v)
        seen.add(v)
        assert set(past).issubset(seen)
