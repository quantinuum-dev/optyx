import pytest


from optyx.compiler.single_emitter.many_measure import (
    compile_single_emitter_multi_measurement,
)

from optyx.compiler.single_emitter import FusionNetworkSE
from optyx.compiler.mbqc import (
    PartialOrder,
    Measurement,
    add_fusion_order_to_partial_order,
)

from optyx.compiler.protocols import (
    FusionOp,
    MeasureOp,
    NextNodeOp,
)

from optyx.compiler.single_emitter.simulator import (
    FusionPatternSE,
    decompile_to_fusion_pattern,
)

from optyx.compiler.single_emitter.tests.common import (
    create_unique_measurements,
    numeric_order,
)


def test_linear_graph_compilation():
    m = create_unique_measurements(3)
    fp = FusionNetworkSE([0, 1, 2], m, [])

    compile_and_verify(fp, numeric_order)


def test_triangle_compilation():
    m = create_unique_measurements(3)
    fp = FusionNetworkSE([0, 1, 2], m, [(0, 2)])

    compile_and_verify(fp, numeric_order)


def test_triangle_reverse_compilation():
    m = create_unique_measurements(3)
    fp = FusionNetworkSE([0, 1, 2], m, [(0, 2)])

    def reverse_order(n: int) -> list[int]:
        return list(range(n, 3))

    compile_and_verify(fp, reverse_order)


def compile_and_verify(fn: FusionNetworkSE, order: PartialOrder):
    ins = compile_single_emitter_multi_measurement(fn, order)

    fp = decompile_to_fusion_pattern(ins)
    _fbqc_pattern_matches_network(fn, fp)
    _pattern_satisfies_order(fp.measurements, order)


# Tests that some cases compiled without fusion ordering, will fail to satisfy
# the partial order that takes the fusion order into account
def test_triangle_reverse_compilation_fails():
    m = create_unique_measurements(4)
    fn = FusionNetworkSE([0, 1, 2, 3], m, [(0, 2)])

    ins = compile_single_emitter_multi_measurement(fn, numeric_order)

    fp = decompile_to_fusion_pattern(ins)
    _fbqc_pattern_matches_network(fn, fp)

    order_with_fusions = add_fusion_order_to_partial_order(
        fn.fusions, numeric_order
    )
    with pytest.raises(Exception):
        _pattern_satisfies_order(fp.measurements, order_with_fusions)


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
    fn = FusionNetworkSE([0, 1, 2], m, [(0, 2)])

    def order(n: int) -> list[int]:
        if n == 1:
            return [1, 0]
        return [n]

    ins = compile_single_emitter_multi_measurement(fn, order)

    assert ins == [
        NextNodeOp(0),
        FusionOp(3),
        MeasureOp(0, m[0]),
        NextNodeOp(1),
        MeasureOp(0, m[1]),  # Here node 1 can be measured immediately
        NextNodeOp(2),
        FusionOp(0),
        MeasureOp(0, m[2]),
    ]

    # Recompile but with the fusion order enhanced partial order
    order_with_fusions = add_fusion_order_to_partial_order(fn.fusions, order)
    ins = compile_single_emitter_multi_measurement(fn, order_with_fusions)

    assert ins == [
        NextNodeOp(0),
        FusionOp(3),
        MeasureOp(0, m[0]),
        NextNodeOp(1),
        MeasureOp(3, m[1]),  # With the fusion order it needs wait for node 2
        NextNodeOp(2),
        FusionOp(0),
        MeasureOp(0, m[2]),
    ]


# TODO just convert pattern to network
# Checks that the fusion pattern is actually an instance of the fusion network.
def _fbqc_pattern_matches_network(fn: FusionNetworkSE, fp: FusionPatternSE):
    assert fn.path == fp.path

    # Ensure measurements are applied to the correct nodes
    for v, measurement in fp.measurements:
        assert fn.measurements[v] == measurement

    # Check the fusions are the same
    fn.fusions = sorted(fn.fusions)
    fp.fusions = sorted(fp.fusions)

    assert len(fn.fusions) == len(fp.fusions)
    for i in range(len(fn.fusions)):
        fn_pair = fn.fusions[i]
        fp_pair = fp.fusions[i]
        assert fn_pair == fp_pair or (fn_pair == (fp_pair[1], fp_pair[0]))


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
