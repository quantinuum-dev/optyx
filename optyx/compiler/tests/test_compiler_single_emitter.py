import pytest

from optyx.compiler.mbqc import FusionNetwork, Fusion
from optyx.compiler.semm import (
    compute_completion_times,
    compute_creation_times,
    compile_linear_fn,
)

from optyx.compiler.protocols import (
    MeasureOp,
    FusionOp,
    NextNodeOp,
    NextResourceStateOp,
    UnmeasuredOp,
)

from optyx.compiler.tests.common import (
    numeric_order,
    create_unique_measurements,
)


def test_linear_graph_compilation():
    m = create_unique_measurements(2)
    fn = FusionNetwork([[0, 1, 2]], m, [])

    ins = compile_linear_fn(fn, numeric_order)
    assert ins == [
        NextResourceStateOp(),
        NextNodeOp(0),
        MeasureOp(0, m[0]),
        NextNodeOp(1),
        MeasureOp(0, m[1]),
        NextNodeOp(2),
        UnmeasuredOp(),
    ]


def test_triangle_compilation():
    m = create_unique_measurements(2)
    fn = FusionNetwork([[0, 1, 2]], m, [Fusion(0, 2, "X")])

    ins = compile_linear_fn(fn, numeric_order)
    assert ins == [
        NextResourceStateOp(),
        NextNodeOp(0),
        FusionOp(3, "X"),
        MeasureOp(0, m[0]),
        NextNodeOp(1),
        MeasureOp(0, m[1]),
        NextNodeOp(2),
        FusionOp(0, "X"),
        UnmeasuredOp(),
    ]


@pytest.mark.parametrize("num_nodes", range(2, 5))
def test_linear_graph_measurements(num_nodes: int):
    measurements = create_unique_measurements(num_nodes - 1)
    fn = FusionNetwork([list(range(num_nodes))], measurements, [])

    # Should maybe use the other function
    c = compute_creation_times(fn)
    m = compute_completion_times(fn, numeric_order, c)
    assert m == {i: i + 1 for i in range(num_nodes)}


def test_triangle_measurements():
    measurements = create_unique_measurements(2)

    fn = FusionNetwork([[0, 1, 2]], measurements, [Fusion(0, 2, "X")])

    c = compute_creation_times(fn)
    m = compute_completion_times(fn, numeric_order, c)

    assert m == {0: 2, 1: 3, 2: 5}


def test_triangle_with_reverse_order_measurements():
    measurements = create_unique_measurements(2)

    fn = FusionNetwork([[0, 1, 2]], measurements, [Fusion(0, 2, "X")])

    def reverse_order(n: int) -> list[int]:
        return list(range(n, 3))

    c = compute_creation_times(fn)
    m = compute_completion_times(fn, reverse_order, c)

    assert m == {0: 7, 1: 6, 2: 5}


def test_triangle_with_interesting_order_measurements():
    measurements = create_unique_measurements(2)

    fn = FusionNetwork([[0, 1, 2]], measurements, [Fusion(0, 2, "X")])

    def custom_order(n: int) -> list[int]:
        if n == 0:
            return [2, 0]
        else:
            return [n]

    c = compute_creation_times(fn)
    m = compute_completion_times(fn, custom_order, c)

    assert m == {0: 6, 1: 3, 2: 5}


def test_creation_times():
    measurements = create_unique_measurements(3)
    fusions = [Fusion(0, 2, "X")]

    fn = FusionNetwork([[0, 1, 2]], measurements, fusions)

    c = compute_creation_times(fn)

    assert c == {0: 2, 1: 3, 2: 5}


def test_creation_times_many_fusions():
    measurements = create_unique_measurements(3)
    fusions = [Fusion(0, 2, "X"), Fusion(0, 3, "X"), Fusion(1, 3, "X")]

    fn = FusionNetwork([[0, 1, 2, 3]], measurements, fusions)

    c = compute_creation_times(fn)

    assert c == {0: 3, 1: 5, 2: 7, 3: 10}
