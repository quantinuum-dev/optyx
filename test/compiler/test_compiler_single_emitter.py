import pytest
import math

from optyx.compiler.mbqc import FusionNetwork, Fusion, Measurement
from optyx.compiler.semm import (
    get_measurement_times,
    get_creation_times,
    fn_to_semm,
)

from optyx.compiler.protocols import (
    MeasureOp,
    FusionOp,
    NextNodeOp,
)


def numeric_order(n: int) -> list[int]:
    return list(range(n + 1))


# Returns a list of unique measurements, all with different angles.
def create_unique_measurements(n: int) -> dict[int, Measurement]:
    small_angle = 2 * math.pi / float(n)
    return {i: Measurement(i * small_angle, "XY") for i in range(n)}


def test_linear_graph_compilation():
    m = create_unique_measurements(3)
    fn = FusionNetwork([0, 1, 2], m, [])

    ins = fn_to_semm(fn, numeric_order)
    assert ins == [
        NextNodeOp(0),
        MeasureOp(0, m[0]),
        NextNodeOp(1),
        MeasureOp(0, m[1]),
        NextNodeOp(2),
        MeasureOp(0, m[2]),
    ]


def test_triangle_compilation():
    m = create_unique_measurements(3)
    fn = FusionNetwork([0, 1, 2], m, [Fusion(0, 2, "X")])

    ins = fn_to_semm(fn, numeric_order)
    assert ins == [
        NextNodeOp(0),
        FusionOp(3, "X"),
        MeasureOp(0, m[0]),
        NextNodeOp(1),
        MeasureOp(0, m[1]),
        NextNodeOp(2),
        FusionOp(0, "X"),
        MeasureOp(0, m[2]),
    ]


@pytest.mark.parametrize("num_measurements", range(1, 3))
def test_linear_graph_measurements(num_measurements: int):
    measurements = create_unique_measurements(num_measurements)
    fn = FusionNetwork([0, 1, 2], measurements, [])

    # Should maybe use the other function
    c = [1, 2, 3]
    m = get_measurement_times(fn, numeric_order, c)
    assert m == [1, 2, 3]


def test_triangle_measurements():
    measurements = create_unique_measurements(3)

    fn = FusionNetwork([0, 1, 2], measurements, [Fusion(0, 2, "X")])

    c = [2, 3, 5]
    m = get_measurement_times(fn, numeric_order, c)

    assert m == [2, 3, 5]


def test_triangle_with_reverse_order_measurements():
    measurements = create_unique_measurements(3)

    fn = FusionNetwork([0, 1, 2], measurements, [Fusion(0, 2, "X")])

    def reverse_order(n: int) -> list[int]:
        return list(range(n, 3))

    c = [2, 3, 5]
    m = get_measurement_times(fn, reverse_order, c)

    assert m == [7, 6, 5]


def test_triangle_with_interesting_order_measurements():
    measurements = create_unique_measurements(3)

    fn = FusionNetwork([0, 1, 2], measurements, [Fusion(0, 2, "X")])

    def custom_order(n: int) -> list[int]:
        if n == 0:
            return [2, 0]
        else:
            return [n]

    c = [2, 3, 5]
    m = get_measurement_times(fn, custom_order, c)

    assert m == [6, 3, 5]


def test_creation_times():
    measurements = create_unique_measurements(3)
    fusions = [Fusion(0, 2, "X")]

    fn = FusionNetwork([0, 1, 2], measurements, fusions)

    c = get_creation_times(fn)

    assert c == [2, 3, 5]


def test_creation_times_many_fusions():
    measurements = create_unique_measurements(4)
    fusions = [Fusion(0, 2, "X"), Fusion(0, 3, "X"), Fusion(1, 3, "X")]

    fn = FusionNetwork([0, 1, 2, 3], measurements, fusions)

    c = get_creation_times(fn)

    assert c == [3, 5, 7, 10]
