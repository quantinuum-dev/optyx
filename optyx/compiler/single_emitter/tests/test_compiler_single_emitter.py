import pytest
from optyx.compiler.single_emitter import FusionNetworkSE
from optyx.compiler.single_emitter.many_measure import (
    get_measurement_times,
    get_creation_times,
    compile_single_emitter_multi_measurement,
)

from optyx.compiler.protocols import (
    MeasureOp,
    FusionOp,
    NextNodeOp,
)

from optyx.compiler.single_emitter.tests.common import (
    numeric_order,
    create_unique_measurements,
)


def test_linear_graph_compilation():
    m = create_unique_measurements(3)
    fp = FusionNetworkSE([0, 1, 2], m, [])

    ins = compile_single_emitter_multi_measurement(fp, numeric_order)
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
    fp = FusionNetworkSE([0, 1, 2], m, [(0, 2)])

    ins = compile_single_emitter_multi_measurement(fp, numeric_order)
    assert ins == [
        NextNodeOp(0),
        FusionOp(3),
        MeasureOp(0, m[0]),
        NextNodeOp(1),
        MeasureOp(0, m[1]),
        NextNodeOp(2),
        FusionOp(0),
        MeasureOp(0, m[2]),
    ]


@pytest.mark.parametrize("num_measurements", range(1, 3))
def test_linear_graph_measurements(num_measurements: int):
    measurements = create_unique_measurements(num_measurements)
    fp = FusionNetworkSE([0, 1, 2], measurements, [])

    # Should maybe use the other function
    c = [1, 2, 3]
    m = get_measurement_times(fp, numeric_order, c)
    assert m == [1, 2, 3]


def test_triangle_measurements():
    measurements = create_unique_measurements(3)

    fp = FusionNetworkSE([0, 1, 2], measurements, [(0, 2)])

    c = [2, 3, 5]
    m = get_measurement_times(fp, numeric_order, c)

    assert m == [2, 3, 5]


def test_triangle_with_reverse_order_measurements():
    measurements = create_unique_measurements(3)

    fp = FusionNetworkSE([0, 1, 2], measurements, [(0, 2)])

    def reverse_order(n: int) -> list[int]:
        return list(range(n, 3))

    c = [2, 3, 5]
    m = get_measurement_times(fp, reverse_order, c)

    assert m == [7, 6, 5]


def test_triangle_with_interesting_order_measurements():
    measurements = create_unique_measurements(3)

    fp = FusionNetworkSE([0, 1, 2], measurements, [(0, 2)])

    def custom_order(n: int) -> list[int]:
        if n == 0:
            return [2, 0]
        else:
            return [n]

    c = [2, 3, 5]
    m = get_measurement_times(fp, custom_order, c)

    assert m == [6, 3, 5]


def test_creation_times():
    measurements = create_unique_measurements(3)
    fusions = [(0, 2)]

    fp = FusionNetworkSE([0, 1, 2], measurements, fusions)

    c = get_creation_times(fp)

    assert c == [2, 3, 5]


def test_creation_times_many_fusions():
    measurements = create_unique_measurements(4)
    fusions = [(0, 2), (0, 3), (1, 3)]

    fp = FusionNetworkSE([0, 1, 2, 3], measurements, fusions)

    c = get_creation_times(fp)

    assert c == [3, 5, 7, 10]
