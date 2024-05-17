import pytest

from optyx.compiler.mbqc import Measurement
from optyx.compiler.semm_decompiler import (
    SingleEmitterMultiMeasure,
)
from optyx.compiler.tests.common import (
    create_unique_measurements,
)


def test_single_path():
    """Tests a one-node graph state with a single measurement is handled"""
    se = SingleEmitterMultiMeasure()

    m = Measurement(1, "XY")
    se.next_node(0)
    se.measure(m)

    fp = se.fusion_pattern()
    assert fp.path == [0]
    assert fp.measurements == [(0, m)]


def test_triangle():
    """Constructs a triangle via a fusion"""
    se = SingleEmitterMultiMeasure()

    m = create_unique_measurements(3)

    se.next_node(0)
    se.measure(m[0])
    se.delay_then_fuse(2)
    se.next_node(1)
    se.measure(m[1])
    se.next_node(2)
    se.fuse()
    se.measure(m[2])

    fp = se.fusion_pattern()
    assert fp.path == [0, 1, 2]
    assert fp.measurements == [(0, m[0]), (1, m[1]), (2, m[2])]


def test_no_input():
    """Tests that doing nothing before asking for output does something
    reasonable. NOTE: this may need to fail if we decide that every node must
    be measured
    """
    se = SingleEmitterMultiMeasure()
    se.next_node(0)

    fp = se.fusion_pattern()
    assert fp.path == [0]
    assert len(fp.measurements) == 0


def test_measuring_twice():
    """Measuring a node twice should fail"""
    se = SingleEmitterMultiMeasure()

    m = Measurement(1, "XY")
    se.next_node(0)
    se.measure(m)

    with pytest.raises(Exception):
        se.measure(m)


def test_no_leading_next_node():
    """Not starting the command sequence with a call to next_node() should
    fail"""
    se = SingleEmitterMultiMeasure()

    m = Measurement(1, "XY")

    with pytest.raises(Exception):
        se.measure(m)


def test_colliding_fusion_photons():
    """Tests two fusion photons can't be delayed to arrive back in the machine
    at the same time."""
    se = SingleEmitterMultiMeasure()

    se.next_node(0)
    se.delay_then_fuse(2)

    with pytest.raises(Exception):
        se.delay_then_fuse(1)


def test_mismatched_fusion():
    """Attempting a fusion without a delayed fusion photon should fail"""
    se = SingleEmitterMultiMeasure()

    with pytest.raises(Exception):
        se.fuse()


def test_measure_weird_order():
    """Tests nodes can be measured in an arbitrary order if we want."""
    se = SingleEmitterMultiMeasure()

    m = create_unique_measurements(4)

    # Measures the photons in reverse order
    se.next_node(0)
    se.delay_then_measure(30, m[0])
    se.delay_then_fuse(3)
    se.delay_then_fuse(4)
    se.next_node(1)
    se.delay_then_measure(20, m[1])
    se.next_node(2)
    se.fuse()
    se.delay_then_measure(10, m[2])
    se.next_node(3)
    se.fuse()
    se.delay_then_measure(1, m[3])

    fp = se.fusion_pattern()
    assert fp.path == [0, 1, 2, 3]
    assert fp.measurements == [(3, m[3]), (2, m[2]), (1, m[1]), (0, m[0])]


def test_multi_fusion():
    """Constructs a graph with a node that has multiple fusions"""
    se = SingleEmitterMultiMeasure()

    m = create_unique_measurements(4)

    se.next_node(0)
    se.measure(m[0])
    se.delay_then_fuse(3)
    se.delay_then_fuse(4)
    se.next_node(1)
    se.measure(m[1])
    se.next_node(2)
    se.fuse()
    se.measure(m[2])
    se.next_node(3)
    se.fuse()
    se.measure(m[3])

    fp = se.fusion_pattern()
    assert fp.path == [0, 1, 2, 3]
    assert fp.measurements == [(0, m[0]), (1, m[1]), (2, m[2]), (3, m[3])]
