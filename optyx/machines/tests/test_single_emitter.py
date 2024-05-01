import pytest

from optyx.machines.single_emitter import (
    SingleEmitterMultiMeasure,
    Measurement,
)


def test_single_path():
    """Tests a one-node graph state with a single measurement is handled"""
    se = SingleEmitterMultiMeasure()

    m = Measurement(1)
    se.measure(m)

    fp = se.fusion_pattern()
    assert fp.path_length == 1
    assert fp.measurements == [(1, m)]


def test_triangle():
    """Constructs a triangle via a fusion"""
    se = SingleEmitterMultiMeasure()

    m1 = Measurement(1)
    m2 = Measurement(2)
    m3 = Measurement(3)

    se.measure(m1)
    se.delay_then_fuse(2)
    se.next_node()
    se.measure(m2)
    se.next_node()
    se.fuse()
    se.measure(m3)

    fp = se.fusion_pattern()
    assert fp.path_length == 3
    assert fp.measurements == [(1, m1), (2, m2), (3, m3)]


def test_no_input():
    """Tests that doing nothing before asking for output does something
    reasonable. NOTE: this may need to fail if we decide that every node must
    be measured
    """
    se = SingleEmitterMultiMeasure()
    fp = se.fusion_pattern()
    assert fp.path_length == 1
    assert len(fp.measurements) == 0


def test_measuring_twice():
    """Measuring a node twice should fail"""
    se = SingleEmitterMultiMeasure()

    m = Measurement(1)
    se.measure(m)

    with pytest.raises(Exception):
        se.measure(m)


def test_colliding_fusion_photons():
    """Tests two fusion photons can't be delayed to arrive back in the machine
    at the same time."""
    se = SingleEmitterMultiMeasure()

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

    m1 = Measurement(1)
    m2 = Measurement(2)
    m3 = Measurement(3)
    m4 = Measurement(4)

    # Measures the photons in reverse order
    se.delay_then_measure(30, m1)
    se.delay_then_fuse(3)
    se.delay_then_fuse(4)
    se.next_node()
    se.delay_then_measure(20, m2)
    se.next_node()
    se.fuse()
    se.delay_then_measure(10, m3)
    se.next_node()
    se.fuse()
    se.delay_then_measure(1, m4)

    fp = se.fusion_pattern()
    assert fp.path_length == 4
    assert fp.measurements == [(4, m4), (3, m3), (2, m2), (1, m1)]


def test_multi_fusion():
    """Constructs a graph with a node that has multiple fusions"""
    se = SingleEmitterMultiMeasure()

    m1 = Measurement(1)
    m2 = Measurement(2)
    m3 = Measurement(3)
    m4 = Measurement(4)

    se.measure(m1)
    se.delay_then_fuse(3)
    se.delay_then_fuse(4)
    se.next_node()
    se.measure(m2)
    se.next_node()
    se.fuse()
    se.measure(m3)
    se.next_node()
    se.fuse()
    se.measure(m4)

    fp = se.fusion_pattern()
    assert fp.path_length == 4
    assert fp.measurements == [(1, m1), (2, m2), (3, m3), (4, m4)]
