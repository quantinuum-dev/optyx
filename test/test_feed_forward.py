import pytest
import numpy as np
import itertools
import copy as cp

from optyx.feed_forward.classical_control import *
from optyx.feed_forward.measurement import *
from optyx.feed_forward.controlled_gates import *
from optyx.zw import Create, W
from optyx.optyx import PhotonThresholdDetector, Mode, Swap, Scalar, DualRail, Id
from optyx.lo import Phase, BS, MZI, BS_hadamard
from optyx.utils import matrix_to_zw
from optyx.zx import X, Z

# Helper Functions and Data

def assert_allclose_with_shape_mismatch(res_1, res_2):
    """
    Assert that res_1 and res_2 are elementwise close, allowing for shape mismatches
    by comparing only on the overlapping slice.
    """
    if res_1.shape != res_2.shape:
        min_shape = [min(r1, r2) for r1, r2 in zip(res_1.shape, res_2.shape)]
        slices = tuple(slice(0, s) for s in min_shape)
        assert np.allclose(res_1[slices], res_2[slices])
    else:
        assert np.allclose(res_1, res_2)



# Boolean classical functions

@pytest.mark.skip(reason="Helper function for testing")
def xor_2bits(x):
    """Return [x[0] ^ x[1]]."""
    return [x[0] ^ x[1]]

@pytest.mark.skip(reason="Helper function for testing")
def duplicate_bit(x):
    """Return [x[0], x[0]]."""
    return [x[0], x[0]]

@pytest.mark.skip(reason="Helper function for testing")
def swap_and_xor(x):
    """Return [x[0] ^ x[1], x[0]]."""
    return [x[0] ^ x[1], x[0]]

@pytest.mark.skip(reason="Helper function for testing")
def xor_both_bits(x):
    """Return [x[0] ^ x[1], x[0] ^ x[1]]."""
    return [x[0] ^ x[1], x[0] ^ x[1]]

@pytest.mark.skip(reason="Helper function for testing")
def complicated_xor(x):
    """
    Swap bits 2 and 3, then compute a few XORs. If xor_1 == 0 or xor_3 == 0,
    return 0; otherwise return [xor_2].
    """
    bitstring = cp.deepcopy(x)
    bitstring[2] = x[3]
    bitstring[3] = x[2]

    xor_1 = bitstring[0] ^ bitstring[1]
    xor_2 = bitstring[1] ^ bitstring[2]
    xor_3 = bitstring[2] ^ bitstring[3]

    if xor_1 == 0:
        return 0
    if xor_3 == 0:
        return 0
    else:
        return [xor_2]


# Real-valued classical functions (for Phase shifts)

@pytest.mark.skip(reason="Helper function for testing")
def real_f_1(x):
    x_val = x[0]
    return [x_val*0.234, x_val*0.912, x_val*0.184]

@pytest.mark.skip(reason="Helper function for testing")
def real_f_2(x):
    return [0.23*x[0]]

@pytest.mark.skip(reason="Helper function for testing")
def real_f_3(x):
    _ = x[0]
    return [0.8362, 0.193, 0.654]


# Parametrized data sets

CIRCUITS_TO_TEST = [
    (Phase(0.1), None),
    (Phase(0.456), Phase(0.8765)),
    (BS, None),
    (BS, MZI(0.324, 0.9875)),
    (W(2).dagger() >> W(2), MZI(0.324, 0.9875)),
]

CLASSICAL_FUNCTIONS_TO_TEST_MATRIX = (
    (xor_2bits, [1, 1]),
    (duplicate_bit, [[1],
                     [1]]),
    (swap_and_xor, [[1, 1],
                    [1, 0]]),
    (xor_both_bits, [[1, 1],
                     [1, 1]]),
)

REAL_FUNCS = [real_f_1, real_f_2, real_f_3]


# Test Classes

class TestBinaryControlledBox:
    """
    Tests for the BitControlledBox, verifying correct action vs. default
    based on a created photon (1) or not (0).
    """

    @pytest.mark.parametrize("action, default", CIRCUITS_TO_TEST)
    def test_binary_controlled_box(self, action, default):
        action_result = action.to_zw().to_tensor().eval().array
        default_result = default.to_zw().to_tensor().eval().array if default is not None else None

        if default is None:
            action_test = (
                (Create(1) >> PhotonThresholdDetector()) @ Mode(len(action.cod))
                >> BitControlledBox(action)
            ).to_zw().to_tensor().eval().array
            default_test = (
                (Create(0) >> PhotonThresholdDetector()) @ Mode(len(action.cod))
                >> BitControlledBox(action)
            ).to_zw().to_tensor().eval().array
        else:
            action_test = (
                (Create(1) >> PhotonThresholdDetector()) @ Mode(len(action.cod))
                >> BitControlledBox(action, default.to_zw())
            ).to_zw().to_tensor().eval().array
            default_test = (
                (Create(0) >> PhotonThresholdDetector()) @ Mode(len(action.cod))
                >> BitControlledBox(action, default.to_zw())
            ).to_zw().to_tensor().eval().array

        assert np.allclose(action_result, action_test)
        if default is not None:
            assert np.allclose(default_result, default_test)

    @pytest.mark.parametrize("action, default", CIRCUITS_TO_TEST)
    def test_binary_controlled_box_dagger(self, action, default):
        res_1 = BitControlledBox(action, default).to_zw().to_tensor().dagger().eval().array
        res_2 = BitControlledBox(action, default).dagger().to_zw().to_tensor().eval().array
        assert_allclose_with_shape_mismatch(res_1, res_2)


class TestClassicalFunctionBox:
    """
    Tests that a ClassicalFunctionBox(f, dom, cod) yields the same matrix
    as the known symbolic circuit or matrix box.
    """

    @pytest.mark.parametrize("function, m", CLASSICAL_FUNCTIONS_TO_TEST_MATRIX)
    def test_logical_matrix_box(self, function, m):
        m = np.array(m)
        if len(m.shape) == 1:
            m = m.reshape(1, -1)
        cod = Bit(len(m))
        dom = Bit(len(m[0]))

        f_arr = ClassicalFunctionBox(function, dom, cod).to_tensor().eval().array
        circ_arr = BinaryMatrixBox(m).to_tensor().eval().array
        assert np.allclose(f_arr, circ_arr)

    @pytest.mark.parametrize("function, m", CLASSICAL_FUNCTIONS_TO_TEST_MATRIX)
    def test_logical_matrix_box_dagger(self, function, m):
        """
        Verify dagger consistency for BinaryMatrixBox.
        """
        m = np.array(m)
        if len(m.shape) == 1:
            m = m.reshape(1, -1)
        cod = Bit(len(m))
        dom = Bit(len(m[0]))

        res_1 = BinaryMatrixBox(m).to_tensor().dagger().eval().array
        res_2 = BinaryMatrixBox(m).dagger().to_tensor().eval().array
        assert_allclose_with_shape_mismatch(res_1, res_2)


class TestControlledPhaseShift:
    """
    Tests involving real-valued classical functions for phase shifts and
    verifying dagger consistency.
    """

    @pytest.mark.parametrize("f", REAL_FUNCS)
    @pytest.mark.parametrize("diagram_creator", [
        lambda f: ControlledPhaseShift(f, len(f([0]))),
        lambda f: Create(4) @ Mode(len(f([0]))) >> ControlledPhaseShift(f, len(f([0])))
    ])
    def test_dagger_controlled_phase_shift(self, f, diagram_creator):
        """
        Confirm that taking the dagger at the diagram level is consistent
        with converting the diagram to a tensor, then taking dagger.
        """
        diagram = diagram_creator(f)
        res_1 = diagram.to_zw().to_tensor().dagger().eval().array
        res_2 = diagram.dagger().to_zw().to_tensor().eval().array
        assert_allclose_with_shape_mismatch(res_1, res_2)

    diagrams_to_test_2 = [
        ControlledPhaseShift(lambda x: [x[0]], 1),
        ControlledPhaseShift(lambda x: [0.23*x[0], 0.456*x[0], 0.876*x[0], 0.654*x[0]], 4),
        ControlledPhaseShift(lambda x: [0.23*x[0]], 1),
        ControlledPhaseShift(lambda x: [0.23*x[0], 0.456*x[0], 0.876*x[0]], 3),
    ]

    @pytest.mark.parametrize("diagram", diagrams_to_test_2)
    def test_dagger_controlled_phase_shift_2(self, diagram):
        """
        Additional tests for ControlledPhaseShift, verifying that dagger
        consistency holds.
        """
        res_1 = diagram.to_zw().to_tensor().dagger().eval().array
        res_2 = diagram.dagger().to_zw().to_tensor().eval().array
        assert_allclose_with_shape_mismatch(res_1, res_2)

    xs = range(5)

    @pytest.mark.parametrize("f", REAL_FUNCS)
    @pytest.mark.parametrize("x", xs)
    def test_controlled_phase_shift_numeric(self, f, x):
        """
        For each real-valued function f and input x, build a circuit with
        ControlledPhaseShift, compare to a ZBox with the matching phase exponent.
        """
        n = len(f([0]))
        diag = Create(x) @ Mode(n) >> ControlledPhaseShift(f, n)

        # Build the "manual" ZBox product
        zbox = Id(Mode(0))
        for y in f([x]):
            zbox @= ZBox(1, 1, lambda i, y=y: np.exp(2 * np.pi * 1j * y) ** i)

        # Check forward and dagger equivalences
        assert np.allclose(zbox.to_tensor().eval().array, diag.to_tensor().eval().array)
        assert np.allclose(zbox.dagger().to_tensor().eval().array, diag.dagger().to_tensor().eval().array)
        assert np.allclose(
            zbox.to_tensor().dagger().eval().array,
            zbox.dagger().to_tensor().eval().array
        )


class TestPhotonThresholdDetector:
    """
    Tests verifying that PhotonThresholdDetector's dagger
    is consistent with the array representation.
    """

    circuits_to_test = [
        PhotonThresholdDetector()
    ]

    @pytest.mark.parametrize("circ", circuits_to_test)
    def test_photon_threshold_detector_dagger(self, circ):
        res_1 = circ.to_zw().to_tensor().dagger().eval().array
        res_2 = circ.dagger().to_zw().to_tensor().eval().array
        assert_allclose_with_shape_mismatch(res_1, res_2)


class TestMatrixToZW:
    """
    A sanity check that matrix_to_zw(...) yields correct transformations.
    """

    def test_matrix_to_zw(self):
        U = np.array([[1, 1], [1, 1]])
        diagram1 = matrix_to_zw(U)

        diagram2 = (
            W(2) @ W(2) >>
            Mode(1) @ Swap(Mode(1), Mode(1)) @ Mode(1) >>
            W(2).dagger() @ W(2).dagger()
        )

        assert np.allclose(diagram1.to_tensor().eval().array,
                           diagram2.to_tensor().eval().array)


def test_teleportation():
    kraus_map_fusion = (
        Mode(1) @ Swap(Mode(1), Mode(1)) @ Mode(1) >>
        Mode(1) @ BS_hadamard @ Mode(1) >>
        Swap(Mode(1), Mode(1)) @ Mode(1) @ Mode(1) >>
        Mode(1) @ Swap(Mode(1), Mode(1)) @ Mode(1) >>
        Mode(1) @ Mode(1) @ BS_hadamard
    )

    fusion = Channel(
        "Fusion",
        kraus_map_fusion
    )

    def fusion_function(x):
        a = x[0]
        b = x[1]
        c = x[2]
        d = x[3]
        s = (a % 2) ^ (b % 2)
        k = int(s*(b + d) + (1-s)*(1 - (a + b)/2))%2
        return [s, k]

    classical_function = ControlChannel(
        ClassicalFunctionBox(
            fusion_function,
            Mode(4),
            Bit(2)
        )
    )

    postselect_1 = postselect_1 = X(1, 0, 0.5) @ Scalar(0.5**0.5)

    from optyx.channel import (
        bit,
        qmode
    )

    fusion_failure_processing = ControlChannel(
        postselect_1
    )

    correction = Channel(
        "Phase Correction",
        BitControlledBox(
            Phase(0.5) @ Mode(1)
        ),
        dom = bit @ qmode**2
    )

    channel_bell = Channel(
        "Bell pair preparation",
        Z(0, 2) @ Scalar(0.5**0.5) >> DualRail() @ DualRail()
    )

    dual_rail_input = Channel(
        "Dual rail of input",
        DualRail()
    )

    teleportation = (
        dual_rail_input @ channel_bell >>
        fusion @ qmode**2 >>
        Measure(qmode**4) @ qmode**2 >>
        classical_function @ qmode**2 >>
        fusion_failure_processing @ correction >>
        Channel("Dual rail projection", DualRail().dagger())
    )

    array_teleportation = teleportation.double().to_zw().to_tensor().eval().array

    array_id = Channel(
        "Identity",
        Id(Bit(1)) @ Scalar(0.5**0.5)
    ).double().to_zw().to_tensor().eval().array

    assert np.allclose(array_teleportation, array_id)