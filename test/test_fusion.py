from optyx.photonic import DualRail, FusionTypeI, FusionTypeII, Phase, HadamardBS
from optyx.qubits import Z, Scalar
from optyx.core.channel import qmode, bit
from optyx.classical import PostselectBit, BitControlledGate
from optyx.photonic import Id
from optyx.core import channel
import numpy as np

def test_fusion_ii():
    correction = BitControlledGate(
        HadamardBS() >>
        (Phase(0.5) @ qmode) >>
        HadamardBS()
    )

    channel_bell = (
        Z(0, 2) @ Scalar(0.5**0.5) >> DualRail(1) @ DualRail(1)
    )

    teleportation = (
        DualRail(1) @ channel_bell >>
        FusionTypeII() @ qmode**2 >>
        PostselectBit(1) @ correction >>
        DualRail(1).dagger()
    )

    array_teleportation = (teleportation.double().to_tensor().to_quimb()^...).data

    array_id = (
        Id(1) @ Scalar(0.5**0.5)
    ).double().to_tensor().eval().array

    assert np.allclose(array_teleportation, array_id)


def test_fusion_i():
    correction = BitControlledGate(
        Phase(0.5) @ qmode
    )

    spider = (
        DualRail(1) @ DualRail(1) >>
        FusionTypeI() >>
        qmode**2 @ PostselectBit(1) @ bit >>
        qmode @ channel.Swap(qmode, bit) >>
        channel.Swap(qmode, bit) @ qmode >>
        correction >>
        DualRail(1).dagger()
    )

    spider_arr = (spider.double().to_tensor().to_quimb()^...).data
    spider_logical_arr = Z(2, 1).double().to_tensor().eval().array
    assert np.allclose(spider_arr, spider_logical_arr)