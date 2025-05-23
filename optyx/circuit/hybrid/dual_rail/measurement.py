from optyx.circuit.qmode.lo import Phase, HadamardBS
from optyx.circuit.qmode.measurement import NumberResolvingMeasurement
from optyx.circuit.classical.classical_arithmetic import DiscardMode
from optyx.diagram.channel import qmode, mode
from optyx.diagram.optyx import Swap, Mode, Bit
from optyx.diagram.lo import BS_hadamard
from optyx.diagram.channel import Channel, qmode, Circuit
from optyx.circuit.classical.classical_functions import ClassicalFunction


class ZMeasurement(Circuit):
    def __new__(cls, alpha):
        """
        ZMeasurement circuit that performs a measurement in the Z basis
        after applying a phase shift of alpha.
        """
        return (
            qmode @ Phase(alpha) >>
            HadamardBS >>
            NumberResolvingMeasurement(2) >>
            DiscardMode(1) @ mode
        )


class XMeasurement(Circuit):
    def __new__(cls, alpha):
        """
        XMeasurement circuit that performs a measurement in the X basis
        after applying a Hadamard beam splitter.
        """
        return (
            HadamardBS >>
            ZMeasurement(alpha)
        )


class FusionTypeI(Circuit):
    def __new__(cls):
        kraus_map_fusion_I = (
            Mode(1) @ Swap(Mode(1), Mode(1)) @ Mode(1) >>
            Mode(1) @ BS_hadamard @ Mode(1) >>
            Mode(2) @ Swap(Mode(1), Mode(1)) >>
            Mode(1) @ Swap(Mode(1), Mode(1)) @ Mode(1)
        )

        fusion_I = Channel(
            "Fusion I", kraus_map_fusion_I
        )

        def fusion_I_function(x):
            """
            A classical function that returns two bits based on an input x,
            based on the classical logical for the Fusion type I circuit.
            """
            a = x[0]
            b = x[1]
            s = (a % 2) ^ (b % 2)
            k = int(s*b + (1-s)*(1 - (a + b)/2))%2
            return [s, k]

        classical_function_I = ClassicalFunction(
            fusion_I_function,
            Mode(2),
            Bit(2)
        )

        return (
            fusion_I >>
            qmode**2 @ NumberResolvingMeasurement(2) >>
            qmode**2 @ classical_function_I
        )


class FusionTypeII(Circuit):
    def __new__(cls):
        fusion_II = Channel(
            "Fusion II",
            (
                BS_hadamard @ BS_hadamard >>
                Mode(1) @ Swap(Mode(1), Mode(1)) @ Mode(1) >>
                Mode(1) @ BS_hadamard @ Mode(1) >>
                Mode(2) @ Swap(Mode(1), Mode(1)) >>
                Mode(1) @ Swap(Mode(1), Mode(1)) @ Mode(1) >>
                BS_hadamard @ Mode(2)
            )
        )

        def fusion_II_function(x):
            """
            A classical function that returns two bits based on an input x,
            based on the classical logical for the Fusion type II circuit.
            """
            a = x[0]
            b = x[1]
            d = x[3]
            s = (a % 2) ^ (b % 2)
            k = int(s*(b + d) + (1-s)*(1 - (a + b)/2))%2
            return [s, k]

        classical_function_II = ClassicalFunction(
            fusion_II_function,
            Mode(4),
            Bit(2)
        )

        return (
            fusion_II >>
            NumberResolvingMeasurement(4) >>
            classical_function_II
        )
