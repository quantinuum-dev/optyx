from optyx.circuit.qmode.lo import Phase, HadamardBS
from optyx.circuit.qmode.measurement import NumberResolvingMeasurement
from optyx.circuit.classical.classical_circuit import DiscardMode
from optyx.diagram.channel import qmode, mode
from optyx.diagram.optyx import Swap, Mode, Bit
from optyx.diagram.lo import BS_hadamard
from optyx.diagram.channel import Channel, qmode
from optyx.circuit.classical.classical_functions import ClassicalFunction

ZMeasurement = lambda alpha: (
    qmode @ Phase(alpha) >>
    HadamardBS >>
    NumberResolvingMeasurement(2) >>
    DiscardMode(1) @ mode
)

XMeasurement = lambda alpha: (
    HadamardBS >>
    ZMeasurement(alpha)
)


kraus_map_fusion_I = (
    Mode(1) @ Swap(Mode(1), Mode(1)) @ Mode(1) >>
    Mode(1) @ BS_hadamard @ Mode(1) >>
    Mode(2) @ Swap(Mode(1), Mode(1)) >>
    Mode(1) @ Swap(Mode(1), Mode(1)) @ Mode(1)
)

fusion_I = Channel(
    "Fusion", kraus_map_fusion_I
)

def fusion_I_function(x):
    """
    A classical function that returns two bits based on an input x,
    based on the classical logical for the Fusion type II circuit.
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

fusion_type_I = (
    fusion_I >>
    qmode**2 @ NumberResolvingMeasurement(2) >>
    qmode**2 @ classical_function_I
)

fusion_II = Channel(
    "Fusion",
    (
        BS_hadamard @ BS_hadamard >>
        kraus_map_fusion_I >>
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
    c = x[2]
    d = x[3]
    s = (a % 2) ^ (b % 2)
    k = int(s*(b + d) + (1-s)*(1 - (a + b)/2))%2
    return [s, k]

classical_function_II = ClassicalFunction(
    fusion_II_function,
    Mode(4),
    Bit(2)
)

fusion_type_II = (
    fusion_II >>
    NumberResolvingMeasurement(4) >>
    classical_function_II
)
