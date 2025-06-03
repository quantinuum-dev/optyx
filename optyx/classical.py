import numpy as np

from typing import Callable, List
from optyx.core import (
    channel,
    control,
    zw,
    zx,
    diagram,
    path
)

class BitControlledGate(channel.Channel):
    """
    Represents a gate that is
    controlled by a classical bit.
    It uses a `BitControlledBox` to define
    the Kraus operators for the gate.
    """
    def __init__(self,
                 control_gate,
                 default_gate=None):
        if isinstance(control_gate, (channel.Diagram, channel.Channel)):
            assert control_gate.is_pure, \
                 "The input gates must be pure quantum channels"
            control_gate_single = control_gate.get_kraus()
        if isinstance(default_gate, (channel.Diagram, channel.Channel)):
            assert default_gate.is_pure, \
                 "The input gates must be pure quantum channels"
            default_gate = default_gate.get_kraus()
        kraus = control.BitControlledBox(control_gate_single, default_gate)
        super().__init__(
            f"BitControlledGate({control_gate}, {default_gate})",
            kraus,
            channel.bit @ control_gate.dom,
            control_gate.cod
        )


class BitControlledPhaseShift(channel.Channel):
    """
    Represents a phase shift operation
    that is controlled by classical bits.
    It uses a `ControlledPhaseShiftBox` to
    define the Kraus operators for the phase shift.
    """
    def __init__(self,
                 function: Callable[[List[int]], List[int]],
                 n_modes: int = 1,
                 n_control_modes: int = 1):
        kraus = control.ControlledPhaseShift(function, n_modes, n_control_modes)
        super().__init__(
            "BitControlledPhaseShift",
            kraus,
            channel.mode**n_control_modes @ channel.qmode**n_modes,
            channel.mode**n_modes,
        )


DiscardBit = lambda n: channel.Discard(channel.bit**n)
DiscardMode = lambda n: channel.Discard(channel.mode**n)


class ClassicalBox(channel.CQMap):
    pass


class Scalar(ClassicalBox):
    def __init__(self, value):
        super().__init__(
            f"{value}",
            diagram.Scalar(value),
            channel.bit**0,
            channel.bit**0
        )


class AddN(ClassicalBox):
    """
    Classical addition of n natural numbers.
    The domain of the map is n modes.
    The map will perform addition on the basis states.
    """
    def __init__(self, n):
        super().__init__(
            f"AddInt({n})",
            zw.Add(n),
            channel.mode**n,
            channel.mode
        )


class SubN(ClassicalBox):
    """
    Classical subtraction: subtract the first number from the second.
    The domain of the map is 2 modes.
    The map will perform subtraction on the basis states.
    If the result is negative, it will return a 0 map.
    """
    def __init__(self):
        super().__init__(
            "SubInt",
            (
                zw.Add(2).dagger() @ diagram.Mode(1) >>
                diagram.Mode(1) @ diagram.Spider(2, 0, diagram.Mode(1))
            ).
            channel.mode**2,
            channel.mode
        )


class MultiplyN(ClassicalBox):
    """
    Classical multiplication of 2 natural numbers.
    The domain of the map is 2 modes.
    The map will perform multiplication on the basis states.
    """
    def __init__(self):
        super().__init__(
            "MultiplyInt",
            zw.Multiply(),
            channel.mode**2,
            channel.mode
        )


class DivideN(ClassicalBox):
    """
    Classical division: divide the first number by the second.
    The domain of the map is 2 modes.
    The map will perform division on the basis states.
    If the result is not an integer, it will return a 0 map.
    """
    def __init__(self):
        super().__init__(
            "DivideInt",
            zw.Divide(),
            channel.mode**2,
            channel.mode
        )


class ModN(ClassicalBox):
    """
    Classical modulo 2.
    The domain of the map is a mode.
    The codomain of the map is a bit.
    The map will perform modulo 2 on the basis states.
    """
    def __init__(self):
        super().__init__(
            "ModInt",
            zw.Mod2(),
            channel.mode,
            channel.bit
        )


class CopyN(ClassicalBox):
    """
    Classical copy of n natural numbers.
    The domain of the map is a mode.
    The codomain of the map is n modes.
    The map will perform copy on the basis states.
    """
    def __init__(self, n):
        super().__init__(
            f"CopyInt({n})",
            lambda n: diagram.Spider(1, n, diagram.Mode(1)),
            channel.mode,
            channel.mode**n
        )


class SwapN(ClassicalBox):
    """
    Classical swap of 2 natural numbers.
    The domain of the map is 2 modes.
    The codomain of the map is 2 modes.
    The map will perform swap on the basis states.
    """
    def __init__(self):
        super().__init__(
            "SwapInt",
            diagram.Swap(diagram.Mode(1), diagram.Mode(1)),
            channel.mode**2,
            channel.mode**2
        )


class PostselectBit(ClassicalBox):
    """
    Postselect on a bit result.
    The domain of the map is a bit.
    """
    def __init__(self, result):

        if result not in (0, 1):
            raise ValueError("Result must be 0 or 1.")
        if result == 0:
            super().__init__(
                f"PostselectBit(0)",
                zx.X(1, 0) @ diagram.Scalar(1 / np.sqrt(2)),
                channel.bit,
                channel.bit**0
            )
        else:
            super().__init__(
                f"PostselectBit(1)",
                zx.X(1, 0, 0.5) @ diagram.Scalar(1 / np.sqrt(2)),
                channel.bit,
                channel.bit**0
            )


class InitBit(ClassicalBox):
    """
    Initialize a bit to 0 or 1.
    The domain of the map is a bit.
    The codomain of the map is a bit.
    The map will perform initialization on the basis states.
    """
    def __init__(self, value):
        if value not in (0, 1):
            raise ValueError("Value must be 0 or 1.")
        if value == 0:
            super().__init__(
                f"InitBit(0)",
                zx.X(0, 1) @ diagram.Scalar(1 / np.sqrt(2)),
                channel.bit**0,
                channel.bit
            )
        else:
            super().__init__(
                f"InitBit(1)",
                zx.X(0, 1, 0.5) @ diagram.Scalar(1 / np.sqrt(2)),
                channel.bit**0,
                channel.bit
            )


class NotBit(ClassicalBox):
    """
    Classical NOT gate.
    The domain of the map is a bit.
    The codomain of the map is a bit.
    The map will perform NOT on the basis states.
    """
    def __init__(self):
        super().__init__(
            "NotBit",
            zx.X(1, 1, 0.5),
            channel.bit,
            channel.bit
        )


class XorBit(ClassicalBox):
    """
    Classical XOR gate.
    The domain of the map is n bits.
    The codomain of the map is a bit.
    The map will perform XOR on the basis states.
    """
    def __init__(self, n=2):
        super().__init__(
            f"XorBit({n})",
            zx.X(n, 1) @ diagram.Scalar(np.sqrt(n)),
            channel.bit**n,
            channel.bit
        )


class AndBit(ClassicalBox):
    """
    Classical AND gate.
    The domain of the map is 2 bits.
    The codomain of the map is a bit.
    The map will perform AND on the basis states.
    """
    def __init__(self, n=2):
        super().__init__(
            "AndBit",
            zx.And(n),
            channel.bit**2,
            channel.bit
        )


class CopyBit(ClassicalBox):
    """
    Classical copy of a bit.
    The domain of the map is a bit.
    The codomain of the map is n bits.
    The map will perform copy on the basis states.
    """
    def __init__(self, n=2):
        super().__init__(
            f"CopyBit({n})",
            zx.Z(1, n),
            channel.bit,
            channel.bit**n
        )


class SwapBit(ClassicalBox):
    """
    Classical swap of 2 bits.
    The domain of the map is 2 bits.
    The codomain of the map is 2 bits.
    The map will perform swap on the basis states.
    """
    def __init__(self):
        super().__init__(
            "SwapBit",
            diagram.Swap(diagram.Bit(1), diagram.Bit(1)),
            channel.bit**2,
            channel.bit**2
        )


class OrBit(ClassicalBox):
    """
    Classical OR gate.
    The domain of the map is n bits.
    The codomain of the map is a bit.
    The map will perform OR on the basis states.
    """
    def __init__(self, n=2):
        super().__init__(
            f"OrBit({n})",
            zx.Or(n),
            channel.bit**n,
            channel.bit
        )


class Z(ClassicalBox):
    """Z spider."""
    tikzstyle_name = "Z"
    color = "green"
    draw_as_spider = True

    def __init__(self, n_legs_in, n_legs_out, phase=0):
        kraus = zx.Z(n_legs_in, n_legs_out, phase)
        super().__init__(
            f"Z({phase})",
            kraus,
            channel.bit**n_legs_in,
            channel.bit**n_legs_out,
        )


class X(ClassicalBox):
    """X spider."""
    tikzstyle_name = "X"
    color = "red"
    draw_as_spider = True

    def __init__(self, n_legs_in, n_legs_out, phase=0):
        kraus = zx.X(n_legs_in, n_legs_out, phase)
        super().__init__(
            f"X({phase})",
            kraus,
            channel.bit**n_legs_in,
            channel.bit**n_legs_out,
        )


class H(ClassicalBox):
    """Hadamard spider."""
    tikzstyle_name = "H"
    color = "blue"
    draw_as_spider = True

    def __init__(self):
        kraus = zx.H()
        super().__init__(
            f"H",
            kraus,
            channel.bit,
            channel.bit,
        )

class ControlChannel(ClassicalBox):
    """
    Syntactic sugar.
    Converts a classical circuit (Diagram or Box)
    into a CQMap, allowing
    it to be used as a control channel in hybrid quantum-classical systems.
    """
    pass


class ClassicalFunction(ControlChannel):
    """
    A classical function box between modes or bits,
    mapping an input list of natural numbers or
    a list of bits to a list of
    natural numbers or a list of bits.

    Example
    -------
    >>> from optyx.classical import X, Scalar
    >>> xor = X(2, 1) @ Scalar(2**0.5)
    >>> f_res = (ClassicalFunction(lambda x: [x[0] ^ x[1]],
    ...         diagram.Bit(2),
    ...         diagram.Bit(1))).double().to_zw().to_tensor().eval().array
    >>> xor_res = xor.double().to_zw().to_tensor().eval().array
    >>> assert np.allclose(f_res, xor_res)
    """

    def __init__(self, function, dom, cod):
        box = control.ClassicalFunctionBox(
            function,
            dom,
            cod
        )
        return super().__init__(
            box.name,
            box,
            channel.Ty(
                *[channel.Ob._classical[ob.name] for ob in box.dom.inside]
            ),
            channel.Ty(
                *[channel.Ob._classical[ob.name] for ob in box.cod.inside]
            ),
        )


class BinaryMatrix(ControlChannel):
    """
    Represents a linear transformation over
    GF(2) using matrix multiplication.

    Example
    -------
    >>> from optyx.classical import X, Scalar
    >>> xor = X(2, 1) @ Scalar(2**0.5)
    >>> matrix = [[1, 1]]
    >>> m_res = BinaryMatrix(matrix).double().to_tensor().eval().array
    >>> xor_res = xor.double().to_zw().to_tensor().eval().array
    >>> assert np.allclose(m_res, xor_res)
    """

    def __init__(self, matrix):
        box = control.BinaryMatrixBox(matrix)
        return super().__init__(
            box.name,
            box,
            channel.Ty(
                *[channel.Ob._classical[ob.name] for ob in box.dom.inside]
            ),
            channel.Ty(
                *[channel.Ob._classical[ob.name] for ob in box.cod.inside]
            ),
        )


class Select(channel.Channel):
    def __init__(self, *photons: int):
        self.photons = photons
        super().__init__(
            f"Select({photons})",
            zw.Select(*photons)
        )

    def to_path(self, dtype=complex) -> path.Matrix:
        array = np.eye(len(self.photons))
        return path.Matrix[dtype](
            array, len(self.photons), 0, selections=self.photons
        )