from typing import Callable, List
from optyx.core import (
    channel,
    classical,
    zw,
    zx,
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
        kraus = classical.BitControlledBox(control_gate_single, default_gate)
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
        kraus = classical.ControlledPhaseShift(function, n_modes, n_control_modes)
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


class AddN(ClassicalBox):
    """
    Classical addition of n natural numbers.
    The domain of the map is n modes.
    The map will perform addition on the basis states.
    """
    def __init__(self, n):
        super().__init__(
            f"AddInt({n})",
            classical.add_N(n),
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
            classical.ubtract_N,
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
            classical.multiply_N,
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
            classical.divide_N,
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
            classical.mod2,
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
            classical.copy_N(n),
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
            classical.swap_N,
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
                classical.postselect_0,
                channel.bit,
                channel.bit**0
            )
        else:
            super().__init__(
                f"PostselectBit(1)",
                classical.postselect_1,
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
                classical.init_0,
                channel.bit**0,
                channel.bit
            )
        else:
            super().__init__(
                f"InitBit(1)",
                classical.init_1,
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
            classical.not_bit,
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
            classical.xor_bits(n),
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
            classical.and_bit(n),
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
            classical.copy_bit(n),
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
            classical.swap_bits,
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
            classical.or_bit(n),
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


class Scalar(ClassicalBox):
    def __init__(self, value: float):
        super().__init__(
            f"Scalar({value})",
            zw.Scalar(value),
            channel.bit**0,
            channel.bit**0,
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
    >>> from optyx.zx import X
    >>> from optyx.optyx import Scalar
    >>> xor = X(2, 1) @ Scalar(np.sqrt(2))
    >>> f_res = ClassicalFunctionBox(lambda x: [x[0] ^ x[1]],
    ...         Bit(2),
    ...         Bit(1)).to_zw().to_tensor().eval().array
    >>> xor_res = xor.to_zw().to_tensor().eval().array
    >>> assert np.allclose(f_res, xor_res)
    """

    def __init__(self, function, dom, cod):
        box = classical.ClassicalFunctionBox(
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
    >>> from optyx.zx import X
    >>> from optyx.optyx import Scalar
    >>> xor = X(2, 1) @ Scalar(np.sqrt(2))
    >>> matrix = [[1, 1]]
    >>> m_res = BinaryMatrixBox(matrix).to_tensor().eval().array
    >>> xor_res = xor.to_zw().to_tensor().eval().array
    >>> assert np.allclose(m_res, xor_res)
    """

    def __init__(self, matrix):
        box = classical.BinaryMatrixBox(self, matrix)
        return super().__new__(
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
    def __init__(self, *n_photons: int):
        super().__init__(
            f"Select({n_photons})",
            zw.Scalar(*n_photons)
        )