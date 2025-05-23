from optyx.diagram.classical_arithmetic import *
from optyx.diagram.channel import (
    bit,
    mode,
    CQMap,
    Discard
)
from optyx.diagram.zx import (
    Z as ZSingle,
    X as XSingle,
    H as HSingle,
)
from optyx.diagram.zw import Scalar as ScalarSingle


DiscardBit = lambda n: Discard(bit**n)
DiscardMode = lambda n: Discard(mode**n)


class ClassicalBox(CQMap):
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
            add_N(n),
            mode**n,
            mode
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
            subtract_N,
            mode**2,
            mode
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
            multiply_N,
            mode**2,
            mode
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
            divide_N,
            mode**2,
            mode
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
            mod2,
            mode,
            bit
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
            copy_N(n),
            mode,
            mode**n
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
            swap_N,
            mode**2,
            mode**2
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
                postselect_0,
                bit,
                bit**0
            )
        else:
            super().__init__(
                f"PostselectBit(1)",
                postselect_1,
                bit,
                bit**0
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
                init_0,
                bit**0,
                bit
            )
        else:
            super().__init__(
                f"InitBit(1)",
                init_1,
                bit**0,
                bit
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
            not_bit,
            bit,
            bit
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
            xor_bits(n),
            bit**n,
            bit
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
            and_bit(n),
            bit**2,
            bit
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
            copy_bit(n),
            bit,
            bit**n
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
            swap_bits,
            bit**2,
            bit**2
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
            or_bit(n),
            bit**n,
            bit
        )


class Z(ClassicalBox):
    """Z spider."""
    tikzstyle_name = "Z"
    color = "green"
    draw_as_spider = True

    def __init__(self, n_legs_in, n_legs_out, phase=0):
        kraus = ZSingle(n_legs_in, n_legs_out, phase)
        super().__init__(
            f"Z({phase})",
            kraus,
            bit**n_legs_in,
            bit**n_legs_out,
        )


class X(ClassicalBox):
    """X spider."""
    tikzstyle_name = "X"
    color = "red"
    draw_as_spider = True

    def __init__(self, n_legs_in, n_legs_out, phase=0):
        kraus = XSingle(n_legs_in, n_legs_out, phase)
        super().__init__(
            f"X({phase})",
            kraus,
            bit**n_legs_in,
            bit**n_legs_out,
        )


class H(ClassicalBox):
    """Hadamard spider."""
    tikzstyle_name = "H"
    color = "blue"
    draw_as_spider = True

    def __init__(self):
        kraus = HSingle()
        super().__init__(
            f"H",
            kraus,
            bit,
            bit,
        )


class Scalar(ClassicalBox):
    def __init__(self, value: float):
        super().__init__(
            f"Scalar({value})",
            ScalarSingle(value),
            bit**0,
            bit**0,
        )