from optyx.diagram.channel import Discard, bit, mode
from optyx.diagram.zx import (
    Z as ZSingle,
    X as XSingle,
    H as HSingle,
)
from optyx.diagram.channel import CQMap

DiscardBit = lambda n: Discard(bit**n)
DiscardMode = lambda n: Discard(mode**n)


class ClassicalBox(CQMap):
    pass


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