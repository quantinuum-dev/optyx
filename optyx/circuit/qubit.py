from optyx.diagram.channel import (
    Measure,
    Encode,
    qubit,
    bit,
    Discard,
    Channel,
    Circuit,
)
from optyx.diagram.zx import (
    X as XSingle,
    Z as ZSingle,
    H as HSingle,
    decomp,
    zx2path
)
from optyx.utils import explode_channel
from optyx.diagram.zw import Scalar as ScalarSingle

class MeasureQubits(Measure):
    """
    Ideal qubit measurement (in computational basis) from qubit to bit.
    """

    def __init__(self, n):
        super().__init__(
            qubit**n
        )


class DiscardQubits(Discard):
    """
    Discard :math:`n` qubits.
    """

    def __init__(self, n):
        super().__init__(
            qubit**n
        )


class EncodeBits(Encode):
    """
    Encode :math:`n` bits into :math:`n` qubits.
    """

    def __init__(self, n):
        super().__init__(
            bit**n
        )


class QubitChannel(Channel):
    """Qubit channel."""

    def decomp(self):
        decomposed = decomp(self.kraus)
        return explode_channel(
            decomposed,
            QubitChannel,
            Circuit
        )

    def to_dual_rail(self):
        """Convert to dual-rail encoding."""
        kraus_path = zx2path(self.kraus)
        return explode_channel(
            kraus_path,
            Channel,
            Circuit
        )


class Z(QubitChannel):
    """Z spider."""

    tikzstyle_name = "Z"
    color = "green"
    draw_as_spider = True

    def __init__(self, n_legs_in, n_legs_out, phase=0):
        kraus = ZSingle(n_legs_in, n_legs_out, phase)
        super().__init__(
            f"Z({phase})",
            kraus,
            qubit**n_legs_in,
            qubit**n_legs_out,
        )


class X(QubitChannel):
    """X spider."""

    tikzstyle_name = "X"
    color = "red"
    draw_as_spider = True

    def __init__(self, n_legs_in, n_legs_out, phase=0):
        kraus = XSingle(n_legs_in, n_legs_out, phase)
        super().__init__(
            f"X({phase})",
            kraus,
            qubit**n_legs_in,
            qubit**n_legs_out,
        )


class H(QubitChannel):
    """Hadamard gate."""

    tikzstyle_name = "H"
    color = "yellow"

    def __init__(self):
        super().__init__(
            "H",
            HSingle(),
            qubit,
            qubit,
        )


class Scalar(QubitChannel):
    def __init__(self, value: float):
        super().__init__(
            f"Scalar({value})",
            ScalarSingle(value),
            qubit**0,
            qubit**0,
        )

