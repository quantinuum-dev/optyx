from optyx.channel import (
    Measure,
    Encode,
    qubit,
    bit,
    Discard,
    Channel,
    Circuit,
)
from optyx.diagram import optyx
from optyx.zx import (
    X as XSingle,
    Z as ZSingle,
    H as HSingle,
    decomp,
    zx2path
)
from optyx._utils import explode_channel
from optyx.zw import Scalar as ScalarSingle
from optyx import channel
import numpy as np

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


class BitFlipError(channel.Channel):
    """
    Represents a bit-flip error channel.
    """

    def __init__(self, prob):
        from optyx import zx
        x_error = zx.X(1, 2) >> zx.Id(1) @ zx.ZBox(
            1, 1, np.sqrt((1 - prob) / prob)
        ) @ zx.Scalar(np.sqrt(prob * 2))
        super().__init__(
            name=f"BitFlipError({prob})",
            kraus=x_error,
            dom=channel.qubit,
            cod=channel.qubit,
            env=optyx.Bit(1),
        )

    def dagger(self):
        return self


class DephasingError(channel.Channel):
    """
    Represents a quantum dephasing error channel.
    """
    def __init__(self, prob):
        from optyx import zx
        z_error = (
            zx.H
            >> zx.X(1, 2)
            >> zx.H
            @ zx.ZBox(1, 1, np.sqrt((1 - prob) / prob))
            @ zx.Scalar(np.sqrt(prob * 2))
        )
        super().__init__(
            name=f"DephasingError({prob})",
            kraus=z_error,
            dom=channel.qubit,
            cod=channel.qubit,
            env=optyx.Bit(1),
        )

    def dagger(self):
        return self