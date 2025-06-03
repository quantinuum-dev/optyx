
import numpy as np

from optyx.core import (
    channel,
    diagram,
    zx,
)
from optyx._utils import explode_channel

class MeasureQubits(channel.Measure):
    """
    Ideal qubit measurement (in computational basis) from qubit to bit.
    """

    def __init__(self, n):
        super().__init__(
            channel.qubit**n
        )


class DiscardQubits(channel.Discard):
    """
    Discard :math:`n` qubits.
    """

    def __init__(self, n):
        super().__init__(
            channel.qubit**n
        )


class EncodeBits(channel.Encode):
    """
    Encode :math:`n` bits into :math:`n` qubits.
    """

    def __init__(self, n):
        super().__init__(
            channel.bit**n
        )


class QubitChannel(channel.Channel):
    """Qubit channel."""

    def decomp(self):
        decomposed = zx.decomp(self.kraus)
        return explode_channel(
            decomposed,
            QubitChannel,
            channel.Diagram
        )

    def to_dual_rail(self):
        """Convert to dual-rail encoding."""
        kraus_path = zx.zx2path(self.kraus)
        return explode_channel(
            kraus_path,
            channel.Channel,
            channel.Diagram
        )


class Z(QubitChannel):
    """Z spider."""

    tikzstyle_name = "Z"
    color = "green"
    draw_as_spider = True

    def __init__(self, n_legs_in, n_legs_out, phase=0):
        kraus = zx.Z(n_legs_in, n_legs_out, phase)
        super().__init__(
            f"Z({phase})",
            kraus,
            channel.qubit**n_legs_in,
            channel.qubit**n_legs_out,
        )


class X(QubitChannel):
    """X spider."""

    tikzstyle_name = "X"
    color = "red"
    draw_as_spider = True

    def __init__(self, n_legs_in, n_legs_out, phase=0):
        kraus = zx.Z(n_legs_in, n_legs_out, phase)
        super().__init__(
            f"X({phase})",
            kraus,
            channel.qubit**n_legs_in,
            channel.qubit**n_legs_out,
        )


class H(QubitChannel):
    """Hadamard gate."""

    tikzstyle_name = "H"
    color = "yellow"

    def __init__(self):
        super().__init__(
            "H",
            zx.H(),
            channel.qubit,
            channel.qubit,
        )


class Scalar(QubitChannel):
    def __init__(self, value: float):
        super().__init__(
            f"Scalar({value})",
            zx.Scalar(value),
            channel.qubit**0,
            channel.qubit**0,
        )


class BitFlipError(channel.Channel):
    """
    Represents a bit-flip error channel.
    """

    def __init__(self, prob):
        from optyx.core import zx
        x_error = zx.X(1, 2) >> zx.Id(1) @ zx.ZBox(
            1, 1, np.sqrt((1 - prob) / prob)
        ) @ zx.Scalar(np.sqrt(prob * 2))
        super().__init__(
            name=f"BitFlipError({prob})",
            kraus=x_error,
            dom=channel.qubit,
            cod=channel.qubit,
            env=diagram.Bit(1),
        )

    def dagger(self):
        return self


class DephasingError(channel.Channel):
    """
    Represents a quantum dephasing error channel.
    """
    def __init__(self, prob):
        from optyx.core import zx
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
            env=diagram.Bit(1),
        )

    def dagger(self):
        return self


class Circuit(channel.Circuit):
    """
    A circuit that operates on qubits.
    It can be initialised from ZX diagram, PyZX diagram,
    a tket circuit, or a discopy circuit. This is black box circuit
    until evaluated.
    """

    def __init__(self, circuit):
        self.circuit = circuit

    def to_optyx(self):
        """
        Convert the circuit to an optyx channel diagram.
        """

        #check if the circuit is the type (PyZX, tket, discopy)

    def _from_tket(self, tket_circuit):
        """
        Convert a tket circuit to an optyx channel diagram.
        """
        return self.from_tket(tket_circuit)

    def _from_pyzx(self, pyzx_circuit):
        """
        Convert a PyZX circuit to an optyx channel diagram.
        """

    def _from_discopy(self, discopy_circuit):
        """
        Convert a discopy circuit to an optyx channel diagram.
        """
