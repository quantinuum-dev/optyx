
import numpy as np
from pyzx import Graph
from discopy import quantum as quantum_discopy
from pytket import circuit as tket_circuit
from discopy.cat import Category
from optyx._utils import explode_channel
from optyx.core import (
    channel,
    diagram,
    zx
)
from optyx import photonic
from optyx._utils import explode_channel


class Circuit(channel.Diagram):
    """
    A circuit that operates on qubits.
    It can be initialised from a ZX diagram, PyZX diagram,
    a tket circuit, or a discopy circuit. This is black box circuit
    until evaluated.
    """

    def __init__(self, circuit):
        self._underlying_circuit = circuit

    def double(self):
        """
        Convert the circuit to a double circuit.
        """
        return self._to_optyx().double()

    @property
    def is_pure(self):
        """
        Check if the circuit is pure.
        """
        return self._to_optyx().is_pure

    @property
    def get_kraus(self):
        """
        Get the kraus operators of the circuit.
        """
        return self._to_optyx().get_kraus()

    def _to_optyx(self):
        """
        Convert the circuit to an optyx channel diagram.
        """
        if isinstance(self._underlying_circuit, quantum_discopy.circuit.Circuit):
            return self._to_optyx_from_discopy()
        if isinstance(self._underlying_circuit, Graph):
            return self._to_optyx_from_pyzx()
        if isinstance(self._underlying_circuit, tket_circuit.Circuit):
            return self._to_optyx_from_tket()
        if isinstance(self._underlying_circuit, channel.Diagram):
            return self._to_optyx_from_zx()

    def _to_optyx_from_tket(self):
        """
        Convert a tket circuit to an optyx channel diagram.
        """
        self._to_optyx_from_discopy(
            quantum_discopy.circuit.Circuit.from_tk(
                self._underlying_circuit
            )
        )

    def _to_optyx_from_pyzx(self):
        """
        Convert a PyZX circuit to an optyx channel diagram.
        """
        zx_diagram = diagram.Diagram.from_pyzx(self._underlying_circuit)
        return explode_channel(
            zx_diagram,
            QubitChannel,
            channel.Diagram
        )

    def _to_optyx_from_discopy(self):
        """
        Convert a discopy circuit to an optyx channel diagram.
        """
        return quantum_discopy.circuit.Functor(
            ob={
                quantum_discopy.circuit.qubit: channel.qubit
            },
            ar=QubitChannel.from_discopy,
            dom=Category(
                quantum_discopy.circuit.Ty,
                quantum_discopy.circuit.Circuit
            ),
            cod=Category(
                channel.Ty,
                channel.Diagram
            ),
        )(self._underlying_circuit)

    def _to_optyx_from_zx(self):
        return self._underlying_circuit


class QubitChannel(channel.Channel):
    """Qubit channel."""

    def _decomp(self):
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

    @classmethod
    def from_discopy(cls, box):
        """Turns gates into ZX diagrams."""
        #pylint: disable=import-outside-toplevel
        from discopy.quantum.gates import Bra, Ket, Rz, Rx, CX, CZ, Controlled
        from discopy.quantum.gates import Scalar as GatesScalar
        ##### need to add mixed gates
        Id = channel.Diagram.id
        bit = channel.bit
        root2 = photonic.Scalar(2**0.5)
        if isinstance(box, (Bra, Ket)):
            dom, cod = (1, 0) if isinstance(box, Bra) else (0, 1)
            spiders = [X(dom, cod, phase=0.5 * bit) for bit in box.bitstring]
            return Id(bit**0).tensor(*spiders) @ photonic.Scalar(
                pow(2, -len(box.bitstring) / 2)
            )
        if isinstance(box, (Rz, Rx)):
            return (Z if isinstance(box, Rz) else X)(1, 1, box.phase)
        if isinstance(box, Controlled) and box.name.startswith("CRz"):
            return (
                Z(1, 2) @ Z(1, 2, box.phase / 2)
                >> Id(1) @ (X(2, 1) >> Z(1, 0, -box.phase / 2)) @ Id(1) @ root2
            )
        if isinstance(box, Controlled) and box.name.startswith("CRx"):
            return (
                X(1, 2) @ X(1, 2, box.phase / 2)
                >> Id(1) @ (Z(2, 1) >> X(1, 0, -box.phase / 2)) @ Id(1) @ root2
            )
        if isinstance(box, quantum_discopy.CU1):
            return Z(1, 2, box.phase) @ Z(1, 2, box.phase) >> Id(1) @ (
                X(2, 1) >> Z(1, 0, -box.phase)
            ) @ Id(1)
        if isinstance(box, GatesScalar):
            if box.is_mixed:
                raise NotImplementedError
            return photonic.Scalar(box.data)
        if isinstance(box, Controlled) and box.distance != 1:
            return Circuit._from_discopy(box._decompose())
        standard_gates = {
            quantum_discopy.H: H,
            quantum_discopy.Z: Z(1, 1, 0.5),
            quantum_discopy.X: X(1, 1, 0.5),
            quantum_discopy.Y: Z(1, 1, 0.5) >> X(1, 1, 0.5) @ photonic.Scalar(1j),
            quantum_discopy.S: Z(1, 1, 0.25),
            quantum_discopy.T: Z(1, 1, 0.125),
            CZ: Z(1, 2) @ Id(1) >> Id(1) @ H @ Id(1) >> Id(1) @ Z(2, 1) @ root2,
            CX: Z(1, 2) @ Id(1) >> Id(1) @ X(2, 1) @ root2,
        }
        return standard_gates[box]


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