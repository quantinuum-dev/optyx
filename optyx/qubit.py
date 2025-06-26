
import numpy as np
from typing import Literal
from pyzx.graph.base import BaseGraph
from discopy import quantum as quantum_discopy
from discopy import symmetric
from pytket import circuit as tket_circuit
from optyx._utils import explode_channel
from optyx.core import (
    channel,
    diagram,
    zx
)


class Circuit(channel.Diagram):
    """
    A circuit that operates on qubits.
    It can be initialised from a ZX diagram, PyZX diagram,
    a tket circuit, or a discopy circuit. This is black box circuit
    until evaluated.
    """

    def __init__(self, circuit):
        self._underlying_circuit = circuit
        self.name = "Circuit"
        self.type = self._detect_type()
        dom, cod = self._get_dom_cod()
        inside = channel.Channel(
            "Circuit",
            diagram.Box(
                name="Circuit",
                dom=dom.single(),
                cod=cod.single()
            )
        )
        super().__init__(
            dom=dom,
            cod=cod,
            inside=inside.inside
        )

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

    def get_kraus(self):
        """
        Get the kraus operators of the circuit.
        """
        return self._to_optyx().get_kraus()

    def _get_dom_cod(self):
        if self.type == "tket":
            self._underlying_circuit = quantum_discopy.circuit.Circuit.from_tk(
                self._underlying_circuit
            )
            self.type = "discopy"
        if self.type == "discopy":
            return (
                channel.qubit**len(self._underlying_circuit.dom),
                channel.qubit**len(self._underlying_circuit.cod)
            )
        if self.type == "pyzx":
            return (
                channel.qubit**len(self._underlying_circuit.inputs()),
                channel.qubit**len(self._underlying_circuit.outputs())
            )
        if self.type == "zx":
            return (
                self._underlying_circuit.dom,
                self._underlying_circuit.cod
            )
        raise TypeError("Unsupported circuit type")  # pragma: no cover

    def _detect_type(self):
        """
        Detect the type of the underlying circuit.
        """
        if isinstance(self._underlying_circuit,
                      quantum_discopy.circuit.Circuit):
            return "discopy"
        if isinstance(self._underlying_circuit, BaseGraph):
            return "pyzx"
        if isinstance(self._underlying_circuit, tket_circuit.Circuit):
            return "tket"
        if isinstance(self._underlying_circuit, channel.Diagram):
            return "zx"
        raise TypeError("Unsupported circuit type")  # pragma: no cover

    def _to_optyx(self):
        """
        Convert the circuit to an optyx channel diagram.
        """
        if self.type == "discopy":
            return self._to_optyx_from_discopy()
        if self.type == "pyzx":
            return self._to_optyx_from_pyzx()
        if self.type == "tket":
            return self._to_optyx_from_tket()
        if self.type == "zx":
            return self._to_optyx_from_zx()

    def _to_optyx_from_tket(self):
        """
        Convert a tket circuit to an optyx channel diagram.
        """
        self._underlying_circuit = quantum_discopy.circuit.Circuit.from_tk(
            self._underlying_circuit
        )
        return self._to_optyx_from_discopy()

    def _to_optyx_from_pyzx(self):
        """
        Convert a PyZX circuit to an optyx channel diagram.
        """
        zx_diagram = zx.ZXDiagram.from_pyzx(self._underlying_circuit)
        return explode_channel(
            zx_diagram,
            channel.Channel,
            channel.Diagram
        )

    def to_dual_rail(self):
        """Convert to dual-rail encoding."""
        return self._to_optyx().to_dual_rail()

    def _to_optyx_from_discopy(self):
        """
        Convert a discopy circuit to an optyx channel diagram.
        """

        def ob(o):
            if o.name == "qubit":
                return channel.qubit**len(o)
            if o.name == "bit":
                return channel.bit**len(o)

        return symmetric.Functor(
            ob=ob,
            ar=QubitChannel.from_discopy,
            dom=symmetric.Category(
                quantum_discopy.circuit.Ty,
                quantum_discopy.circuit.Circuit
            ),
            cod=symmetric.Category(
                channel.Ty,
                channel.Diagram
            ),
        )(self._underlying_circuit)

    def _to_optyx_from_zx(self):
        return self._underlying_circuit


class QubitChannel(channel.Channel, Circuit):
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
        # pylint: disable=import-outside-toplevel
        from discopy.quantum.gates import (
            Bra, Ket, Rz, Rx,
            CX, CZ, Controlled, Digits
        )
        from discopy.quantum.gates import Scalar as GatesScalar
        from optyx import classical

        def get_perm(n):
            return sorted(sorted(list(range(n))), key=lambda i: i % 2)

        Id = channel.Diagram.id
        root2 = Scalar(2**0.5)
        if isinstance(box, (Bra, Ket)):
            dom, cod = (1, 0) if isinstance(box, Bra) else (0, 1)
            spiders = [X(dom, cod, phase=0.5 * bit) for bit in box.bitstring]
            return Id(channel.qubit**0).tensor(*spiders) @ Scalar(
                pow(2, -len(box.bitstring) / 2)
            )
        if isinstance(box, (Rz, Rx)):
            return (Z if isinstance(box, Rz) else X)(1, 1, box.phase)
        if isinstance(box, Controlled) and box.name.startswith("CRz"):
            return (
                Z(1, 2) @ Z(1, 2, box.phase / 2)
                >> Id(channel.qubit) @
                (X(2, 1) >> Z(1, 0, -box.phase / 2)) @
                Id(channel.qubit) @ root2
            )
        if isinstance(box, Controlled) and box.name.startswith("CRx"):
            return (
                X(1, 2) @ X(1, 2, box.phase / 2)
                >> Id(channel.qubit) @
                (Z(2, 1) >> X(1, 0, -box.phase / 2)) @
                Id(channel.qubit) @ root2
            )
        if isinstance(box, Digits):
            dgrm = Id(channel.bit**0)
            for d in box.digits:
                if d > 1:
                    raise ValueError(
                        "Only qubits supported. Digits must be 0 or 1."
                    )
                dgrm @= classical.X(0, 1, 0.5**d) @ classical.Scalar(0.5**0.5)
            return dgrm
        if isinstance(box, quantum_discopy.CU1):
            return (
                Z(1, 2, box.phase) @ Z(1, 2, box.phase) >>
                Id(channel.qubit) @
                (X(2, 1) >> Z(1, 0, -box.phase)) @
                Id(channel.qubit)
            )
        if isinstance(box, GatesScalar):
            return Scalar(box.data)
        if isinstance(box, Controlled) and box.distance != 1:
            return Circuit(box._decompose())._to_optyx()
        if isinstance(box, quantum_discopy.Discard):
            return DiscardQubits(len(box.dom))
        if isinstance(box, quantum_discopy.Measure):
            no_qubits = sum([1 if i.name == "qubit" else 0 for i in box.dom])
            dgrm = MeasureQubits(no_qubits)
            if box.override_bits:
                dgrm @= channel.Discard(channel.bit**no_qubits)
            if box.destructive:
                return dgrm
            else:
                dgrm >>= classical.CopyBit(2)**no_qubits
                dgrm >>= channel.Diagram.permutation(
                    get_perm(2 * no_qubits), channel.bit**(2 * no_qubits)
                )
                dgrm >>= EncodeBits(no_qubits) @ Id(channel.bit**no_qubits)
                return dgrm
        if isinstance(box, quantum_discopy.Encode):
            raise NotImplementedError(
                "Converting Encode to QubitChannel is not implemented."
            )
        standard_gates = {
            quantum_discopy.H: H(),
            quantum_discopy.Z: Z(1, 1, 0.5),
            quantum_discopy.X: X(1, 1, 0.5),
            quantum_discopy.Y: Z(1, 1, 0.5) >> X(1, 1, 0.5) @ Scalar(1j),
            quantum_discopy.S: Z(1, 1, 0.25),
            quantum_discopy.T: Z(1, 1, 0.125),
            CZ: (
                Z(1, 2) @ Id(channel.qubit) >>
                Id(channel.qubit) @ H() @ Id(channel.qubit) >>
                Id(channel.qubit) @ Z(2, 1) @ root2
                ),
            CX: (
                Z(1, 2) @ Id(channel.qubit) >>
                Id(channel.qubit) @ X(2, 1) @ root2
                ),
        }
        return standard_gates[box]


class MeasureQubits(channel.Measure):
    """
    Ideal qubit measurement (in computational basis)
    from qubit to bit.
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


class Z(channel.Channel):
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


class X(channel.Channel):
    """X spider."""

    tikzstyle_name = "X"
    color = "red"
    draw_as_spider = True

    def __init__(self, n_legs_in, n_legs_out, phase=0):
        kraus = zx.X(n_legs_in, n_legs_out, phase)
        super().__init__(
            f"X({phase})",
            kraus,
            channel.qubit**n_legs_in,
            channel.qubit**n_legs_out,
        )


class H(channel.Channel):
    """Hadamard gate."""

    tikzstyle_name = "H"
    color = "yellow"

    def __init__(self):
        super().__init__(
            "H",
            zx.H,
            channel.qubit,
            channel.qubit,
        )


class Scalar(channel.Channel):
    def __init__(self, value: float):
        super().__init__(
            f"Scalar({value})",
            zx.scalar(value),
            channel.qubit**0,
            channel.qubit**0,
        )


class BitFlipError(channel.Channel):
    """
    Represents a bit-flip error channel.
    """

    def __init__(self, prob):
        # pylint: disable=import-outside-toplevel
        from optyx.core import zx
        x_error = zx.X(1, 2) >> zx.Id(1) @ zx.ZBox(
            1, 1, np.sqrt((1 - prob) / prob)
        ) @ zx.scalar(np.sqrt(prob * 2))
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
        # pylint: disable=import-outside-toplevel
        from optyx.core import zx
        z_error = (
            zx.H
            >> zx.X(1, 2)
            >> zx.H
            @ zx.ZBox(1, 1, np.sqrt((1 - prob) / prob))
            @ zx.scalar(np.sqrt(prob * 2))
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


class Ket(channel.Channel):
    """Computational basis state for qubits"""

    def __init__(
        self, value: Literal[0, 1, "+", "-"], cod: channel.Ty = channel.qubit
    ) -> None:
        spider = zx.X if value in (0, 1) else zx.Z
        phase = 0 if value in (0, "+") else 0.5
        kraus = spider(0, 1, phase) @ diagram.Scalar(1 / np.sqrt(2))
        super().__init__(f"|{value}>", kraus, cod=cod)


class Bra(channel.Channel):
    """Post-selected measurement for qubits"""

    def __init__(
        self, value: Literal[0, 1, "+", "-"], dom: channel.Ty = channel.qubit
    ) -> None:
        spider = zx.X if value in (0, 1) else zx.Z
        phase = 0 if value in (0, "+") else 0.5
        kraus = spider(1, 0, phase) @ diagram.Scalar(1 / np.sqrt(2))
        super().__init__(f"<{value}|", kraus, dom=dom)


def Id(n):
    return channel.Diagram.id(n) if \
          isinstance(n, channel.Ty) else channel.Diagram.id(channel.qubit**n)
