"""
Overview
--------

This module defines classes and functions for working with quantum circuits and
quantum channels using ZX-calculus and related tools. It provides functionality
to decompose circuits, convert between different representations (e.g., tket,
dual-rail encoding), and manipulate ZX-diagrams.

Classes:
- QubitCircuit: Represents a quantum circuit with methods for decomposition,
    conversion to dual-rail encoding, and conversion to/from tket circuits.
- QubitChannel: Represents a quantum channel, inheriting from QubitCircuit and
    Channel, with additional methods for decomposition and dual-rail conversion.
- Z: Represents a Z spider in ZX-calculus with a specified number of input/output
    legs and an optional phase.
- X: Represents an X spider in ZX-calculus with a specified number of input/output
    legs and an optional phase.
- H: Represents a Hadamard gate in ZX-calculus.

Usage:
------
This module is intended for use in quantum circuit design, analysis, and
optimization using ZX-calculus and related frameworks. It allows for seamless
conversion between different representations and provides tools for manipulating
quantum circuits and channels at a high level of abstraction.
"""

from optyx.diagram.zx import (
    X as XSingle,
    Z as ZSingle,
    H as HSingle,
    decomp,
    zx2path
)
from optyx.diagram.channel import (
    Channel,
    Circuit,
    qubit,
    qmode,
    Ty
)
from optyx.utils import explode_channel
from discopy import symmetric
from discopy.cat import factory
from pytket.extensions.pyzx import tk_to_pyzx, pyzx_to_tk
from optyx.diagram.optyx import Bit, Diagram
from pyzx import extract_circuit

@factory
class QubitCircuit(Circuit):
    """Qubit circuit."""

    ty_factory = Ty

    def decomp(self):

        assert self.is_pure, "Circuit must be pure to convert to tket."

        return symmetric.Functor(
            ob=lambda x: qubit**len(x),
            ar=lambda arr: arr.decomp(),
            cod=symmetric.Category(Ty, QubitCircuit),
        )(self)

    def to_dual_rail(self):
        """Convert to dual-rail encoding."""

        assert self.is_pure, "Circuit must be pure to convert to tket."

        return symmetric.Functor(
            ob=lambda x: qmode**(2*len(x)),
            ar=lambda arr: arr.to_dual_rail(),
            cod=symmetric.Category(Ty, Circuit),
        )(self.decomp())

    @classmethod
    def from_tket(self, tket_circuit):
        """Convert from tket circuit."""
        pyzx_circuit = tk_to_pyzx(tket_circuit).to_graph()
        zx_diagram = Diagram.from_pyzx(pyzx_circuit)
        return explode_channel(
            zx_diagram,
            QubitChannel,
            QubitCircuit
        )


    def to_tket(self):
        """
        Convert to tket circuit. The circuit must be a pure circuit.
        """

        assert self.is_pure, "Circuit must be pure to convert to tket."

        kraus_maps = []
        for layer in self:
            left = layer.inside[0][0]
            right = layer.inside[0][2]
            generator = layer.inside[0][1]

            kraus_maps.append(
                Bit(len(left)) @ generator.kraus @ Bit(len(right))
            )

        return pyzx_to_tk(
            extract_circuit(
                Diagram.then(
                    *kraus_maps
                ).to_pyzx()
            ).to_basic_gates()
        )


class QubitChannel(QubitCircuit, Channel):
    """Qubit channel."""

    def decomp(self):
        decomposed = decomp(self.kraus)
        return explode_channel(
            decomposed,
            QubitChannel,
            QubitCircuit
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

    def __init__(self, n_legs_in, n_legs_out, phase=0):
        kraus = ZSingle(n_legs_in, n_legs_out, phase)
        super().__init__(
            f"Z({n_legs_in}, {n_legs_out}, {phase})",
            kraus,
            qubit**n_legs_in,
            qubit**n_legs_out,
        )


class X(QubitChannel):
    """X spider."""

    tikzstyle_name = "X"
    color = "red"

    def __init__(self, n_legs_in, n_legs_out, phase=0):
        kraus = XSingle(n_legs_in, n_legs_out, phase)
        super().__init__(
            f"X({n_legs_in}, {n_legs_out}, {phase})",
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
