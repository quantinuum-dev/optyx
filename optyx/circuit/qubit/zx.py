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
    qubit
)
from optyx.utils import explode_channel
from pytket.extensions.pyzx import tk_to_pyzx
from discopy import symmetric
from pytket.extensions.pyzx import pyzx_to_tk
from optyx.diagram.optyx import Bit, Diagram
from pyzx import extract_circuit
from optyx.utils import explode_channel
from optyx.diagram import optyx


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
