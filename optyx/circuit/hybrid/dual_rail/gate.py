"""
This module defines gates
for working with dual-rail quantum circuits.
"""

from optyx.diagram.optyx import dual_rail, Mode
from optyx.diagram.channel import Channel
from optyx.circuit.qmode.lo import HadamardBS
from optyx.diagram.lo import Phase

class DualRail(Channel):
    """
    Represents a dual-rail quantum channel
    encoding a specified number of qubit registers.
    """
    def __init__(self, n_qubits):
        super().__init__(
            f"DualRail({n_qubits})",
            dual_rail(n_qubits)
        )


Hadamard = HadamardBS


class PhaseShift(Channel):
    """
    Represents a phase shift operation in dual-rail encoding.
    """
    def __init__(self, phase):
        super().__init__(
            f"PhaseShift({phase})",
            Mode(1) @ Phase(phase)
        )


### other gates ????