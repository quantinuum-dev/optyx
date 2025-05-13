from optyx.diagram.optyx import dual_rail, Mode
from optyx.diagram.channel import Channel
from optyx.circuit.qmode.lo import HadamardBS
from optyx.diagram.lo import Phase

class DualRail(Channel):
    def __init__(self, n_qubits):
        super().__init__(
            f"DualRail({n_qubits})",
            dual_rail(n_qubits)
        )


Hadamard = HadamardBS

class PhaseShift(Channel):
    def __init__(self, phase):
        super().__init__(
            f"PhaseShift({phase})",
            Mode(1) @ Phase(phase)
        )
