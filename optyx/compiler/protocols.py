"""Contains the basic pieces for a standardised language for MBQC.

The standard is still in development, but it already looks to be similar to the
measurement calculus"""

from dataclasses import dataclass
from optyx.compiler.mbqc import Measurement


@dataclass
class MeasureOp:
    """Measurement Operation"""

    delay: int
    measurement: Measurement


@dataclass
class FusionOp:
    """Fusion Operation"""

    delay: int


@dataclass
class NextNodeOp:
    """Tells the machine to progress to the next node and sets the node id

    Concretely, it will tell it to produce a Hadamard edge between the
    previously emitted photon and the next one"""

    node_id: int


# Photon Stream Machine Instructions
PSMInstruction = MeasureOp | FusionOp | NextNodeOp
