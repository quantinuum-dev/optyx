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
    fusion_type: str


@dataclass
class NextNodeOp:
    """Tells the machine to progress to the next node and sets the node id

    Concretely, it will tell it to produce a Hadamard edge between the
    previously emitted photon and the next one"""

    node_id: int


@dataclass
class NextResourceStateOp:
    """Tells the machine to start emitting another resource state.

    Concretely, it will tell it to produce a photon that is not entangled with
    the previous photon"""


@dataclass
class UnmeasuredPhotonOp:
    """There is an unmeasured photon i.e. an output photon"""


# Photon Stream Machine Instructions
Instruction = (
    MeasureOp
    | FusionOp
    | NextNodeOp
    | NextResourceStateOp
    | UnmeasuredPhotonOp
)
