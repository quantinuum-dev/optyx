"""Tools for compiling an MBQC pattern to a single emitter device"""

from dataclasses import dataclass
from optyx.compiler import Measurement


@dataclass
class FusionNetworkSE:
    """A fusion network for a single emitter linear resource state

    Specifies the measurements and fusions to perform on the nodes.
    Nodes are indexed by their position in the single linear resource state
    i.e. the first node is 0, second is 1 etc."""

    # Number of nodes in the path. A node may be implemented by many photons
    path: list[int]

    # The measurements applied to each of the nodes indexed by the Node ID
    measurements: list[Measurement]

    # Tuples containing the IDs of nodes connected by a fusion
    fusions: list[tuple[int, int]]


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
