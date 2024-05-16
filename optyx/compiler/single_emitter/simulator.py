""" Single Emitter Quantum Dot

Implements a single emitter FBQC quantum computer for the purpose of validating
instructions compiled for this machine
"""

from dataclasses import dataclass

from optyx.compiler.single_emitter import FusionNetworkSE
from optyx.compiler.mbqc import (
    Measurement,
    FusionOp,
    MeasureOp,
    NextNodeOp,
    PSMInstruction,
)


class ValidationError(Exception):
    """Thrown when machine believes the instructions is invalid"""


@dataclass
class FusionPatternSE:
    """A fusion network for a single emitter resource state together with a
    measurement order"""

    # Number of nodes in a path. A node may be implemented by multiple photons
    # The first node has ID = 1, the second ID = 2 and so on.
    path: list[int]

    # Tuples (Node ID, Measurement) listed in the order the measurements are
    # performed.
    measurements: list[tuple[int, Measurement]]

    # Tuples containing the IDs of the fused nodes in the order they were fused
    fusions: list[tuple[int, int]]


class SingleEmitterMultiMeasure:
    """A single emitter machine with unlimited measurement devices

    This machine can measure as many photons in parallel as it wants.
    It can only fuse one pair of photons at a time (though measurements may
    happen concurrently).
    """

    # Time increases with every instruction performed and hence corresponds to
    # a new photon entering the machine
    time: int

    path: list[int]

    # The nodes that are measured at a given timestamp.
    # Key: the time of measurement
    # Value: a list of Node IDs and their measurements
    measurements: dict[int, list[tuple[int, Measurement]]]

    # Records the photons in the delay loop which will later be fused
    # Key: time the photon will enter the device again
    # Value: the node the photon belongs to
    delayed_fusions: dict[int, int]

    # Fusions between two Node IDs ordered chronologically
    fusions: list[tuple[int, int]]

    def __init__(self):
        """Initialises the machine and sets the id of the current node being
        operated on"""
        self.time = 0
        self.path = []
        self.measurements = {}
        self.delayed_fusions = {}
        self.fusions = []

    def measure(self, m: Measurement):
        """Applies a unitary to the incoming photon and measures it"""
        self.time += 1

        if self.__node_has_been_measured(self.path[-1]):
            raise ValidationError(f"already measured node {self.path[-1]}")

        self.__record_measurement(self.time, self.path[-1], m)

    def __record_measurement(self, time: int, node: int, m: Measurement):
        measurements = self.measurements.get(time, [])
        measurements.append((node, m))
        self.measurements[time] = measurements

    def __node_has_been_measured(self, node: int):
        for t in reversed(range(1, self.time + 1)):
            if t not in self.measurements:
                continue

            for measurement in self.measurements[t]:
                if measurement[0] == node:
                    return True
        return False

    def fuse(self):
        """Fuses the incoming node with an incoming delayed photon"""
        self.time += 1

        if self.time not in self.delayed_fusions:
            raise ValidationError(
                f"no photon exiting delay loop at time {self.time}"
            )

        # The node the delayed photon belongs to
        fusion_node = self.delayed_fusions[self.time]

        self.fusions.append((self.path[-1], fusion_node))

    def delay_then_fuse(self, delay: int):
        """Applies a unitary to the incoming photon and measures it with a
        particular angle"""
        self.time += 1

        delay_until = self.time + delay

        if delay_until in self.delayed_fusions:
            raise ValidationError(
                f"can't fuse node {self.path[-1]} at time {delay_until} as"
                + f" node {self.delayed_fusions[delay_until]} will be fused"
            )

        self.delayed_fusions[delay_until] = self.path[-1]

    def delay_then_measure(self, delay: int, m: Measurement):
        """Delays the photon then measures it"""
        self.time += 1

        delay_until = self.time + delay
        self.__record_measurement(delay_until, self.path[-1], m)

    def next_node(self, node_id: int):
        """Progresses to the next node.

        Sets the id of the next node. This is just used for bookkeeping and
        helps us convert the instructions back into a fusion network."""

        self.path.append(node_id)

    def fusion_pattern(self) -> FusionPatternSE:
        """Outputs the MBQC pattern implemented by the operations"""

        measurements: list[tuple[int, Measurement]] = []
        chronological_order = sorted(self.measurements.keys())

        for t in chronological_order:
            for m in self.measurements[t]:
                measurements.append(m)

        return FusionPatternSE(self.path, measurements, self.fusions)


def fusion_pattern_to_network(fp: FusionPatternSE) -> FusionNetworkSE:
    """Converts a fusion pattern into a fusion network"""
    measurements = {meas[0]: meas[1] for meas in fp.measurements}

    return FusionNetworkSE(fp.path, measurements, fp.fusions)


def decompile_to_fusion_pattern(
    instructions: list[PSMInstruction],
) -> FusionPatternSE:
    """Converts the instructions back into a fusion network"""
    machine = SingleEmitterMultiMeasure()

    for ins in instructions:
        if isinstance(ins, FusionOp):
            if ins.delay == 0:
                machine.fuse()
            else:
                machine.delay_then_fuse(ins.delay)
        elif isinstance(ins, MeasureOp):
            if ins.delay == 0:
                machine.measure(ins.measurement)
            else:
                machine.delay_then_measure(ins.delay, ins.measurement)
        elif isinstance(ins, NextNodeOp):
            machine.next_node(ins.node_id)

    return machine.fusion_pattern()
