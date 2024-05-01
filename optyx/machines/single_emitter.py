""" Single Emitter Quantum Dot

Implements a single emitter FBQC quantum computer for the purpose of validating
instructions compiled for this machine
"""

from dataclasses import dataclass


@dataclass
class Measurement:
    """A symbolic MBQC measurement. We only verify that nodes are measured
    with the intended measurement. So we just use an ID to identify them.
    """

    id: int


class ValidationError(Exception):
    """Thrown when machine believes the instructions is invalid"""


@dataclass
class SingleFusionPattern:
    """A fusion network, together with a measurement order"""

    # Number of nodes in a path. A node may be implemented by multiple photons
    # The first node has ID = 1, the second ID = 2 and so on.
    path_length: int

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

    current_node: int

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
        self.time = 0
        self.current_node = 1
        self.measurements = {}
        self.delayed_fusions = {}
        self.fusions = []

    def measure(self, m: Measurement):
        """Applies a unitary to the incoming photon and measures it"""
        self.time += 1

        if self.__node_has_been_measured(self.current_node):
            raise ValidationError(f"already measured node {self.current_node}")

        self.__record_measurement(self.time, self.current_node, m)

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

        self.fusions.append((self.current_node, fusion_node))

    def delay_then_fuse(self, delay: int):
        """Applies a unitary to the incoming photon and measures it with a
        particular angle"""
        self.time += 1

        delay_until = self.time + delay

        if delay_until in self.delayed_fusions:
            raise ValidationError(
                f"can't fuse node {self.current_node} at time {delay_until} as"
                + f" node {self.delayed_fusions[delay_until]} will be fused"
            )

        self.delayed_fusions[delay_until] = self.current_node

    def delay_then_measure(self, delay: int, m: Measurement):
        """Delays the photon then measures it"""
        self.time += 1

        delay_until = self.time + delay
        self.__record_measurement(delay_until, self.current_node, m)

    def next_node(self):
        """Progresses to the next node"""
        self.current_node += 1

    def fusion_pattern(self) -> SingleFusionPattern:
        """Outputs the MBQC pattern implemented by the operations"""

        path_length = self.current_node

        measurements: list[tuple[int, Measurement]] = []
        chronological_order = sorted(self.measurements.keys())

        for t in chronological_order:
            for m in self.measurements[t]:
                measurements.append(m)

        return SingleFusionPattern(path_length, measurements, self.fusions)
