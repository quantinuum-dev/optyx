"""Decompiler for SEMM machine

Implements a decompiler routine that converts a list of instructions back into
a fusion network. This may be used as one way to check the correctness of
compilation pipelines.

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    decompile_to_fusion_network_multi
"""

from optyx.compiler.mbqc import (
    Measurement,
    ULFusionNetwork,
    FusionNetwork,
    Fusion,
)

from optyx.compiler.protocols import (
    FusionOp,
    MeasureOp,
    NextNodeOp,
    NextResourceStateOp,
    UnmeasuredOp,
    Instruction,
)


class ValidationError(Exception):
    """Thrown when machine believes the instructions is invalid"""


class SingleEmitterMultiMeasureMulti:
    """A single emitter machine with unlimited measurement devices

    This machine can measure as many photons in parallel as it wants.
    It can only fuse one pair of photons at a time (though measurements may
    happen concurrently).
    """

    # Time increases with every instruction performed and hence corresponds to
    # a new photon entering the machine
    time: int

    paths: list[list[int]]

    # The nodes that are measured at a given timestamp.
    # Key: the time of measurement
    # Value: a list of Node IDs and their measurements
    measurements: dict[int, list[tuple[int, Measurement]]]

    # Records the photons in the delay loop which will later be fused
    # Key: time the photon will enter the device again
    # Value: the node the photon belongs to
    delayed_fusions: dict[int, int]

    # Fusions between two Node IDs ordered chronologically
    fusions: list[Fusion]

    def __init__(self):
        """Initialises the machine and sets the id of the current node being
        operated on"""
        self.time = 0
        self.paths = []
        self.measurements = {}
        self.delayed_fusions = {}
        self.fusions = []

    def unmeasured_photon(self):
        """Signifies an unmeasured photon, i.e. an output node"""
        self.time += 1

    def next_resource_state(self):
        """Starts a new resource state"""
        self.paths.append([])

    def measure(self, m: Measurement):
        """Measures the incoming photon"""
        self.time += 1

        if self.__node_has_been_measured(self.paths[-1][-1]):
            raise ValidationError(
                f"already measured node {self.paths[-1][-1]}"
            )

        self.__record_measurement(self.time, self.paths[-1][-1], m)

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

    def fuse(self, fusion_type: str):
        """Fuses the incoming node with an incoming delayed photon"""
        self.time += 1

        if self.time not in self.delayed_fusions:
            raise ValidationError(
                f"no photon exiting delay loop at time {self.time}"
            )

        # The node the delayed photon belongs to
        fusion_node = self.delayed_fusions[self.time]

        self.fusions.append(
            Fusion(self.paths[-1][-1], fusion_node, fusion_type)
        )

    def delay_then_fuse(self, delay: int):
        """Applies a unitary to the incoming photon and measures it with a
        particular angle"""
        self.time += 1
        node = self.paths[-1][-1]

        delay_until = self.time + delay

        if delay_until in self.delayed_fusions:
            raise ValidationError(
                f"can't fuse node {node} at time {delay_until} as"
                + f" node {self.delayed_fusions[delay_until]} will be fused"
            )

        self.delayed_fusions[delay_until] = node

    def delay_then_measure(self, delay: int, m: Measurement):
        """Delays the photon then measures it"""
        self.time += 1

        delay_until = self.time + delay
        self.__record_measurement(delay_until, self.paths[-1][-1], m)

    def next_node(self, node_id: int):
        """Progresses to the next node.

        Sets the id of the next node. This is just used for bookkeeping and
        helps us convert the instructions back into a fusion network."""

        self.paths[-1].append(node_id)

    def fusion_network(self) -> FusionNetwork:
        """Outputs the Fusion pattern implemented by the operations"""

        measurements: list[tuple[int, Measurement]] = []
        chronological_order = sorted(self.measurements.keys())

        for t in chronological_order:
            for m in self.measurements[t]:
                measurements.append(m)

        measurement_dict = {meas[0]: meas[1] for meas in measurements}
        return FusionNetwork(self.paths, measurement_dict, self.fusions)


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
    fusions: list[Fusion]

    def __init__(self):
        """Initialises the machine and sets the id of the current node being
        operated on"""
        self.time = 0
        self.path = []
        self.measurements = {}
        self.delayed_fusions = {}
        self.fusions = []

    def measure(self, m: Measurement):
        """Measures the incoming photon"""
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

    def fuse(self, fusion_type: str):
        """Fuses the incoming node with an incoming delayed photon"""
        self.time += 1

        if self.time not in self.delayed_fusions:
            raise ValidationError(
                f"no photon exiting delay loop at time {self.time}"
            )

        # The node the delayed photon belongs to
        fusion_node = self.delayed_fusions[self.time]

        self.fusions.append(Fusion(self.path[-1], fusion_node, fusion_type))

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

    def fusion_network(self) -> ULFusionNetwork:
        """Outputs the Fusion pattern implemented by the operations"""

        measurements: list[tuple[int, Measurement]] = []
        chronological_order = sorted(self.measurements.keys())

        for t in chronological_order:
            for m in self.measurements[t]:
                measurements.append(m)

        measurement_dict = {meas[0]: meas[1] for meas in measurements}
        return ULFusionNetwork(self.path, measurement_dict, self.fusions)


def decompile_to_fusion_network_multi(
    instructions: list[Instruction],
) -> FusionNetwork:
    """Converts the instructions back into a fusion network

    Example
    -------
    >>> from optyx.compiler.mbqc import FusionNetwork, Measurement
    >>> from .semm import compile_linear_fn
    >>> from optyx.compiler.semm_decompiler import (
    ...     decompile_to_fusion_network_multi
    ... )
    >>> m = {i: Measurement(0.5 * i, "XY") for i in range(2)}
    >>> fn = FusionNetwork([[0, 1], [2]], m, [Fusion(0, 2, "X")])
    >>>
    >>> # We impose any partial order on the nodes for demonstrative purposes
    >>> def order(n: int) -> list[int]:
    ...     return list(range(n, 3))
    >>>
    >>> ins = compile_linear_fn(fn, order)
    >>> fn_decompiled = decompile_to_fusion_network_multi(ins)
    >>> assert fn == fn_decompiled
    """
    machine = SingleEmitterMultiMeasureMulti()

    for ins in instructions:
        if isinstance(ins, FusionOp):
            if ins.delay == 0:
                machine.fuse(ins.fusion_type)
            else:
                machine.delay_then_fuse(ins.delay)
        elif isinstance(ins, MeasureOp):
            if ins.delay == 0:
                machine.measure(ins.measurement)
            else:
                machine.delay_then_measure(ins.delay, ins.measurement)
        elif isinstance(ins, NextNodeOp):
            machine.next_node(ins.node_id)
        elif isinstance(ins, NextResourceStateOp):
            machine.next_resource_state()
        elif isinstance(ins, UnmeasuredOp):
            machine.unmeasured_photon()

    return machine.fusion_network()
