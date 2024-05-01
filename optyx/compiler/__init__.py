"""Toolkit for compiling open graphs in machine instructions"""

from typing import Callable
from dataclasses import dataclass
from optyx.graphs import Graph


@dataclass
class Measurement:
    """A symbolic MBQC measurement. We only verify that nodes are measured
    with the intended measurement. So we just use an ID to identify them.
    """

    id: int

    def is_zero(self) -> bool:
        """Indicates whether it is the zero measurement"""
        return self.id == -1


def zero_measurement() -> Measurement:
    """Returns the zero measurement"""
    return Measurement(-1)


@dataclass
class OpenGraph:
    """Open graph contains the graph, measurement, and input and output
    nodes. This is the graph we wish to implement deterministically"""

    g: Graph

    # The measurement associated with each node in the graph
    m: list[Measurement]

    inputs: list[int]
    outputs: list[int]

    def perform_z_deletions(self):
        """Removes the Z-deleted nodes from the graph"""
        zero_nodes = [i for i, m in enumerate(self.m) if m.is_zero()]

        for node in zero_nodes:
            self.g.remove_node(node)

        # Remove the deleted node's measurements
        self.m = [m for m in self.m if not m.is_zero()]


# Given a node, returns all the nodes in it's past
PartialOrder = Callable[[int], list[int]]
