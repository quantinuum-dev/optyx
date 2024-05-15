"""Contains the fundamental classes requirement to define MBQC patterns"""

from typing import Callable
from copy import deepcopy
from dataclasses import dataclass
import networkx as nx


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

    inside: nx.Graph

    # The measurement associated with each node in the graph
    measurements: list[Measurement]

    inputs: set[int]
    outputs: set[int]

    def __eq__(self, other):
        """Checks the two open graphs are equal

        This doesn't check they are equal up to an isomorphism"""
        return (
            self.inputs == other.inputs
            and self.outputs == other.outputs
            and nx.utils.graphs_equal(self.inside, other.inside)
            and self.measurements == other.measurements
        )

    def __deepcopy__(self, memo):
        return OpenGraph(
            inside=deepcopy(self.inside, memo),
            measurements=deepcopy(self.measurements, memo),
            inputs=deepcopy(self.inputs, memo),
            outputs=deepcopy(self.outputs, memo),
        )

    def perform_z_deletions_in_place(self):
        """Removes the Z-deleted nodes from the graph in place"""
        zero_nodes = [
            i for i, m in enumerate(self.measurements) if m.is_zero()
        ]

        for node in zero_nodes:
            self.inside.remove_node(node)

        # Remove the deleted node's measurements
        self.measurements = [m for m in self.measurements if not m.is_zero()]

    def perform_z_deletions(self):
        """Removes the Z-deleted nodes from the graph"""
        g = deepcopy(self)
        g.perform_z_deletions_in_place()
        return g


# Given a node, returns all the nodes in it's past
PartialOrder = Callable[[int], list[int]]
