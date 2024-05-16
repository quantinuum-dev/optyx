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


def add_fusion_order_to_partial_order(
    fusions: list[tuple[int, int]], order: PartialOrder
) -> PartialOrder:
    """Returns a new partial order that ensures the fusion order is respected.

    We achieve this the simplest way possible, namely, we want the following
    condition to be satisfied in the new order: any node that has w in its
    past, should now contain v and its past.
    """

    def fusion_with_node_order(node: int) -> list[int]:
        past = order(node)

        # We temporarily remove the node itself since we don't want to add the
        # past of the nodes it itself is fused to. We will add it back at the
        # end
        past.remove(node)

        past_set: set[int] = set()
        for el in past:
            fused_nbrs = _get_all_connected_fusions(fusions, el)
            for nbr in fused_nbrs:
                past_set = past_set.union(set(fusion_with_node_order(nbr)))
            past_set = past_set.union(set(fused_nbrs))

        past_set.add(node)
        return list(past_set)

    return fusion_with_node_order


# Find every node that is connected to the given node through (possibly many)
# fusions. It uses a breadth first search to achieve this.
def _get_all_connected_fusions(
    fusions: list[tuple[int, int]], node: int
) -> list[int]:
    seen: set[int] = {node}
    frontier: set[int] = {node}

    while len(frontier) != 0:
        v = frontier.pop()

        nbrs = get_fused_neighbours(fusions, v)
        new_nbrs = set(nbrs) - seen
        seen = seen.union(set(nbrs))
        frontier = frontier.union(new_nbrs)

    return list(seen)


def get_fused_neighbours(
    fusions: list[tuple[int, int]], node: int
) -> list[int]:
    """Returns the nodes that are fused to the given node"""
    fusion_nbrs = []

    for fusion in fusions:
        if fusion[0] == node:
            fusion_nbrs.append(fusion[1])
        elif fusion[1] == node:
            fusion_nbrs.append(fusion[0])

    return fusion_nbrs
