"""Contains the fundamental classes requirement to define MBQC patterns"""

from typing import Callable, Optional
from copy import deepcopy
from dataclasses import dataclass
import networkx as nx
import graphix


@dataclass
class Measurement:
    """An MBQC measurement.

    :param angle: the angle of the measurement. Should be between [0, 2pi)
    :param plane: the measurement plane: 'XY', 'XZ', 'YZ'
    """

    angle: float
    plane: str

    def __eq__(self, other):
        return (
            abs(self.angle - other.angle) < 0.001 and self.plane == other.plane
        )

    def is_z_measurement(self) -> bool:
        """Indicates whether it is a Z measurement"""
        return abs(self.angle) < 0.001 and self.plane == "XY"


@dataclass
class Fusion:
    """A fusion between two nodes

    :param node1: ID of one of the nodes in the fusion
    :param node2: ID of the other node in the fusion
    :param fusion_type: The type of fusion. Currently either: "X", "Y"
    """

    node1: int
    node2: int
    fusion_type: str

    def __eq__(self, other) -> bool:
        if self.fusion_type != other.fusion_type:
            return False

        if self.node1 == other.node1 and self.node2 == other.node2:
            return True

        if self.node1 == other.node2 and self.node2 == other.node1:
            return True

        return False

    def contains(self, node_id: int) -> bool:
        """Indicates whether the node is part of the fusion"""
        return node_id in (self.node1, self.node2)


@dataclass
class FusionNetwork:
    """A fusion network for a single emitter linear resource state

    Specifies the measurements and fusions to perform on the nodes.
    Nodes are indexed by their position in the single linear resource state
    i.e. the first node is 0, second is 1 etc.


    :param path: IDs of the nodes in the path. A node may be implemented by
        many photons
    :param measurements: The measurements applied to each of the nodes. Key is
        the indexed by the Node ID
    :param fusions: Tuples containing the IDs of nodes connected by a fusion
    """

    path: list[int]

    measurements: dict[int, Measurement]

    fusions: list[Fusion]


# Given a node, returns all the nodes in it's past
PartialOrder = Callable[[int], list[int]]


@dataclass
class GFlow:
    """The g-flow structure on an open graph.

    :param g: the function which returns the correction set of a node
    :param layers: the layers of the partial order. The key is a node ID and
        the value is the layer it belongs to in a particular foliation of the
        partial order.
    """

    g: dict[int, set[int]]
    layers: dict[int, int]

    def partial_order(self) -> PartialOrder:
        """Returns a function representing the partial order of the flow"""

        def order(n: int):
            l = self.layers[n]

            return [i for i, layer in self.layers.items() if layer >= l]

        return order


@dataclass
class OpenGraph:
    """Open graph contains the graph, measurement, and input and output
    nodes. This is the graph we wish to implement deterministically


    :param inside: the underlying graph state
    :param measurements: a dictionary whose key is the ID of a node and the
        value is the measurement at that node
    :param inputs: a set of IDs of the nodes that are inputs to the graph
    :param outputs: a set of IDs of the nodes that are outputs of the graph

    Example
    -------
    >>> import networkx as nx
    >>> from . import OpenGraph, Measurement
    >>>
    >>> inside_graph = nx.Graph([(0, 1), (1, 2), (2, 0)])
    >>>
    >>> measurements = [Measurement(0.5 * i, "XY") for i in range(3)]
    >>> inputs = {0}
    >>> outputs = {2}
    >>> og = OpenGraph(inside_graph, measurements, inputs, outputs)
    """

    inside: nx.Graph

    measurements: dict[int, Measurement]

    inputs: set[int]
    outputs: set[int]

    def __eq__(self, other):
        """Checks the two open graphs are equal

        This doesn't check they are equal up to an isomorphism"""

        g1 = self.perform_z_deletions()
        g2 = other.perform_z_deletions()

        return (
            g1.inputs == g2.inputs
            and g1.outputs == g2.outputs
            and nx.utils.graphs_equal(g1.inside, g2.inside)
            and g1.measurements == g2.measurements
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
            id for id, m in self.measurements.items() if m.is_z_measurement()
        ]

        for node in zero_nodes:
            self.inside.remove_node(node)

        # Remove the deleted node's measurements
        self.measurements = {
            id: m
            for id, m in self.measurements.items()
            if not m.is_z_measurement()
        }

    def perform_z_deletions(self):
        """Removes the Z-deleted nodes from the graph"""
        g = deepcopy(self)
        g.perform_z_deletions_in_place()
        return g

    def find_gflow(self) -> Optional[GFlow]:
        """Finds gflow of the open graph.

        Returns None if it does not exist."""

        meas_planes = {i: meas.plane for i, meas in self.measurements.items()}
        g, layers = graphix.gflow.find_gflow(
            self.inside, self.inputs, self.outputs, meas_planes
        )

        if g is None or layers is None:
            return None

        return GFlow(g, layers)


def add_fusions_to_partial_order(
    fusions: list[Fusion], order: PartialOrder
) -> PartialOrder:
    """Returns a new partial order that ensures Pauli errors from Hadamard edge
    fusions can be corrected.

    :param fusions: A list of tuples of node IDs representing a fusion between
                    the two nodes
    :param order: A partial order on the nodes

    We achieve this the simplest way possible, namely, we want the following
    condition to be satisfied in the new order: if nodes v and w are fused,
    then any node that has w in its past, should now contain v and its past and
    vice versa.
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


def _get_all_connected_fusions(fusions: list[Fusion], node: int) -> list[int]:
    """Find every node that is connected to the given node through fusions. It
    uses a breadth first search to achieve this.

    :param fusions: A list of tuples of node IDs representing a fusion between
                    the two nodes
    :param node: The ID of the node in question.

    Example
    -------
    >>> fusions = [Fusion(1, 0, "X"), Fusion(2, 4, "X"), Fusion(0, 3, "X")]
    >>> _get_all_connected_fusions(fusions, 0)
    [0, 1, 3]
    >>> _get_all_connected_fusions(fusions, 1)
    [0, 1, 3]
    >>> _get_all_connected_fusions(fusions, 4)
    [2, 4]
    """
    seen: set[int] = {node}
    frontier: set[int] = {node}

    while len(frontier) != 0:
        v = frontier.pop()

        nbrs = get_fused_neighbours(fusions, v)
        new_nbrs = set(nbrs) - seen
        seen = seen.union(set(nbrs))
        frontier = frontier.union(new_nbrs)

    return list(seen)


def get_fused_neighbours(fusions: list[Fusion], node: int) -> list[int]:
    """Returns the nodes that are fused to the given node.

    :param fusions: A list of tuples of node IDs representing a fusion between
                    the two nodes
    :param node: The ID of the node in question.

    Example
    -------
    >>> fusions = [Fusion(1, 0, "X"), Fusion(2, 3, "X"), Fusion(0, 3, "X")]
    >>> get_fused_neighbours(fusions, 0)
    [1, 3]
    >>> get_fused_neighbours(fusions, 2)
    [3]
    >>> get_fused_neighbours(fusions, 4)
    []
    """
    fusion_nbrs = []

    for fusion in fusions:
        if fusion.node1 == node:
            fusion_nbrs.append(fusion.node2)
        elif fusion.node2 == node:
            fusion_nbrs.append(fusion.node1)

    return fusion_nbrs
