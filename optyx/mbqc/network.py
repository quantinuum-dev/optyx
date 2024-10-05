""" Classes for reasoning about fusion networks """

from dataclasses import dataclass

import networkx as nx

from .graph import Measurement, OpenGraph, PartialOrder


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


def add_fusion_order(
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


def fn_to_open_graph(
    sfn: FusionNetwork, inputs: set[int], outputs: set[int]
) -> OpenGraph:
    """Converts a fusion network into an open graph"""

    g = nx.path_graph(sfn.path)

    for fusion in sfn.fusions:
        g.add_edge(fusion.node1, fusion.node2)

    return OpenGraph(g, sfn.measurements, inputs, outputs)


def pattern_satisfies_order(
    measurements: list[tuple[int, Measurement]], order: PartialOrder
) -> bool:
    """Checks every measurement happens only after everything in its past has
    been measured."""
    seen: set[int] = set()

    for v, _ in measurements:
        past = order(v)
        seen.add(v)
        if not set(past).issubset(seen):
            return False

    return True
