"""Contains the fundamental classes requirement to define MBQC patterns"""

from typing import Callable, Optional, Self
from copy import deepcopy
from dataclasses import dataclass
import networkx as nx
import graphix
import numpy as np


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
        return np.allclose(self.angle, 0.0) and self.plane == "XY"


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
    >>> from optyx.compiler.mbqc import OpenGraph, Measurement
    >>>
    >>> inside_graph = nx.Graph([(0, 1), (1, 2), (2, 0)])
    >>>
    >>> measurements = [Measurement(0.5 * i, "XY") for i in range(3)]
    >>> inputs = [0]
    >>> outputs = [2]
    >>> og = OpenGraph(inside_graph, measurements, inputs, outputs)
    """

    inside: nx.Graph

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

    def __init__(
        self,
        inside: nx.Graph,
        measurements: dict[int, Measurement],
        inputs: list[int],
        outputs: list[int],
    ):
        self.inside = inside

        for i in self.inside.nodes:
            self.inside.nodes[i]["measurement"] = measurements[i]

        for i, node_id in enumerate(inputs):
            self.inside.nodes[node_id]["is_input"] = True
            self.inside.nodes[node_id]["input_order"] = i

        for i, node_id in enumerate(outputs):
            self.inside.nodes[node_id]["is_output"] = True
            self.inside.nodes[node_id]["output_order"] = i

    # The following functions "id", "then" and "tensor" define OpenGraph as a
    # monoidal category

    @staticmethod
    def id():
        """Returns the identity of the tensor product

        Example
        -------
        >>> import networkx as nx
        >>> from optyx.compiler.mbqc import OpenGraph, Measurement
        >>>
        >>> inside_graph = nx.Graph([(0, 1), (1, 2), (2, 0)])
        >>> measurements = [Measurement(0.5 * i, "XY") for i in range(3)]
        >>> inputs = [0]
        >>> outputs = [2]
        >>> og = OpenGraph(inside_graph, measurements, inputs, outputs)
        >>>
        >>> id_graph = OpenGraph.id()
        >>> new_graph = og.tensor(id_graph)
        >>> assert new_graph == og
        """
        return OpenGraph(nx.Graph(), {}, [], [])

    def tensor(self, other: Self) -> Self:
        """Parallel compose the graph with the given graph

        Example
        -------
        >>> import networkx as nx
        >>> from optyx.compiler.mbqc import OpenGraph, Measurement
        >>>
        >>> inside_graph = nx.Graph([(0, 1), (1, 2), (2, 0)])
        >>> measurements = [Measurement(0.5 * i, "XY") for i in range(3)]
        >>> inputs = [0]
        >>> outputs = [2]
        >>> og = OpenGraph(inside_graph, measurements, inputs, outputs)
        >>>
        >>> inside_graph2 = nx.Graph([(1, 0), (0, 3), (2, 0)])
        >>> measurements2 = [Measurement(0.7 * i, "XY") for i in range(4)]
        >>> inputs2 = [1]
        >>> outputs2 = [3]
        >>> og2 = OpenGraph(inside_graph2, measurements2, inputs2, outputs2)
        >>>
        >>> new_graph = og.tensor(og2)
        >>>
        >>> from networkx.algorithms.isomorphism.vf2userfunc import (
        ...     GraphMatcher
        ... )
        >>> gm1 = GraphMatcher(new_graph.inside, og.inside)
        >>> assert gm1.subgraph_is_isomorphic()
        >>> gm2 = GraphMatcher(new_graph.inside, og2.inside)
        >>> assert gm2.subgraph_is_isomorphic()
        """
        # NOTE: This could be simplified by doing my own node renaming before
        # the disjoint_union so I know what the new node IDs will be
        nodes = self.inside.nodes
        max_input_order = max(nodes[i]["input_order"] for i in self.inputs)
        max_output_order = max(nodes[i]["output_order"] for i in self.outputs)

        max_node = max(self.inside.nodes)
        relabeled_inside = nx.relabel_nodes(
            other.inside, lambda x: x + max_node + 1
        )

        for n in relabeled_inside.nodes(data=True):
            if "is_input" in relabeled_inside.nodes[n[0]]:
                relabeled_inside.nodes[n[0]]["input_order"] += (
                    max_input_order + 1
                )
            if "is_output" in relabeled_inside.nodes[n[0]]:
                relabeled_inside.nodes[n[0]]["output_order"] += (
                    max_output_order + 1
                )

        inside = nx.union(self.inside, relabeled_inside)

        return self._from_networkx(inside)

    @classmethod
    def _from_networkx(cls, graph: nx.Graph) -> Self:
        """Returns an OpenGraph built from an networkx graph. Additional
        information is attached the nodes as attributes

        Attributes:

        * "measurement" - instance of `optyx.compiler.Measurement`
        * "is_input" - only present on input nodes and is always `True`
        * "input_order" - the position of this input node in the inputs (starts
                          at 0)
        * "is_output" - only present on output nodes and is always `True`
        * "output_order" - the position of this output node in the outputs
                           (starts at 0)

        Example
        -------
        A graph with four vertices where nodes 0 and 3 are inputs in the order
        (0, 3), and node 2 is the only output.

        >>> import networkx as nx
        >>> from optyx.compiler.mbqc import OpenGraph, Measurement
        >>>
        >>> g = nx.Graph([(0, 1), (1, 2), (2, 0), (3, 0)])
        >>> g.nodes[0]["measurement"] = Measurement(0.0, "XY")
        >>> g.nodes[1]["measurement"] = Measurement(0.5, "XY")
        >>> g.nodes[3]["measurement"] = Measurement(0, "XY")
        >>>
        >>> g.nodes[0]["class"] = "input"
        >>> g.nodes[0]["order"] = 0
        >>> g.nodes[3]["class"] = "input"
        >>> g.nodes[3]["order"] = 1
        >>> g.nodes[2]["class"] = "output"
        >>> g.nodes[2]["order"] = 0
        """
        og = cls.__new__(cls)
        og.inside = graph
        return og

    def then(self, other: Self) -> Self:
        """Sequentially composing the graph with the given graph

        Example
        -------
        >>> import networkx as nx
        >>> from optyx.compiler.mbqc import OpenGraph, Measurement
        >>>
        >>> inside_graph = nx.Graph([(0, 1), (1, 2), (2, 0)])
        >>> measurements = [Measurement(0.5 * i, "XY") for i in range(3)]
        >>> inputs = [0]
        >>> outputs = [2]
        >>> og = OpenGraph(inside_graph, measurements, inputs, outputs)
        >>>
        >>> inside_graph2 = nx.Graph([(1, 0), (0, 3), (2, 0)])
        >>> measurements2 = [Measurement(0.7 * i, "XY") for i in range(4)]
        >>> inputs2 = [1]
        >>> outputs2 = [3]
        >>> og2 = OpenGraph(inside_graph2, measurements2, inputs2, outputs2)
        >>>
        >>> new_graph = og.then(og2)
        >>>
        >>> from networkx.algorithms.isomorphism.vf2userfunc import (
        ...     GraphMatcher
        ... )
        >>> gm = GraphMatcher(new_graph.inside, og.inside)
        >>> assert gm.subgraph_is_isomorphic()
        >>> gm = GraphMatcher(new_graph.inside, og2.inside)
        >>> assert gm.subgraph_is_isomorphic()
        """
        if len(self.outputs) != len(self.inputs):
            raise ValueError(
                f"cannot compose graph with {len(self.outputs)} "
                f"outputs with graph with {len(self.inputs)} inputs"
            )

        max_node = max(self.inside.nodes)
        relabeled_other_inside = nx.relabel_nodes(
            other.inside, lambda x: x + max_node + 1
        )

        # Avoid calling the init method since we don't need to load the
        # measurements and inputs/outputs again
        other_copy = OpenGraph.__new__(OpenGraph)
        other_copy.inside = relabeled_other_inside

        # Now I need to set the outputs and inputs to be the same ID
        all_data = self.inside.nodes(data=True)
        old_self_outputs = sorted(
            self.outputs, key=lambda x: all_data[x]["output_order"]
        )

        relab_data = other_copy.inside.nodes(data=True)
        old_other_inputs = sorted(
            other_copy.inputs, key=lambda x: relab_data[x]["input_order"]
        )

        for node_id in old_other_inputs:
            del other_copy.inside.nodes[node_id]["is_input"]
            del other_copy.inside.nodes[node_id]["input_order"]

        mapping = {
            old_other_inputs[i]: old_self_outputs[i]
            for i in range(len(old_other_inputs))
        }
        other_copy.inside = nx.relabel_nodes(other_copy.inside, mapping)

        result = nx.compose(self.inside, other_copy.inside)

        for node_id in old_self_outputs:
            del result.nodes[node_id]["is_output"]
            del result.nodes[node_id]["output_order"]

        return self._from_networkx(result)

    @property
    def inputs(self) -> list[int]:
        """Returns the inputs of the graph.

        Example
        ------
        >>> import networkx as nx
        >>> from optyx.compiler.mbqc import OpenGraph, Measurement
        >>>
        >>> inside_graph = nx.Graph([(0, 1), (1, 2), (2, 0)])
        >>> measurements = [Measurement(0.5 * i, "XY") for i in range(3)]
        >>> inputs = [0]
        >>> outputs = [2]
        >>>
        >>> og = OpenGraph(inside_graph, measurements, inputs, outputs)
        >>> assert og.inputs == inputs
        """
        unsorted_inputs = [
            i for i in self.inside.nodes(data=True) if "is_input" in i[1]
        ]
        inputs_with_data = sorted(
            unsorted_inputs, key=lambda x: x[1]["input_order"]
        )
        inputs = [i[0] for i in inputs_with_data]
        return inputs

    @property
    def outputs(self) -> list[int]:
        """Returns the outputs of the graph.

        Example
        ------
        >>> import networkx as nx
        >>> from optyx.compiler.mbqc import OpenGraph, Measurement
        >>>
        >>> inside_graph = nx.Graph([(0, 1), (1, 2), (2, 0)])
        >>> measurements = [Measurement(0.5 * i, "XY") for i in range(3)]
        >>> inputs = [0]
        >>> outputs = [2]
        >>>
        >>> og = OpenGraph(inside_graph, measurements, inputs, outputs)
        >>> assert og.outputs == outputs
        """
        unsorted_outputs = [
            i for i in self.inside.nodes(data=True) if "is_output" in i[1]
        ]
        outputs_with_data = sorted(
            unsorted_outputs, key=lambda x: x[1]["output_order"]
        )
        outputs = [i[0] for i in outputs_with_data]
        return outputs

    @property
    def measurements(self) -> dict[int, Measurement]:
        """Returns a dictionary which maps each node to its measurement. Output
        nodes are not measured and hence are not included in the dictionary.

        Example
        ------
        >>> import networkx as nx
        >>> from optyx.compiler.mbqc import OpenGraph, Measurement
        >>>
        >>> inside_graph = nx.Graph([(0, 1), (1, 2), (2, 0)])
        >>> measurements = [Measurement(0.5 * i, "XY") for i in range(3)]
        >>> inputs = [0]
        >>> outputs = [2]
        >>>
        >>> og = OpenGraph(inside_graph, measurements, inputs, outputs)
        >>> assert og.measurements == {
        ...     0: Measurement(0.0, "XY"),
        ...     1: Measurement(0.5, "XY"),
        ...     2: Measurement(1.0, "XY"),
        ... }
        """
        return {
            n[0]: n[1]["measurement"]
            for n in self.inside.nodes(data=True)
            if "measurement" in n[1]
        }

    def perform_z_deletions_in_place(self):
        """Removes the Z-deleted nodes from the graph in place"""
        z_measured_nodes = [
            node
            for node in self.inside.nodes
            if "measurement" in self.inside.nodes[node]
            and self.inside.nodes[node]["measurement"].is_z_measurement()
        ]

        for node in z_measured_nodes:
            self.inside.remove_node(node)

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
            self.inside, set(self.inputs), set(self.outputs), meas_planes
        )

        if g is None or layers is None:
            return None

        return GFlow(g, layers)


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
