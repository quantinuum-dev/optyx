"""Contains the fundamental classes requirement to define MBQC patterns"""

from typing import Callable, Optional, Self
from copy import deepcopy
from dataclasses import dataclass
import networkx as nx
import graphix
import pyzx as zx

from graphix.sim.statevec import Statevec
from graphix.generator import generate_from_graph


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

    Example
    -------
    >>> from optyx.compiler.mbqc import Fusion
    >>> Fusion(0, 1, "X") == Fusion(0, 1, "X")
    True
    >>> Fusion(0, 1, "X") == Fusion(0, 1, "Y")
    False
    >>> Fusion(0, 1, "X") == Fusion(1, 0, "X")
    True
    >>> Fusion(0, 1, "X") == Fusion(0, 2, "X")
    False
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


    :param graph: the underly graph the fusion network implements
    :param resources: the linear resource states used to comprise the graph. We
        use lists of node IDs rather than subgraphs so we know the start and
        end points of the resource states
    :param measurements: The measurements applied to each of the nodes. Key is
        the indexed by the Node ID
    """

    resources: list[list[int]]
    measurements: dict[int, Measurement]
    fusions: list[Fusion]

    def nodes(self) -> list[int]:
        """Returns a list of all nodes in the fusion network"""
        all_nodes: list[int] = []
        for resource in self.resources:
            all_nodes.extend(resource)
        return all_nodes


@dataclass
class ULFusionNetwork:
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

            return [i for i, layer in self.layers.items() if layer > l] + [n]

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
    >>> measurements = {i: Measurement(0.5 * i, "XY") for i in range(2)}
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

        if any(node in outputs for node in measurements):
            raise ValueError("output node can not be measured")

        for node_id, measurement in measurements.items():
            self.inside.nodes[node_id]["measurement"] = measurement

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
        >>> measurements = {i: Measurement(0.5 * i, "XY") for i in range(2)}
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
        >>> measurements = {i: Measurement(0.5 * i, "XY") for i in range(2)}
        >>> inputs = [0]
        >>> outputs = [2]
        >>> og = OpenGraph(inside_graph, measurements, inputs, outputs)
        >>>
        >>> inside_graph2 = nx.Graph([(1, 0), (0, 3), (2, 0)])
        >>> measurements2 = {i: Measurement(0.7 * i, "XY") for i in range(3)}
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
        if self == OpenGraph.id():
            return other
        if other == OpenGraph.id():
            return self

        nodes = self.inside.nodes

        if len(self.inputs) == 0:
            max_input_order = -1
        else:
            max_input_order = max(nodes[i]["input_order"] for i in self.inputs)

        if len(self.outputs) == 0:
            max_output_order = -1
        else:
            max_output_order = max(
                nodes[i]["output_order"] for i in self.outputs
            )

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

        Unexported since this relies on the underlying implementation details
        of the graph, which could change at any time. It is meant to be a
        convenient utility tool to allow us to more efficiently create new open
        graphs in our own methods.

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
        >>> g.nodes[0]["is_input"] = True
        >>> g.nodes[0]["input_order"] = 0
        >>> g.nodes[3]["is_input"] = True
        >>> g.nodes[3]["input_order"] = 1
        >>> g.nodes[2]["is_output"] = True
        >>> g.nodes[2]["output_order"] = 0
        """
        og = cls.__new__(cls)
        og.inside = graph
        return og

    def to_pyzx_graph(self) -> zx.graph.base.BaseGraph:
        """Return a PyZX graph corresponding to the the open graph.

        Example
        -------
        >>> import networkx as nx
        >>> g = nx.Graph([(0, 1), (1, 2)])
        >>> inputs = [0]
        >>> outputs = [2]
        >>> measurements = {0: Measurement(0, "XY"), 1: Measurement(1, "ZY")}
        >>> og = OpenGraph(g, measurements, inputs, outputs)
        >>> reconstructed_pyzx_graph = og.to_pyzx_graph()
        """
        g = zx.Graph()

        # Add vertices into the graph and set their type
        def add_vertices(n: int, ty: zx.VertexType) -> list[zx.VertexType]:
            verts = g.add_vertices(n)
            for vert in verts:
                g.set_type(vert, ty)

            return verts

        # Add input boundary nodes
        in_verts = add_vertices(len(self.inputs), zx.VertexType.BOUNDARY)
        g.set_inputs(in_verts)

        # Add nodes for internal Z spiders - not including the phase gadgets
        body_verts = add_vertices(len(self.inside), zx.VertexType.Z)

        # Add nodes for the phase gadgets. In OpenGraph we don't store the node
        # containing the phase as a seperate node, it is instead just stored as
        # a "measurement" of the node.
        # That is, it is not stored like example 1 and instead like 2 and the
        # measurement metadata is added to node "v".
        #   1           2
        # o            o
        #  \            \
        #   v - O        v
        #  /            /
        # o            o
        optyx_x_meas = [
            i for i, m in self.measurements.items() if m.plane == "YZ"
        ]
        x_meas_verts = add_vertices(len(optyx_x_meas), zx.VertexType.Z)

        out_verts = add_vertices(len(self.outputs), zx.VertexType.BOUNDARY)
        g.set_outputs(out_verts)

        # Maps a node's ID in the optyx Open Graph to it's corresponding node
        # ID in the PyZX graph and vice versa.
        map_to_optyx = dict(zip(body_verts, self.inside.nodes()))
        map_to_pyzx = {v: i for i, v in map_to_optyx.items()}

        # Open Graph's don't have boundary nodes, so we need to connect the
        # input and output Z spiders to their corresponding boundary nodes in
        # pyzx.
        for pyzx_index, optyx_index in zip(in_verts, self.inputs):
            g.add_edge((pyzx_index, map_to_pyzx[optyx_index]))
        for pyzx_index, optyx_index in zip(out_verts, self.outputs):
            g.add_edge((pyzx_index, map_to_pyzx[optyx_index]))

        optyx_edges = self.inside.edges()
        pyzx_edges = [(map_to_pyzx[a], map_to_pyzx[b]) for a, b in optyx_edges]
        g.add_edges(pyzx_edges, zx.EdgeType.HADAMARD)

        # Add the edges between the Z spiders in the graph body
        for optyx_index, meas in self.measurements.items():
            # If it's an X measured node, then we handle it in the next loop
            if meas.plane == "XY":
                g.set_phase(map_to_pyzx[optyx_index], meas.angle)

        # Connect the X measured vertices
        for optyx_index, pyzx_index in zip(optyx_x_meas, x_meas_verts):
            g.add_edge(
                (map_to_pyzx[optyx_index], pyzx_index), zx.EdgeType.HADAMARD
            )
            g.set_phase(pyzx_index, self.measurements[optyx_index].angle)

        return g

    @classmethod
    def from_pyzx_graph(cls, g: zx.graph.base.BaseGraph) -> Self:
        """Constructs an Optyx Open Graph from a PyZX graph.

        NOTE: It may modify the original graph

        Example
        -------
        >>> import pyzx as zx
        >>> from optyx.compiler import OpenGraph
        >>> circ = zx.qasm("qreg q[2]; h q[1]; cx q[0], q[1]; h q[1];")
        >>> g = circ.to_graph()
        >>> optyx_graph = OpenGraph.from_pyzx_graph(g)
        """
        zx.simplify.to_graph_like(g)
        zx.simplify.full_reduce(g)

        measurements = {}
        inputs = g.inputs()
        outputs = g.outputs()

        g_nx = nx.Graph(g.edges())

        # We need to do this since the full reduce simplification can
        # leave either hadamard or plain wires on the inputs and outputs
        for inp in g.inputs():
            nbrs = list(g.neighbors(inp))
            et = g.edge_type((nbrs[0], inp))

            if et == zx.EdgeType.SIMPLE:
                g_nx.remove_node(inp)
                inputs = [i if i != inp else nbrs[0] for i in inputs]

        for out in g.outputs():
            nbrs = list(g.neighbors(out))
            et = g.edge_type((nbrs[0], out))

            if et == zx.EdgeType.SIMPLE:
                g_nx.remove_node(out)
                outputs = [o if o != out else nbrs[0] for o in outputs]

        # Turn all phase gadgets into measurements
        # Since we did a full reduce, any node that isn't an input or output
        # node and has only one neighbour is definitely a phase gadget.
        nodes = list(g_nx.nodes())
        for v in nodes:
            if v in inputs or v in outputs:
                continue

            nbrs = list(g.neighbors(v))
            if len(nbrs) == 1:
                measurements[nbrs[0]] = Measurement(g.phase(v), "YZ")
                g_nx.remove_node(v)

        next_id = max(g_nx.nodes) + 1

        # Since outputs can't be measured, we need to add an extra two nodes
        # in to counter it
        for out in outputs:
            if g.phase(out) == 0:
                continue

            g_nx.add_edges_from([(out, next_id), (next_id, next_id + 1)])
            measurements[next_id] = Measurement(0, "XY")

            outputs = [o if o != out else next_id + 1 for o in outputs]

            next_id += 2

        # Add the phase to all XY measured nodes
        for v in g_nx.nodes:
            if v in outputs or v in measurements:
                continue

            measurements[v] = Measurement(g.phase(v), "XY")

        return cls(g_nx, measurements, inputs, outputs)

    def then(self, other: Self) -> Self:
        """Sequentially composing the graph with the given graph

        Example
        -------
        >>> import networkx as nx
        >>> from optyx.compiler.mbqc import OpenGraph, Measurement
        >>>
        >>> inside_graph = nx.Graph([(0, 1), (1, 2), (2, 0)])
        >>> measurements = {i: Measurement(0.5 * i, "XY") for i in range(2)}
        >>> inputs = [0]
        >>> outputs = [2]
        >>> og = OpenGraph(inside_graph, measurements, inputs, outputs)
        >>>
        >>> inside_graph2 = nx.Graph([(1, 0), (0, 3), (2, 0)])
        >>> measurements2 = {i: Measurement(0.7 * i, "XY") for i in range(3)}
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
        relabeled_inside = nx.relabel_nodes(
            other.inside, lambda x: x + max_node + 1
        )
        other_copy = self._from_networkx(relabeled_inside)

        # Set the outputs and inputs to be the same ID
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

        # Note that the "measurement" attribute in "other_copy"'s node will
        # appear in the corresponding node in the returned graph as expected.
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
        >>> measurements = {i: Measurement(0.5 * i, "XY") for i in range(2)}
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
        >>> measurements = {i: Measurement(0.5 * i, "XY") for i in range(2)}
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
        >>> measurements = {i: Measurement(0.5 * i, "XY") for i in range(2)}
        >>> inputs = [0]
        >>> outputs = [2]
        >>>
        >>> og = OpenGraph(inside_graph, measurements, inputs, outputs)
        >>> assert og.measurements == {
        ...     0: Measurement(0.0, "XY"),
        ...     1: Measurement(0.5, "XY"),
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

    def simulate(self, backend="statevector", **kwargs) -> Statevec:
        """Simulates the measurement pattern and returns the state vector.

        :param backend: either 'statevector' (default) or 'tensornetwork'

        Example
        -------
        >>> import networkx as nx
        >>> from optyx.compiler.mbqc import OpenGraph, Measurement
        >>> graph = nx.Graph([(0, 1), (1, 2), (2, 3)])
        >>> measurements = {i: Measurement(0.5, "XY") for i in range(3)}
        >>> inputs = [0]
        >>> outputs = [3]
        >>> og = OpenGraph(graph, measurements, inputs, outputs)
        >>> og.simulate()
        Statevec, data=[0.5+0.5j 0.5+0.5j], shape=(2,)
        """
        measurements = self.measurements
        angles = {
            i: float(m.angle) if float(m.angle) < 1.0 else float(m.angle) - 2.0
            for i, m in measurements.items()
        }
        # Lets see if this works
        angles = {i: -angle for i, angle in angles.items()}
        meas_planes = {i: m.plane for i, m in measurements.items()}
        inputs = set(self.inputs)
        outputs = set(self.outputs)

        pattern = generate_from_graph(
            self.inside, angles, inputs, outputs, meas_planes
        )

        return pattern.simulate_pattern(backend, **kwargs)


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
