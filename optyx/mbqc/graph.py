""" Classes for reasoning with open graph """

from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Optional

import graphix
import networkx as nx
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
        """Checks if two measurements are equal"""
        return (
            np.allclose(self.angle, other.angle) and self.plane == other.plane
        )

    def is_z_measurement(self) -> bool:
        """Indicates whether it is a Z measurement"""
        return np.allclose(self.angle, 0.0) and self.plane == "XY"


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
