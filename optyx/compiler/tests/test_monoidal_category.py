import networkx as nx
from optyx.compiler.mbqc import OpenGraph, Measurement


# Explicitly test sequential composition of open graphs gives us the exact
# graph we expect
def test_open_graph_sequential_composition():
    inside = nx.Graph([(0, 1), (1, 2), (2, 0)])
    measurements = {i: Measurement(0.5 * i, "XY") for i in range(2)}
    inputs = [0]
    outputs = [2]
    graph1 = OpenGraph(inside, measurements, inputs, outputs)

    inside2 = nx.Graph([(1, 0), (0, 3), (2, 0)])
    measurements2 = {i: Measurement(0.7 * i, "XY") for i in range(3)}
    inputs2 = [1]
    outputs2 = [3]
    graph2 = OpenGraph(inside2, measurements2, inputs2, outputs2)

    result_graph = graph1.then(graph2)

    exp_inside = nx.Graph([(0, 1), (1, 2), (2, 0), (2, 3), (3, 6), (5, 3)])
    exp_meas = {
        0: Measurement(0.0, "XY"),
        1: Measurement(0.5, "XY"),
        2: Measurement(0.7, "XY"),
        3: Measurement(0.0, "XY"),
        5: Measurement(1.4, "XY"),
    }
    exp_inputs = [0]
    exp_outputs = [6]

    expected_graph = OpenGraph(exp_inside, exp_meas, exp_inputs, exp_outputs)

    assert result_graph == expected_graph


# Explicitly test paralle composition of open graphs gives us the exact graph
# we expect
def test_open_graph_parallel_composition():
    inside = nx.Graph([(0, 1), (1, 2), (2, 0)])
    measurements = {i: Measurement(0.5 * i, "XY") for i in range(2)}
    inputs = [0]
    outputs = [2]
    graph1 = OpenGraph(inside, measurements, inputs, outputs)

    inside2 = nx.Graph([(1, 0), (0, 3), (2, 0)])
    measurements2 = {i: Measurement(0.7 * i, "XY") for i in range(3)}
    inputs2 = [1]
    outputs2 = [3]
    graph2 = OpenGraph(inside2, measurements2, inputs2, outputs2)

    result_graph = graph1.tensor(graph2)

    exp_inside = nx.Graph([(0, 1), (1, 2), (2, 0), (4, 3), (3, 6), (5, 3)])
    exp_meas = {
        0: Measurement(0.0, "XY"),
        1: Measurement(0.5, "XY"),
        3: Measurement(0.0, "XY"),
        4: Measurement(0.7, "XY"),
        5: Measurement(1.4, "XY"),
    }
    exp_inputs = [0, 4]
    exp_outputs = [2, 6]
    expected_graph = OpenGraph(exp_inside, exp_meas, exp_inputs, exp_outputs)

    assert result_graph == expected_graph
