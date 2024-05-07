from optyx.graphs import Graph, find_min_path_cover, Node


def test_path_cover_star():
    # This is a star graph so it should find a path cover with 3 paths
    g = Graph(
        {
            0: Node({1, 2, 3, 4}),
            1: Node({0}),
            2: Node({0}),
            3: Node({0}),
            4: Node({0}),
        }
    )

    paths = find_min_path_cover(g)
    assert len(paths) == 3


def test_complete():
    # This is a fully connected graph, so it only requires one path
    g = Graph(
        {
            0: Node({1, 2, 3, 4}),
            1: Node({0, 2, 3, 4}),
            2: Node({0, 1, 3, 4}),
            3: Node({0, 1, 2, 4}),
            4: Node({0, 1, 2, 3}),
        }
    )

    paths = find_min_path_cover(g)
    assert len(paths) == 1


def test_something_weird_looking():
    # 0         3
    #  \       /
    #   1 - - 4
    #  /       \
    # 2         5
    g = Graph(
        {
            0: Node({1}),
            1: Node({0, 2, 4}),
            2: Node({1}),
            3: Node({4}),
            4: Node({1, 3, 5}),
            5: Node({4}),
        }
    )

    paths = find_min_path_cover(g)
    assert len(paths) == 2
