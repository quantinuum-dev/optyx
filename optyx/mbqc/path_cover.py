"""Provides algorithm for computing the minimum path cover of a graph"""

from typing import Optional
import networkx as nx


def find_min_path_cover(g: nx.Graph) -> list[list[int]]:
    """Returns a minimum path cover of the graph.
    This uses a brute force algorithm and so it only works on small graphs with
    approximately less than 15 vertices.


    Example
    -------
    # This is a star graph so it should find a path cover with 3 paths
    >>> g = nx.Graph([(0, 1), (0, 2), (0, 3), (0, 4)])
    >>> paths = find_min_path_cover(g)
    >>> assert len(paths) == 3

    # Fully connected graph, so it only requires one path
    >>> g = nx.Graph([(i, j) for i in range(4) for j in range(i)])
    >>> paths = find_min_path_cover(g)
    >>> assert len(paths) == 1

    # Test something weird looking
    >>> g = nx.Graph([(0, 1), (1, 2), (1, 4), (4, 3), (4, 5)])
    >>> paths = find_min_path_cover(g)
    >>> assert len(paths) == 2
    """
    for i in range(1, g.number_of_nodes()):
        paths: list[list[int]] = []

        path_cover = find_path_cover(g, paths, i)
        if path_cover is not None:
            return path_cover

    # This part is unreachable since in the worst case, we will return the
    # path cover where each path covers only a single vertex.
    raise ValueError("unreachable!")


PathCover = list[list[int]]


def find_path_cover(
    g: nx.Graph, paths: list[list[int]], max_paths: int
) -> Optional[PathCover]:
    """Returns whether there exists a path cover of the graph with numPaths
    number of paths. If it True, then the path cover is contained in the
    "paths" input variable"""

    # Checks whether the element exists in any of the paths in the slice
    def _in_paths(paths: list[list[int]], v: int) -> bool:
        return any(v in p for p in paths)

    # Recursive auxillary function. Which performs a depth first search for the
    # path cover.
    def _find_path_cover_aux(
        g: nx.Graph, paths: list[list[int]], num_paths: int
    ) -> Optional[PathCover]:

        num_nodes_in_paths = sum(len(path) for path in paths)
        if num_nodes_in_paths == g.number_of_nodes():
            return paths

        # The last node in the paths so far
        current_path = paths[-1]
        tail = current_path[-1]

        for nbr in g[tail]:
            if _in_paths(paths, nbr):
                continue

            current_path.append(nbr)
            paths[-1] = current_path

            path_cover = _find_path_cover_aux(g, paths, num_paths)
            if path_cover is not None:
                return paths

            current_path.pop()

        if num_paths > 1:
            return find_path_cover(g, paths, num_paths - 1)

        return None

    for vert in g.nodes:
        # The vertex already belongs to another path
        if _in_paths(paths, vert):
            continue

        paths.append([vert])
        if _find_path_cover_aux(g, paths, max_paths):
            return paths

        paths.pop()

    return None
