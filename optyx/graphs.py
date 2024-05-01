"""Provides common graph functionality

Will most likely be replaced by networkx in future.
Wrote a custom library now for simplicity.
"""

from dataclasses import dataclass


@dataclass
class Node:
    """A node in the graph"""

    neigh: set[int]


@dataclass
class Graph:
    """Graph is a generic graph object

    Will most likely be replaced with a networkx backend at some point
    """

    # Key is the node id and the value is a list of neighbours
    nodes: dict[int, Node]

    def edges(self) -> set[tuple[int, int]]:
        """Returns a list of the edges in the graph"""
        edges: set[tuple[int, int]] = set()

        for node_id, node in self.nodes.items():
            for u in node.neigh:
                edges.add(_sorted_tuple(node_id, u))

        return edges

    def add_edge(self, v: int, w: int):
        """Adds an edge between node v and w.

        Is a no-op if the edge already exists"""
        node = self.nodes.get(v, Node(set()))
        node.neigh.add(w)
        self.nodes[v] = node

        node = self.nodes.get(w, Node(set()))
        node.neigh.add(v)
        self.nodes[w] = node

    def remove_node(self, v: int):
        """Removes a node from the graph"""
        del self.nodes[v]

        for u, node in self.nodes.items():
            self.nodes[u] = Node({w for w in node.neigh if w != v})


def _sorted_tuple(a: int, b: int) -> tuple[int, int]:
    return (min(a, b), max(a, b))


def find_min_path_cover(g: Graph) -> list[list[int]]:
    """Returns a minimum path cover of the graph.
    This uses a brute force algorithm and so it only works on small graphs with
    approximately less than 15 vertices."""
    for i in range(1, len(g.nodes)):
        paths: list[list[int]] = []

        ok = find_min_path_cover_with_k_paths(g, paths, i)
        if ok:
            return paths

    # This part is unreachable since in the worst case, we will return the
    # path cover where each path covers only a single vertex.
    raise ValueError("unreachable!")


def find_min_path_cover_with_k_paths(
    g: Graph, paths: list[list[int]], num_paths: int
) -> bool:
    """Returns whether there exists a path cover of the graph with numPaths
    number of paths. If it True, then the path cover is contained in the
    "paths" input variable"""
    for vert in g.nodes:
        # The vertex already belongs to another path
        if _in_paths(paths, vert):
            continue

        paths.append([vert])
        if _find_path_cover_aux(g, paths, num_paths):
            return True

        paths.pop()

    return False


# The recursive auxillary function to FindMinPathCoverWithKPaths. Returns
# whether there exists a path covering with the given number of paths
def _find_path_cover_aux(
    g: Graph, paths: list[list[int]], num_paths: int
) -> bool:
    if _total_length(paths) == len(g.nodes):
        return True

    # The last node in the paths so far
    current_path = paths[-1]
    tail = current_path[-1]

    for edge in g.nodes[tail].neigh:
        if _in_paths(paths, edge):
            continue

        current_path.append(edge)
        paths[-1] = current_path

        ok = _find_path_cover_aux(g, paths, num_paths)
        if ok:
            return True

        current_path.pop()

    if num_paths > 1:
        return find_min_path_cover_with_k_paths(g, paths, num_paths - 1)

    return False


# Checked whether the element exists in any of the paths in the slice
def _in_paths(paths: list[list[int]], v: int) -> bool:
    return any(v in p for p in paths)


def _total_length(p: list[list[int]]) -> int:
    return sum(len(v) for v in p)
