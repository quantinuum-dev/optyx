""" Stage 3 Compiler

Compiles a fusion network for a linear resource state into instructions for a
single emitter multiple measurement device
"""

from optyx.compiler import PartialOrder
from . import (
    FusionNetworkSE,
    PSMInstruction,
    FusionOp,
    MeasureOp,
    NextNodeOp,
)


def compile_single_emitter_multi_measurement(
    fp: FusionNetworkSE, node_past: PartialOrder
) -> list[PSMInstruction]:
    """Compiles the fusion network into a series of instructions that can be
    executed on a single emitter/multi measurement machine"""

    c = get_creation_times(fp)
    m = get_measurement_times(fp, node_past, c)
    f = _get_fusion_photons(fp.fusions, fp.path, c)

    ins: list[PSMInstruction] = []

    photon = 0
    for v in fp.path:
        ins.append(NextNodeOp(v))

        # Number of photons in a given node
        num_fusions = len(_get_fused_neighbours(fp.fusions, v))

        for _ in range(num_fusions):
            photon += 1
            pair = f[photon]

            delay = max(0, pair - photon)
            ins.append(FusionOp(delay))

        # Calculate measurement delay
        photon += 1
        measurement = fp.measurements[v]
        delay = max(0, m[v] - c[v])
        ins.append(MeasureOp(delay, measurement))

    return ins


# Convert the fusions between nodes into fusions beteen specific photons
# There are more optimisations I could perform here to reduce the delay, but
# for now we will choose an arbitrary order for simplicity
#
# If photon 1 fuses with photon 5, then there will be two entries in the
# returned dictionary. 1: 5, and 5: 1
def _get_fusion_photons(
    fusions: list[tuple[int, int]], path: list[int], c: list[int]
) -> dict[int, int]:
    seen: dict[int, int] = {}
    fusion_photons: dict[int, int] = {}

    reverse_list = {v: i for i, v in enumerate(path)}

    for fusion in fusions:
        photon_num1 = seen.get(fusion[0], 0) + 1
        photon_num2 = seen.get(fusion[1], 0) + 1

        seen[fusion[0]] = photon_num1
        seen[fusion[1]] = photon_num2

        photon_index1 = c[reverse_list[fusion[0]]] - photon_num1
        photon_index2 = c[reverse_list[fusion[1]]] - photon_num2

        fusion_photons[photon_index1] = photon_index2
        fusion_photons[photon_index2] = photon_index1

    return fusion_photons


# Returns the number of fusion edges a node has
def _num_fusions(fusions: list[tuple[int, int]], node: int) -> int:
    return sum(node in fusion for fusion in fusions)


def get_creation_times(fp: FusionNetworkSE) -> list[int]:
    """Returns a list containing the creation times of the measurement photon
    of every node"""
    acc = 0
    c = []
    for node in fp.path:
        # One photon for each fusion, and one measurement photon
        acc += _num_fusions(fp.fusions, node) + 1
        c.append(acc)

    return c


def get_measurement_times(
    fp: FusionNetworkSE, order: PartialOrder, c: list[int]
) -> list[int]:
    """Returns a list containing the time the measurement photon of a given
    node can be measured"""

    m = [-1] * len(fp.path)

    # Recursively evaluate all the measurement times
    def get_measurement(node: int) -> int:
        if m[node] != -1:
            return m[node]

        past = order(node)
        # Don't want to recurse forever
        past.remove(node)

        if len(past) == 0:
            m[node] = c[node]
            return m[node]

        latest_past_measurement = 0
        if len(past) != 0:
            latest_past_measurement = max(get_measurement(u) for u in past) + 1

        m[node] = max(c[node], latest_past_measurement)
        return m[node]

    for v in fp.path:
        get_measurement(v)

    return m


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

        nbrs = _get_fused_neighbours(fusions, v)
        new_nbrs = set(nbrs) - seen
        seen = seen.union(set(nbrs))
        frontier = frontier.union(new_nbrs)

    return list(seen)


# Returns the nodes that are fused to the given node.
def _get_fused_neighbours(
    fusions: list[tuple[int, int]], node: int
) -> list[int]:
    fusion_nbrs = []

    for fusion in fusions:
        if fusion[0] == node:
            fusion_nbrs.append(fusion[1])
        elif fusion[1] == node:
            fusion_nbrs.append(fusion[0])

    return fusion_nbrs
