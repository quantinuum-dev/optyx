"""Provides a parser for the graph6 format

http://users.cecs.anu.edu.au/%7Ebdm/data/formats.txt
"""

from optyx.graphs import Graph


def read_graph6(lines: list[bytes]) -> list[Graph]:
    """Convert each line into a Graph from the graph6 format

    Stolen from networkx
    https://networkx.org/documentation/stable/_modules/networkx/readwrite/graph6.html
    """
    glist = []
    for line in lines:
        line = line.strip()
        if len(line) == 0:
            continue
        glist.append(from_graph6_bytes(line))
    return glist


def from_graph6_bytes(bytes_in):
    """Read a simple undirected graph in graph6 format from bytes.

    Stolen from networkx
    https://networkx.org/documentation/stable/_modules/networkx/readwrite/graph6.html
    """

    def bits():
        """Returns sequence of individual bits from 6-bit-per-value
        list of data values."""
        for d in data:
            for i in [5, 4, 3, 2, 1, 0]:
                yield (d >> i) & 1

    if bytes_in.startswith(b">>graph6<<"):
        bytes_in = bytes_in[10:]

    data = [c - 63 for c in bytes_in]
    if any(c > 63 for c in data):
        raise ValueError("each input character must be in range(63, 127)")

    n, data = data_to_n(data)
    nd = (n * (n - 1) // 2 + 5) // 6
    if len(data) != nd:
        expected = n * (n - 1) // 2
        raise ValueError(
            f"Expected {expected} bits but got {len(data) * 6} in graph6"
        )

    g = Graph({})
    for (i, j), b in zip(
        ((i, j) for j in range(1, n) for i in range(j)), bits()
    ):
        if b:
            g.add_edge(i, j)

    return g


def data_to_n(data):
    """Read initial one-, four- or eight-unit value from graph6
    integer sequence.

    Return (value, rest of seq.)

    Stolen from networkx
    https://networkx.org/documentation/stable/_modules/networkx/readwrite/graph6.html
    """
    if data[0] <= 62:
        return data[0], data[1:]
    if data[1] <= 62:
        return (data[1] << 12) + (data[2] << 6) + data[3], data[4:]
    return (
        (data[2] << 30)
        + (data[3] << 24)
        + (data[4] << 18)
        + (data[5] << 12)
        + (data[6] << 6)
        + data[7],
        data[8:],
    )
