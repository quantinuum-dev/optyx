import networkx as nx
import pytest
from networkx.generators.random_graphs import gnp_random_graph

from optyx.compiler.x_fusions import (
    photon_bounded_min_trail_decomp,
    photon_bounded_min_trail_decomp_count,
)
import random
import math

random.seed(42)

@pytest.mark.parametrize("photon_length", range(3, 20, 3))
def test_photon_bounded_trail_decompositions(photon_length):
    for i in range(30):
        for r in range(1, math.floor((photon_length+1)/2)):
            g  = gnp_random_graph(10, 0.3)
            new_tc = photon_bounded_min_trail_decomp(g, photon_length, r)
            count = photon_bounded_min_trail_decomp_count(g, photon_length, r)

            assert len(new_tc) == count
