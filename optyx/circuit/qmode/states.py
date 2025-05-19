"""
This module provides functionality for creating and encoding quantum modes
and photons in a quantum circuit.
"""

from optyx.diagram.channel import Encode, mode, Channel
from optyx.diagram.zw import Create

class EncodeModes(Encode):
    """
    Encode :math:`n` modes into :math:`n` qmodes.
    """
    def __init__(self, n):
        super().__init__(mode**n)

class Create(Channel):
    """
    Create a quantum channel that initializes a specified number of photons
    in a specified number of qmodes.
    """
    def __init__(self, *n_photons: int):
        super().__init__(
            f"Create({n_photons})",
            Create(*n_photons)
        )