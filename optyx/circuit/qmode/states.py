from optyx.diagram.channel import Encode, mode, Channel
from optyx.diagram.zw import Create

EncodeMode = lambda n: Encode(mode**n)

class Create(Channel):
    def __init__(self, *n_photons: int):
        super().__init__(
            f"Create({n_photons})",
            Create(*n_photons)
        )