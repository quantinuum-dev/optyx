from optyx.diagram.channel import Encode, mode, Channel
from optyx.diagram.zw import Create

Encode = lambda internal_states=None: Encode(
    mode,
    internal_states=internal_states,
)

class Create(Channel):
    def __init__(self, *n_photons: int):
        super().__init__(
            f"Create({n_photons})",
            Create(*n_photons)
        )