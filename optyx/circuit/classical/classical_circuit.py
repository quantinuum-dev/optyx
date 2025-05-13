from optyx.diagram.channel import Discard, bit, mode

DiscardBit = lambda n: Discard(bit**n)
DiscardMode = lambda n: Discard(mode**n)