from optyx.diagram.channel import (
    Measure,
    Encode,
    qubit,
    bit,
    Discard
)

MeasureQubit = lambda n: Measure(qubit**n)
DiscardQubit = lambda n: Discard(qubit**n)
EncodeBit = lambda n: Encode(bit**n)



