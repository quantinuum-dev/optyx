from optyx.diagram.channel import (
    Measure,
    Encode,
    qubit,
    bit,
    Discard
)

class MeasureQubits(Measure):
    """
    Ideal qubit measurement (in computational basis) from qubit to bit.
    """

    def __init__(self, n):
        super().__init__(
            qubit**n
        )


class DiscardQubits(Discard):
    """
    Discard :math:`n` qubits.
    """

    def __init__(self, n):
        super().__init__(
            qubit**n
        )


class EncodeBits(Encode):
    """
    Encode :math:`n` bits into :math:`n` qubits.
    """

    def __init__(self, n):
        super().__init__(
            bit**n
        )
