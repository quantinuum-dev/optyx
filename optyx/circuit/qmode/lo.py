"""
This module defines various quantum optical components as subclasses of the `Channel` class.
Each component represents a specific type of quantum gate or operation, and provides methods
for creating and manipulating these components.

Gates:
    - Gate: Represents a general quantum photonic unitary
             with a specified matrix representation.
    - Phase: Represents a phase shift operation with a given angle.
    - BBS: Represents a biased beam splitter with a specified bias.
    - TBS: Represents a tunable beam splitter with a specified angle.
    - MZI: Represents a Mach-Zehnder interferometer with specified parameters.

"""

from optyx.diagram.lo import (
    Gate as GateSingle,
    Phase as PhaseSingle,
    BBS as BBSSingle,
    TBS as TBSSingle,
    MZI as MZISingle,
    ansatz as ansatz_single,
    BS_hadamard as BS_hadamard_single,
)
from optyx.diagram.channel import Channel

import numpy as np

class Gate(Channel):
    def __init__(
        self,
        array,
        dom: int,
        cod: int,
        name: str,
        is_dagger = False
    ):
        super().__init__(
            name,
            GateSingle(array, dom, cod, is_dagger=is_dagger),
        )


    def dagger(self):
        return Gate(
            np.conjugate(self.array.T),
            len(self.cod),
            len(self.dom),
            self.name,
            is_dagger=not self.is_dagger,
        )


class Phase(Channel):
    def __init__(self, angle: float):
        super().__init__(
            f"Phase({angle})",
            PhaseSingle(angle)
        )


class BBS(Channel):
    def __init__(self, bias: float):
        super().__init__(
            f"BBS({bias})",
            BBSSingle(bias)
        )

    def dagger(self):
        return BBS(0.5 - self.bias)


class TBS(Channel):
    def __init__(self, theta: float, is_dagger=False):
        super().__init__(
            f"TBS({theta})",
            TBSSingle(theta, is_dagger=is_dagger)
        )

    def dagger(self):
        return TBS(self.theta, is_dagger=not self.is_dagger)


class MZI(Channel):
    def __init__(self, theta: float, phi: float, is_dagger=False):
        super().__init__(
            f"MZI({theta}, {phi})",
            MZISingle(theta, phi, is_dagger=is_dagger)
        )

    def dagger(self):
        return MZI(self.theta, self.phi, is_dagger=not self.is_dagger)


def ansatz(width, depth):
    return Channel(
        f"Ansatz({width}, {depth})",
        ansatz_single(width, depth)
    )


BS = BBS(0)

HadamardBS = Channel(
    "HadamardBS",
    BS_hadamard_single
)