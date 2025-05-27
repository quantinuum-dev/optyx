from optyx.diagram.channel import (
    Encode,
    mode,
    Channel,
    qmode,
    bit,
    qubit,
    Discard,
    Measure,
    Circuit
)
from optyx.diagram.zw import (
    Create as CreateSingle,
    Select as SelectSingle
)
from optyx.diagram.optyx import PhotonThresholdDetector
from optyx.diagram.lo import (
    Gate as GateSingle,
    Phase as PhaseSingle,
    BBS as BBSSingle,
    TBS as TBSSingle,
    MZI as MZISingle,
    ansatz as ansatz_single,
    BS_hadamard as BS_hadamard_single,
)
from optyx.diagram import optyx, zx
from optyx.diagram.optyx import dual_rail, Mode, Swap, Bit

from optyx.circuit.classical import ClassicalFunction, DiscardMode
import numpy as np


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
            CreateSingle(*n_photons)
        )


class PhotonThresholdMeasurement(Channel):
    """
    Ideal photon-number non-resolving detector
    from mode to bit from qmode to bit.
    Detects whether one or more photons are present.
    """

    def __init__(self):
        super().__init__(
            "PhotonThresholdMeasurement",
            PhotonThresholdDetector(),
            cod=bit
        )

class NumberResolvingMeasurement(Measure):
    """
    Number-resolving measurement of :math:`n` photons.
    """

    def __init__(self, n):
        super().__init__(qmode**n)


class DiscardQModes(Discard):
    """
    Discard :math:`n` qmodes.
    """

    def __init__(self, n):
        super().__init__(qmode**n)


class Select(Channel):
    def __init__(self, *n_photons: int):
        super().__init__(
            f"Select({n_photons})",
            SelectSingle(*n_photons)
        )


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


class BitFlipError(Channel):
    """
    Represents a bit-flip error channel.
    """

    def __init__(self, prob):
        x_error = zx.X(1, 2) >> zx.Id(1) @ zx.ZBox(
            1, 1, np.sqrt((1 - prob) / prob)
        ) @ zx.Scalar(np.sqrt(prob * 2))
        super().__init__(
            name=f"BitFlipError({prob})",
            kraus=x_error,
            dom=qubit,
            cod=qubit,
            env=optyx.bit,
        )

    def dagger(self):
        return self


class DephasingError(Channel):
    """
    Represents a quantum dephasing error channel.
    """
    def __init__(self, prob):
        z_error = (
            zx.H
            >> zx.X(1, 2)
            >> zx.H
            @ zx.ZBox(1, 1, np.sqrt((1 - prob) / prob))
            @ zx.Scalar(np.sqrt(prob * 2))
        )
        super().__init__(
            name=f"DephasingError({prob})",
            kraus=z_error,
            dom=qubit,
            cod=qubit,
            env=optyx.bit,
        )

    def dagger(self):
        return self


class DualRail(Channel):
    """
    Represents a dual-rail quantum channel
    encoding a specified number of qubit registers.
    """
    def __init__(self, n_qubits):
        super().__init__(
            f"DualRail({n_qubits})",
            dual_rail(n_qubits)
        )


Hadamard = HadamardBS


class PhaseShift(Channel):
    """
    Represents a phase shift operation in dual-rail encoding.
    """
    from optyx.diagram.lo import Phase as PhaseSingle
    def __init__(self, phase):
        super().__init__(
            f"PhaseShift({phase})",
            Mode(1) @ PhaseSingle(phase)
        )


class ZMeasurement(Circuit):
    def __new__(cls, alpha):
        """
        ZMeasurement circuit that performs a measurement in the Z basis
        after applying a phase shift of alpha.
        """
        return (
            qmode @ Phase(alpha) >>
            HadamardBS >>
            NumberResolvingMeasurement(2) >>
            DiscardMode(1) @ mode
        )


class XMeasurement(Circuit):
    def __new__(cls, alpha):
        """
        XMeasurement circuit that performs a measurement in the X basis
        after applying a Hadamard beam splitter.
        """
        return (
            HadamardBS >>
            ZMeasurement(alpha)
        )


class FusionTypeI(Circuit):
    def __new__(cls):
        kraus_map_fusion_I = (
            Mode(1) @ Swap(Mode(1), Mode(1)) @ Mode(1) >>
            Mode(1) @ BS_hadamard_single @ Mode(1) >>
            Mode(2) @ Swap(Mode(1), Mode(1)) >>
            Mode(1) @ Swap(Mode(1), Mode(1)) @ Mode(1)
        )

        fusion_I = Channel(
            "Fusion I", kraus_map_fusion_I
        )

        def fusion_I_function(x):
            """
            A classical function that returns two bits based on an input x,
            based on the classical logical for the Fusion type I circuit.
            """
            a = x[0]
            b = x[1]
            s = (a % 2) ^ (b % 2)
            k = int(s*b + (1-s)*(1 - (a + b)/2))%2
            return [s, k]

        classical_function_I = ClassicalFunction(
            fusion_I_function,
            Mode(2),
            Bit(2)
        )

        return (
            fusion_I >>
            qmode**2 @ NumberResolvingMeasurement(2) >>
            qmode**2 @ classical_function_I
        )


class FusionTypeII(Circuit):
    def __new__(cls):
        fusion_II = Channel(
            "Fusion II",
            (
                BS_hadamard_single @ BS_hadamard_single >>
                Mode(1) @ Swap(Mode(1), Mode(1)) @ Mode(1) >>
                Mode(1) @ BS_hadamard_single @ Mode(1) >>
                Mode(2) @ Swap(Mode(1), Mode(1)) >>
                Mode(1) @ Swap(Mode(1), Mode(1)) @ Mode(1) >>
                BS_hadamard_single @ Mode(2)
            )
        )

        def fusion_II_function(x):
            """
            A classical function that returns two bits based on an input x,
            based on the classical logical for the Fusion type II circuit.
            """
            a = x[0]
            b = x[1]
            d = x[3]
            s = (a % 2) ^ (b % 2)
            k = int(s*(b + d) + (1-s)*(1 - (a + b)/2))%2
            return [s, k]

        classical_function_II = ClassicalFunction(
            fusion_II_function,
            Mode(4),
            Bit(2)
        )

        return (
            fusion_II >>
            NumberResolvingMeasurement(4) >>
            classical_function_II
        )
