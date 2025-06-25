import numpy as np
import sympy as sp
from sympy import Expr, lambdify, Symbol, Mul
from discopy.cat import rsubs
from typing import Set
from functools import cached_property
from abc import abstractmethod, ABC
from collections.abc import Iterable

from optyx.core import (
    channel,
    diagram,
    zw
)

from optyx.classical import ClassicalFunction, DiscardMode
from optyx._utils import matrix_to_zw


class Scalar(channel.Channel):
    def __init__(self, value):
        if not isinstance(value, (Symbol, Mul)):
            self.scalar = complex(value)
        else:
            self.scalar = value
        super().__init__(
            f"{value}",
            diagram.Scalar(value)
        )
        self.data = value

    def subs(self, *args):
        data = rsubs(self.scalar, *args)
        return Scalar(data)

    def grad(self, var, **params):
        """Gradient with respect to :code:`var`."""
        if var not in self.free_symbols:
            return self.sum_factory((), self.dom, self.cod)
        return Scalar(self.scalar.diff(var))

    def lambdify(self, *symbols, **kwargs):
        from sympy import lambdify

        return lambda *xs: type(self)(
            lambdify(symbols, self.scalar, **kwargs)(*xs)
        )


class EncodePhotonic(channel.Encode):
    """
    Encode :math:`n` modes into :math:`n` qmodes.
    """
    def __init__(self, n):
        super().__init__(channel.mode**n) # pragma: no cover


class DiscardPhotonic(channel.Discard):
    """
    Discard :math:`n` qmodes.
    """

    def __init__(self, n):
        super().__init__(channel.qmode**n) # pragma: no cover


class PhotonThresholdMeasurement(channel.Channel):
    """
    Ideal photon-number non-resolving detector
    from mode to bit from qmode to bit.
    Detects whether one or more photons are present.
    """

    def __init__(self):
        super().__init__(
            "PhotonThresholdMeasurement",
            diagram.PhotonThresholdDetector(),
            cod=channel.bit
        )


class NumberResolvingMeasurement(channel.Measure):
    """
    Number-resolving measurement of :math:`n` photons.
    """

    def __init__(self, n):
        super().__init__(channel.qmode**n) # pragma: no cover


class Create(channel.Channel):
    """
    Create a quantum channel that initializes
    a specified number of photons
    in a specified number of channel.qmodes.
    """
    def __init__(self, *photons: int,
                 internal_states: tuple[list[int]] = None):
        self.photons = photons
        super().__init__(
            f"Create({photons})",
            zw.Create(*photons, internal_states=internal_states)
        )


class AbstractGate(channel.Channel, ABC):

    def __init__(
        self,
        dom: int,
        cod: int,
        name: str,
        data=None
    ):

        self.dtype = Expr if self._contains_expr(data) else complex
        super().__init__(
            name,
            self._normal_form(dom, cod)
        )
        self.data = data

    def _normal_form(self, dom, cod):
        return matrix_to_zw(self.array.reshape(dom, cod))

    @cached_property
    def array(self):
        return np.asarray(self._compute_array())

    @abstractmethod
    def _compute_array(self):
        pass # pragma: no cover

    def _contains_expr(self, obj):
        if isinstance(obj, Expr):
            return True
        if isinstance(obj, Iterable) and not isinstance(obj, (str, bytes)):
            return any(self._contains_expr(item) for item in obj)
        return False


class Gate(AbstractGate):
    """
    Unitary LO gate in a diagram.

    Parameters:
        array : Unitary matrix (not checked on initialisation)
        dom : int
        cod : int
        name : str

    Example
    -------
    >>> hbs_array = (1 / 2) ** (1 / 2) * np.array([[1, 1], [1, -1]])
    >>> HBS = Gate(hbs_array, 2, 2, "HBS")
    >>> assert np.allclose(
    ...     (HBS.dagger() >> HBS).to_path().eval(2).array,
    ...                 diagram.Id(diagram.Mode(2)).to_path().eval(2).array)
    """

    # need to make it a Circuit?
    # this should also take a perceval matrix
    # as an input
    def __init__(
        self,
        matrix,
        dom: int,
        cod: int,
        name: str,
        data=None
    ):
        self._matrix = np.asanyarray(matrix)
        super().__init__(dom, cod, name, data=data)

    def _compute_array(self):
        return self._matrix

    def dagger(self):
        return Gate(
            np.conjugate(self.array.T),
            len(self.cod),
            len(self.dom),
            self.name
        )

    def conjugate(self):
        return Gate(
            np.conjugate(self.array),
            len(self.dom),
            len(self.cod),
            self.name
        ) # pragma: no cover


class Phase(AbstractGate):
    """
    Phase shift with angle parameter between 0 and 1

    Parameters:
        angle : Phase parameter between 0 and 1

    Example
    -------
    >>> Phase(1/2).to_path().eval(1).array.round(3)
    array([[-1.+0.j]])
    >>> from sympy.abc import psi
    >>> derivative = Phase(psi).grad(psi).subs((psi,
    ...                     0.5)).to_path().eval(2).array
    >>> assert np.allclose(derivative, 4 * np.pi * 1j)
    """

    def __init__(self, angle: float):
        self.angle = angle
        super().__init__(
            1, 1,
            f"Phase({angle})",
            data=angle
        )

    def _compute_array(self):
        backend = sp if self.dtype is Expr else np
        return [backend.exp(2 * np.pi * 1j * self.angle)]

    def grad(self, var):
        """Gradient with respect to :code:`var`."""
        if var not in self.free_symbols:
            return self.sum_factory((), self.dom, self.cod)
        s = 2j * np.pi * self.angle.diff(var)
        d = Scalar(s) @ (self >> NumOp())
        return d

    def lambdify(self, *symbols, **kwargs):
        return lambda *xs: type(self)(
            lambdify(symbols, self.angle, **kwargs)(*xs)
        )

    def dagger(self):
        return Phase(-self.angle)

    def conjugate(self):
        return Phase(-self.angle)


class NumOp(channel.Channel):
    def __init__(self):
        super().__init__(
            "NumOp",
            (
                zw.Split(2) >>
                zw.Id(1) @ (zw.Select() >> zw.Create()) >>
                zw.Merge(2)
            )
        )

class BBS(AbstractGate):
    """
    Beam splitter with a bias.

    Corresponds to :py:class:`Matrix`
    :math:`\\begin{pmatrix}
    \\tt{sin}((0.25 + bias)\\pi)
    & i \\tt{cos}((0.25 + bias)\\pi) \\\\
    i \\tt{cos}((0.25 + bias)\\pi)
    & \\tt{sin}((0.25 + bias)\\pi) \\end{pmatrix}`.

    Parameters
    ----------
    bias : float
        Bias from standard 50/50 beam splitter, parameter between 0 and 1.

    Example
    -------
    The standard beam splitter is:

    >>> BS = BBS(0)

    We can check the Hong-Ou-Mandel effect:

    >>> from optyx.classical import Select
    >>> d = Create(1, 1) >> BS
    >>> assert np.isclose((d >> Select(0, 2)).to_path().prob().array,
    ...                                                                0.5)
    >>> assert np.isclose((d >> Select(2, 0)).to_path().prob().array,
    ...                                                                0.5)
    >>> assert np.isclose((d >> Select(1, 1)).to_path().prob().array,
    ...                                                                  0)

    Check the dagger:

    >>> y = BBS(0.4)
    >>> x = y.get_kraus()
    >>> assert np.allclose((
    ...     y >> y.dagger()).to_path().eval(2).array,
    ...             diagram.Id(diagram.Mode(2)).to_path().eval(2).array)
    >>> comp = (x @ x >> diagram.Id(diagram.Mode(1)) @ x @ diagram.Id(diagram.Mode(1))) >> \\
    ...             (x @ x >> diagram.Id(diagram.Mode(1)) @ x @ diagram.Id(diagram.Mode(1))).dagger()
    >>> assert np.allclose(comp.to_path().eval(2).array,
    ...                     diagram.Id(diagram.Mode(4)).to_path().eval(2).array)

    """

    def __init__(self, bias, is_conj=False):
        self.bias = bias
        self.is_conj = is_conj
        super().__init__(
            2, 2,
            f"BBS({bias})",
            data=bias
        )

    def _compute_array(self):
        backend = sp if self.dtype is Expr else np
        sin = backend.sin((0.25 + self.bias) * np.pi)
        cos = backend.cos((0.25 + self.bias) * np.pi)
        if self.is_conj:
            array = [-1j * cos, sin, sin, -1j * cos]
        else:
            array = [1j * cos, sin, sin, 1j * cos]
        return np.array(array)

    def lambdify(self, *symbols, **kwargs):
        return lambda *xs: type(self)(
            lambdify(symbols, self.bias, **kwargs)(*xs)
        )

    def dagger(self):
        return BBS(0.5 - self.bias)

    def conjugate(self):
        return BBS(self.bias, not self.is_conj)


class TBS(AbstractGate):
    """
    Tunable Beam Splitter.

    Corresponds to :py:class:`Matrix`
    :math:`\\begin{pmatrix}
    \\tt{sin}(\\theta \\, \\pi)
    & \\tt{cos}(\\theta \\, \\pi) \\\\
    \\tt{cos}(\\theta \\, \\pi) & - \\tt{sin}(\\theta \\, \\pi)
    \\end{pmatrix}`.

    Parameters
    ----------
    theta : float
        TBS parameter ranging from 0 to 1.

    Example
    -------
    >>> BS = BBS(0)
    >>> tbs = lambda x: BS >> channel.Diagram.id(channel.qmode) @ Phase(x) >> BS
    >>> assert np.allclose(
    ...     TBS(0.15).to_path().array, tbs(0.15).to_path().array)
    >>> assert np.allclose(
    ...     (TBS(0.25) >> TBS(0.25).dagger()).to_path().array,
    ...     channel.Diagram.id(channel.qmode**2).to_path().array)
    >>> assert (TBS(0.25).dagger().global_phase ==\\
    ...         np.conjugate(TBS(0.25).global_phase))

    """

    def __init__(self,
                 theta,
                 is_dagger=False,
                 is_conj=False):
        self.theta = theta
        self.is_dagger = is_dagger
        self.is_conj = is_conj
        super().__init__(
            2, 2,
            f"TBS({theta})",
            data=theta
        )
        self.is_dagger = is_dagger

    @cached_property
    def global_phase(self):
        backend = sp if self.dtype is Expr else np
        return (
            -1j * backend.exp(-1j * self.theta * backend.pi)
            if self.is_dagger
            else 1j * backend.exp(1j * self.theta * backend.pi)
        )

    def _compute_array(self):
        backend = sp if self.dtype is Expr else np
        sin = backend.sin(self.theta * backend.pi)
        cos = backend.cos(self.theta * backend.pi)
        array = np.array([sin, cos, cos, -sin])
        return np.conjugate(array * self.global_phase) if \
            self.is_conj else array * self.global_phase

    def lambdify(self, *symbols, **kwargs):
        return lambda *xs: type(self)(
            lambdify(symbols, self.theta, **kwargs)(*xs),
            is_dagger=self.is_dagger,
        )

    def _decomp(self):
        d = BS >> channel.qmode @ Phase(self.theta) >> BS
        return d.dagger() if self.is_dagger else d

    def grad(self, var):
        """Gradient with respect to :code:`var`."""
        if var not in self.free_symbols:
            return self.sum_factory((), self.dom, self.cod)
        return self._decomp().grad(var)

    def conjugate(self):
        return TBS(self.theta, self.is_dagger, not self.is_conj)

    def dagger(self):
        return TBS(self.theta, is_dagger=not self.is_dagger)


class MZI(AbstractGate):
    """
    Mach-Zender interferometer.

    Corresponds to :py:class:`Matrix`
    :math:`\\begin{pmatrix}
    e^{2\\pi i \\phi} \\tt{sin}(\\theta \\, \\pi)
    & \\tt{cos}(\\theta \\, \\pi) \\\\
    e^{2\\pi i \\phi} \\tt{cos}(\\theta \\, \\pi)
    & - \\tt{sin}(\\theta \\, \\pi) \\end{pmatrix}`.

    Parameters
    ----------
    theta: float
        Internal phase parameter, ranging from 0 to 1.
    phi: float
        External phase parameter, ranging from 0 to 1.

    Example
    -------
    >>> assert np.allclose(
    ...     MZI(0.28, 0).to_path().array,
    ...     TBS(0.28).to_path().array)
    >>> assert np.isclose(
    ...    MZI(0.28, 0.3).global_phase,
    ...    TBS(0.28).global_phase)
    >>> assert np.isclose(
    ...     MZI(0.12, 0.3).global_phase.conjugate(),
    ...     MZI(0.12, 0.3).dagger().global_phase)
    >>> mach = lambda x, y: TBS(x) >> Phase(y) @ channel.Diagram.id(channel.qmode)
    >>> assert np.allclose(
    ...     MZI(0.28, 0.9).to_path().array,
    ...     mach(0.28, 0.9).to_path().array)
    >>> assert np.allclose(
    ...     (MZI(0.28, 0.34) >> MZI(0.28, 0.34).dagger()).to_path().array,
    ...     channel.Diagram.id(channel.qmode**2).to_path().array)

    """

    def __init__(self,
                 theta,
                 phi,
                 is_dagger=False,
                 is_conj=False):
        self.theta, self.phi = theta, phi
        self.is_dagger = is_dagger
        self.is_conj = is_conj
        super().__init__(
            2, 2,
            f"MZI({theta}, {phi})",
            data=(theta, phi)
        )
        self.is_dagger = is_dagger

    @cached_property
    def global_phase(self):
        backend = sp if self.dtype is Expr else np
        return (
            -1j * backend.exp(-1j * self.theta * backend.pi)
            if self.is_dagger
            else 1j * backend.exp(1j * self.theta * backend.pi)
        )

    def _compute_array(self):
        backend = sp if self.dtype is Expr else np
        cos = backend.cos(backend.pi * self.theta)
        sin = backend.sin(backend.pi * self.theta)
        exp = backend.exp(1j * 2 * backend.pi * self.phi)
        array = np.array([exp * sin, cos, exp * cos, -sin])
        return np.conjugate(array * self.global_phase) if \
            self.is_conj else array * self.global_phase

    def lambdify(self, *symbols, **kwargs):
        return lambda *xs: type(self)(
            *lambdify(symbols, [self.theta, self.phi], **kwargs)(*xs),
            is_dagger=self.is_dagger,
        )

    def _decomp(self):
        x, y = self.theta, self.phi
        d = BS >> channel.qmode @ Phase(x) >> BS >> Phase(y) @ channel.qmode
        return d.dagger() if self.is_dagger else d

    def grad(self, var):
        """Gradient with respect to :code:`var`."""
        if var not in self.free_symbols:
            return self.sum_factory((), self.dom, self.cod)
        return self._decomp().grad(var)

    def dagger(self):
        return MZI(self.theta, self.phi, is_dagger=not self.is_dagger)

    def conjugate(self):
        return MZI(self.theta, self.phi, self.is_dagger, not self.is_conj)


def ansatz(width, depth):
    """
    Returns a universal interferometer given width, depth and parameters x,
    based on https://arxiv.org/abs/1603.08788.

    Parameters
    ----------
    width: int
        Number of modes in the ansatz.
    depth: int
        Number of layers in the ansatz.

    Example
    -------
    >>> ansatz(6, 4).draw(path='docs/_static/ansatz6_4.png')
    >>> ansatz(5, 4).draw(path='docs/_static/ansatz5_4.png')

    .. image:: /_static/ansatz6_4.png
        :align: center

    .. image:: /_static/ansatz5_4.png
        :align: center
    """

    def p(i, j):
        return sp.Symbol(f"a_{i}_{j}"), sp.Symbol(f"b_{i}_{j}")

    d = channel.Diagram.id(channel.qmode**width)
    for i in range(depth):
        n_mzi = (width - 1) // 2 if i % 2 else width // 2
        left = channel.qmode**(i % 2)
        right = channel.qmode**(width - (i % 2) - 2 * n_mzi)
        d >>= left @ channel.Diagram.tensor(*[MZI(*p(i, j)) for j in range(n_mzi)]) @ right

    return d


class HadamardBS(Gate):
    def __init__(self):
        matrix = np.sqrt(1 / 2) * np.array([[1, 1], [1, -1]])
        super().__init__(
            matrix, 2, 2, "HadamardBS"
        )


class DualRail(channel.Channel):
    """
    Represents a dual-rail quantum channel
    encoding a specified number of qubit registers.
    """
    def __init__(self, n_qubits, internal_states=None):
        super().__init__(
            f"DualRail({n_qubits})",
            diagram.dual_rail(n_qubits, internal_states=internal_states)
        )


class PhaseShiftDR(channel.Channel):
    """
    Represents a phase shift operation in dual-rail encoding.
    """

    def __init__(self, phase):
        super().__init__(
            f"PhaseShift({phase})",
            diagram.Mode(1) @ Phase(phase).get_kraus()
        )


class ZMeasurementDR(channel.Diagram):
    def __new__(cls, alpha):
        """
        ZMeasurement circuit that performs a measurement in the Z basis
        after applying a phase shift of alpha.
        """
        return (
            channel.qmode @ Phase(alpha) >>
            HadamardBS >>
            NumberResolvingMeasurement(2) >>
            DiscardMode(1) @ channel.mode
        )


class XMeasurementDR(channel.Diagram):
    def __new__(cls, alpha):
        """
        XMeasurement circuit that performs a measurement in the X basis
        after applying a Hadamard beam splitter.
        """
        return (
            HadamardBS >>
            ZMeasurementDR(alpha)
        )


class FusionTypeI(channel.Diagram):
    def __new__(cls):
        kraus_map_fusion_I = (
            diagram.Mode(1) @ diagram.Swap(diagram.Mode(1),
                                       diagram.Mode(1)) @ diagram.Mode(1) >>
            diagram.Mode(1) @ HadamardBS().get_kraus() @ diagram.Mode(1) >>
            diagram.Mode(2) @ diagram.Swap(diagram.Mode(1),
                                       diagram.Mode(1)) >>
            diagram.Mode(1) @ diagram.Swap(diagram.Mode(1),
                                       diagram.Mode(1)) @ diagram.Mode(1)
        )

        fusion_I = channel.Channel(
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
            diagram.Mode(2),
            diagram.Bit(2)
        )

        return (
            fusion_I >>
            channel.qmode**2 @ NumberResolvingMeasurement(2) >>
            channel.qmode**2 @ classical_function_I
        )


class Swap(channel.Swap):
    def __init__(self, left, right):
        super().__init__(channel.qmode**left, channel.qmode**right)


class FusionTypeII(channel.Diagram):
    def __new__(cls):
        fusion_II = channel.Channel(
            "Fusion II",
            (
                HadamardBS().get_kraus() @ HadamardBS().get_kraus() >>
                diagram.Mode(1) @ diagram.Swap(diagram.Mode(1),
                                           diagram.Mode(1)) @ diagram.Mode(1) >>
                diagram.Mode(1) @ HadamardBS().get_kraus() @ diagram.Mode(1) >>
                diagram.Mode(2) @ diagram.Swap(diagram.Mode(1),
                                           diagram.Mode(1)) >>
                diagram.Mode(1) @ diagram.Swap(diagram.Mode(1),
                                           diagram.Mode(1)) @ diagram.Mode(1) >>
                HadamardBS().get_kraus() @ diagram.Mode(2)
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
            diagram.Mode(4),
            diagram.Bit(2)
        )

        return (
            fusion_II >>
            NumberResolvingMeasurement(4) >>
            classical_function_II
        )

BS = BBS(0)

def Id(n):
    return channel.Diagram.id(n) if \
          isinstance(n, channel.Ty) else channel.Diagram.id(channel.qmode**n)