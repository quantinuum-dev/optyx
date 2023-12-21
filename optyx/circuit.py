"""
Linear optical circuits

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Gate
    Phase
    BBS
    TBS
    MZI

.. admonition:: Functions

    .. autosummary::
        :template: function.rst
        :nosignatures:
        :toctree:

        npperm
        occupation_numbers

Example
-------
We can differentiate the expectation values of optical circuits.

>>> from sympy.abc import psi
>>> circuit = BS >> Phase(psi) @ Id(1) >> BS.dagger()
>>> state = Create(2, 0) >> circuit
>>> observable = num_op @ Id(1)
>>> expectation = state >> observable >> state.dagger()
>>> assert np.allclose(
...     expectation.subs((psi, 1/2)).eval().array, np.array([0.]))
>>> assert np.allclose(
...     expectation.subs((psi, 1/4)).eval().array, np.array([1.]))
>>> assert np.allclose(
...     expectation.grad(psi).subs((psi, 1/2)).eval().array, np.array([0.]))
>>> assert np.allclose(
...     expectation.grad(psi).subs((psi, 1/4)).eval().array,
...     np.array([-2*np.pi]))
>>> assert np.allclose(
...     expectation.grad(psi).grad(psi).subs((psi, 1/4)).eval().array,
...     np.array([0.]))
"""

import numpy as np
from sympy import Expr 
import sympy as sp

from optyx.qpath import Box, Id, Matrix, Scalar
from optyx.qpath import Create, Select, Split, Merge


class Gate(Box):
    """
    Unitary gate in a diagram.

    Parameters:
        array : Unitary matrix (not checked on initialisation)
        dom : int
        cod : int
        name : str

    Example
    -------
    >>> hbs_array = (1 / 2) ** (1 / 2) * np.array([[1, 1], [1, -1]])
    >>> HBS = Gate(hbs_array, 2, 2, "HBS")
    >>> assert np.allclose((HBS.dagger() >> HBS).eval(2).array, Id(2).eval(2).array)
    """
    def __init__(self, array, dom: int, cod: int, name: str, is_dagger=False):
        self.array = array
        super().__init__(
            f"{name}" + ".dagger()" if is_dagger else "",
            dom,
            cod,
            is_dagger=is_dagger,
        )

    def to_path(self, dtype=complex):
        result = Matrix[dtype](self.array, len(self.dom), len(self.cod))
        return result.dagger() if self.is_dagger else result

    def dagger(self):
        return Gate(
            self.array,
            self.dom,
            self.cod,
            self.name,
            is_dagger=not self.is_dagger,
        )


class Phase(Box):
    """
    Phase shift with angle parameter between 0 and 1

    Parameters:
        angle : Phase parameter between 0 and 1

    Example
    -------
    >>> Phase(1/2).eval(1).array.round(3)
    array([[-1.+0.j]])
    >>> from sympy.abc import psi
    >>> derivative = Phase(psi).grad(psi).subs((psi, 0.5)).eval(2).array
    >>> assert np.allclose(derivative, 4 * np.pi * 1j)
    """

    def __init__(self, angle: float):
        self.angle = angle
        super().__init__(f"Phase({angle})", 1, 1, data=angle)

    def to_path(self, dtype=complex):
        backend = sp if dtype is Expr else np
        exp = backend.exp(2 * backend.pi * 1j * self.angle)
        return Matrix[dtype]([exp], 1, 1)

    def dagger(self):
        return Phase(-self.angle)

    def grad(self, var):
        if var not in self.free_symbols:
            return self.sum_factory((), self.dom, self.cod)
        s = 2j * np.pi * self.angle.diff(var)
        num_op = (
            Split()
            >> Id(1) @ (Select() >> Create())
            >> Merge()
        )
        d = Scalar(s) @ (self >> num_op)
        return d

    def lambdify(self, *symbols, **kwargs):
        from sympy import lambdify

        return lambda *xs: type(self)(
            lambdify(symbols, self.angle, **kwargs)(*xs)
        )


class BBS(Box):
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

    >>> diagram = Create(1, 1) >> BS
    >>> assert np.isclose((diagram >> Select(0, 2)).prob().array, 0.5)
    >>> assert np.isclose((diagram >> Select(2, 0)).prob().array, 0.5)
    >>> assert np.isclose((diagram >> Select(1, 1)).prob().array, 0)

    Check the dagger:

    >>> y = BBS(0.4)
    >>> assert np.allclose((y >> y.dagger()).eval(2).array, Id(2).eval(2).array)
    >>> comp = (y @ y >> Id(1) @ y @ Id(1)) >> (y @ y >> Id(1) @ y @ Id(1)
    ...   ).dagger()
    >>> assert np.allclose(comp.eval(2).array, Id(4).eval(2).array)
    """
    def __init__(self, bias):
        self.bias = bias
        super().__init__('BBS({})'.format(bias), 2, 2)

    def __repr__(self):
        return 'BS' if self.bias == 0 else super().__repr__()

    def to_path(self, dtype=complex):
        backend = sp if dtype is Expr else np
        sin = backend.sin((0.25 + self.bias) * backend.pi)
        cos = backend.cos((0.25 + self.bias) * backend.pi)
        array = [sin, 1j * cos, 1j * cos, sin]
        return Matrix[dtype](array, len(self.dom), len(self.cod))

    def dagger(self):
        return BBS(0.5 - self.bias)


class TBS(Box):
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
    >>> tbs = lambda x: BS >> Phase(x) @ Id(1) >> BS
    >>> assert np.allclose(TBS(0.15).to_path().array * TBS(0.15).global_phase,
    ...                    tbs(0.15).to_path().array)
    >>> assert np.allclose((TBS(0.25) >> TBS(0.25).dagger()).to_path().array,
    ...                    Id(2).to_path().array)
    >>> assert (TBS(0.25).dagger().global_phase ==\\
    ...         np.conjugate(TBS(0.25).global_phase))
    """
    def __init__(self, theta, is_dagger=False):
        self.theta = theta
        name = 'TBS({})'.format(theta)
        super().__init__(name, 2, 2, is_dagger=is_dagger)

    @property
    def global_phase(self):
        if self.is_dagger:
            return - 1j * np.exp(- 1j * self.theta * np.pi)
        else:
            return 1j * np.exp(1j * self.theta * np.pi)

    def to_path(self, dtype=complex):
        backend = sp if dtype is Expr else np
        sin = backend.sin(self.theta * backend.pi)
        cos = backend.cos(self.theta * backend.pi)
        array = [sin, cos, cos, -sin]
        matrix = Matrix[dtype](array, len(self.dom), len(self.cod))
        matrix = matrix.dagger() if self.is_dagger else matrix
        return matrix

    def dagger(self):
        return TBS(self.theta, is_dagger=not self.is_dagger)


class MZI(Box):
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
    >>> assert np.allclose(MZI(0.28, 0).to_path().array, TBS(0.28).to_path().array)
    >>> assert np.isclose(MZI(0.28, 0.3).global_phase, TBS(0.28).global_phase)
    >>> assert np.isclose(MZI(0.12, 0.3).global_phase.conjugate(),
    ...                   MZI(0.12, 0.3).dagger().global_phase)
    >>> mach = lambda x, y: TBS(x) >> Phase(y) @ Id(1)
    >>> assert np.allclose(MZI(0.28, 0.9).to_path().array, mach(0.28, 0.9).to_path().array)
    >>> assert np.allclose((MZI(0.28, 0.34) >> MZI(0.28, 0.34).dagger()).to_path().array,
    ...                    Id(2).to_path().array)
    """
    def __init__(self, theta, phi, is_dagger=False):
        self.theta, self.phi = theta, phi
        super().__init__('MZI', 2, 2, is_dagger=is_dagger)

    @property
    def global_phase(self):
        if not self.is_dagger:
            return 1j * np.exp(1j * self.theta * np.pi)
        else:
            return - 1j * np.exp(- 1j * self.theta * np.pi)

    def to_path(self, dtype=complex):
        backend = sp if dtype is Expr else np
        cos = backend.cos(backend.pi * self.theta)
        sin = backend.sin(backend.pi * self.theta)
        exp = backend.exp(1j * 2 * backend.pi * self.phi)
        array = np.array([exp * sin, cos, exp * cos, -sin])
        matrix = Matrix[dtype](array, len(self.dom), len(self.cod))
        matrix = matrix.dagger() if self.is_dagger else matrix
        return matrix

    def dagger(self):
        return MZI(self.theta, self.phi, is_dagger=not self.is_dagger)


bs_array = (1 / 2) ** (1 / 2) * np.array([[1j, 1], [1, 1j]])
BS = Gate(bs_array, 2, 2, "BS")
num_op = (Split() >> Id(1) @ (Select() >> Create()) >> Merge())