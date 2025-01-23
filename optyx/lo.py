"""

Overview
--------

This module can be used to build arbitrary
linear optical unitary circuits using biased/tunable
beam splitters, phase shifters, and Mach-Zender interferometers.
The :class:`lo` generators have an underlying
matrix representation in :class:`path` [FC23]_,
which allows to evaluate the amplitudes of circuits
by computing permanents.

Generators and diagrams
------------------------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Gate
    Phase
    BBS
    TBS
    MZI

Functions
----------

.. autosummary::
    :template: function.rst
    :nosignatures:
    :toctree:

    ansatz

Examples of usage
------------------

Let us check if a beam splitter showcases a valid Hang-Ou-Mandel effect:

>>> BS = BBS(0)
>>> diagram = Create(1, 1) >> BS
>>> assert np.isclose((diagram >> Select(0, 2)).to_path().prob().array, 0.5)
>>> diagram.draw(path='docs/_static/BS.png')

.. image:: /_static/BS.png
    :align: center

The function :code:`ansatz` generates a universal interferometer:

>>> ansatz(6, 4).draw(path='docs/_static/ansatz6_4.png')

.. image:: /_static/ansatz6_4.png
    :align: center

Each diagram of the module can be converted to a :class:`zw` diagram:

>>> from discopy.drawing import Equation
>>> BS = BBS(0)
>>> double_BS = BS.to_zw()
>>> Equation(BS, double_BS, symbol="$\\mapsto$").draw(\\
... path="docs/_static/double_BS.png")

.. image:: /_static/double_BS.png
    :align: center

**Evaluating linear optical circuits**

:class:`lo` generators correspond to physical linear
optical devices. We can use them to build photonic "chips"
to simulate quantum photonics experiments.

As an example, let us consider a beam splitter and the
Hong-Ou-Mandel effect.

First, let's create a beam splitter:

>>> BS = BBS(0)
>>> BS.draw(path='docs/_static/BS_hom.png', figsize=(2, 2))

.. image:: /_static/BS_hom.png
    :align: center

If we want to evaluate the effect of
inputting two photons using :code:`quimb`,
we need to feed the circuit with two photons.
Finally, let's check the effect of having both
photons on two output modes.

>>> diagram_qpath = Create(1, 1) >> BS >> Select(1, 1)
>>> diagram_qpath.draw(path='docs/_static/BS_hom_2.png', figsize=(3, 3))

.. image:: /_static/BS_hom_2.png
    :align: center

>>> np.round(diagram_qpath.to_zw().to_tensor().to_quimb()^..., 1)
-0.0

We can also do the same using :code:`Perceval`:

>>> diagram_qpath.to_path().prob_with_perceval().array[0, 0]
0j

We can also obtain the entire probability distribution (or amplitudes)
for all possible outcomes of an experiment. Let us use the interferometer
on three modes with three photons:

>>> circuit = Create(1, 1, 1) >> ansatz(3, 3)
>>> circuit.draw(path='docs/_static/ansatz3_3.png', figsize=(5, 5))

.. image:: /_static/ansatz3_3.png
    :align: center

>>> from optyx.utils import tensor_2_amplitudes
>>> symbs = list(circuit.free_symbols)
>>> s = [(i, 0) for i in symbs]
>>> circuit = circuit.subs(*s)
>>> assert np.allclose( \\
...  np.abs(np.round( \\
...  (circuit.to_zw().to_tensor(max_dim=4).to_quimb()^...).data, 1))**2, \\
...  circuit.to_path().prob_with_perceval(as_tensor=True).array)

**Differentiation**

We can also differentiate the expectation values of optical circuits.

>>> from sympy.abc import psi
>>> circuit = BS >> Phase(psi) @ Id(1) >> BS.dagger()
>>> state = Create(2, 0) >> circuit
>>> observable = num_op @ Id(1)
>>> expectation = state >> observable >> state.dagger()
>>> assert np.allclose(
...     expectation.subs((psi, 1/2)).to_path().eval().array, np.array([0.]))
>>> assert np.allclose(
...     expectation.subs((psi, 1/4)).to_path().eval().array, np.array([1.]))
>>> exp = expectation.grad(psi).subs((psi, 1/2))
>>> assert np.allclose(
...     sum([exp.terms[i].to_path().eval().array[0] \\
...      for i in range(len(exp.terms))]), 0.)
>>> exp = expectation.grad(psi).subs((psi, 1/4))
>>> assert np.allclose(
...     sum([exp.terms[i].to_path().eval().array[0] \\
...      for i in range(len(exp.terms))]),
...     -2*np.pi)
>>> exp = expectation.grad(psi).grad(psi).subs((psi, 1/4))
>>> assert np.allclose(
...     sum([exp.terms[i].to_path().eval().array[0] \\
...      for i in range(len(exp.terms))]),
...     np.array([0.]))

References
----------
.. [FC23] de Felice, G., & Coecke, B. (2023). Quantum Linear Optics \
    via String Diagrams. In Proceedings 19th International \
    Conference on Quantum Physics and Logic, Wolfson College, \
    Oxford, UK, 27 June - 1 July 2022 (pp. 83-100). \
        Open Publishing Association.
.. [FSP+23] de Felice, G., Shaikh, R., Poór, B., Yeh, L., Wang, Q., \
    & Coecke, B. (2023). Light-Matter Interaction in the \
    ZXW Calculus. In  Proceedings of the Twentieth \
    International Conference on Quantum Physics and Logic,  \
    Paris, France, 17-21st July 2023 (pp. 20-46). Open Publishing Association.
"""

import numpy as np
from sympy import Expr, lambdify
import sympy as sp

from optyx.optyx import Mode, Box, Scalar
from optyx.path import Matrix
from optyx.zw import Z, W, Create, Select, Endo
from optyx.zw import Split, Merge, Id, SWAP


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
    >>> assert np.allclose(
    ...     (HBS.dagger() >> HBS).to_path().eval(2).array,
    ...                 Id(2).to_path().eval(2).array)
    """

    def __init__(self, array, dom: int, cod: int, name: str, is_dagger=False):
        self.array = array
        super().__init__(
            f"{name}" + ".dagger()" if is_dagger else "",
            Mode(dom),
            Mode(cod),
            is_dagger=is_dagger,
        )

    def conjugate(self):
        array = self.array.conjugate()
        name = self.name + '.conjugate()'
        return Gate(array, len(self.dom), len(self.cod), name, self.is_dagger)

    def to_path(self, dtype=complex):
        result = Matrix[dtype](self.array, len(self.dom), len(self.cod))
        return result.dagger() if self.is_dagger else result

    def dagger(self):
        return Gate(
            np.conjugate(self.array.T),
            len(self.cod),
            len(self.dom),
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
    >>> Phase(1/2).to_path().eval(1).array.round(3)
    array([[-1.+0.j]])
    >>> from sympy.abc import psi
    >>> derivative = Phase(psi).grad(psi).subs((psi,
    ...                     0.5)).to_path().eval(2).array
    >>> assert np.allclose(derivative, 4 * np.pi * 1j)
    """

    def __init__(self, angle: float):
        self.angle = angle
        super().__init__(f"Phase({angle})", Mode(1), Mode(1), data=angle)

    def conjugate(self):
        return Phase(-self.angle)

    def to_path(self, dtype=complex):
        backend = sp if dtype is Expr else np
        exp = backend.exp(2 * backend.pi * 1j * self.angle)
        return Matrix[dtype]([exp], 1, 1)

    def to_zw(self, dtype=complex):
        backend = sp if dtype is Expr else np
        exp = backend.exp(2 * backend.pi * 1j * self.angle)
        return Z(lambda i: exp**i, 1, 1)

    def dagger(self):
        return Phase(-self.angle)

    def grad(self, var):
        """Gradient with respect to :code:`var`."""
        if var not in self.free_symbols:
            return self.sum_factory((), self.dom, self.cod)
        s = 2j * np.pi * self.angle.diff(var)
        d = Scalar(s) @ (self >> num_op)
        return d

    def lambdify(self, *symbols, **kwargs):
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
    >>> assert np.isclose((diagram >> Select(0, 2)).to_path().prob().array,
    ...                                                                0.5)
    >>> assert np.isclose((diagram >> Select(2, 0)).to_path().prob().array,
    ...                                                                0.5)
    >>> assert np.isclose((diagram >> Select(1, 1)).to_path().prob().array,
    ...                                                                  0)

    Check the dagger:

    >>> y = BBS(0.4)
    >>> assert np.allclose((
    ...     y >> y.dagger()).to_path().eval(2).array,
    ...             Id(2).to_path().eval(2).array)
    >>> comp = (y @ y >> Id(1) @ y @ Id(1)) >> \\
    ...             (y @ y >> Id(1) @ y @ Id(1)).dagger()
    >>> assert np.allclose(comp.to_path().eval(2).array,
    ...                     Id(4).to_path().eval(2).array)

    """

    def __init__(self, bias, conj=False):
        self.bias = bias
        self.conj = conj
        name = f"BBS({bias})" if not conj else f"BBS({bias}).conjugate()"
        super().__init__(name, Mode(2), Mode(2), data=bias)

    def conjugate(self):
        return BBS(self.bias, not self.conj)

    def __repr__(self):
        return "BS" if self.bias == 0 else super().__repr__()

    def to_path(self, dtype=complex):
        backend = sp if dtype is Expr else np
        sin = backend.sin((0.25 + self.bias) * backend.pi)
        cos = backend.cos((0.25 + self.bias) * backend.pi)
        if self.conj:
            array = [-1j * cos, sin, sin, -1j * cos]
        else:
            array = [1j * cos, sin, sin, 1j * cos]
        return Matrix[dtype](array, len(self.dom), len(self.cod))

    def to_zw(self, dtype=complex):
        backend = sp if dtype is Expr else np
        sin = backend.sin((0.25 + self.bias) * backend.pi)
        cos = backend.cos((0.25 + self.bias) * backend.pi)
        zb_sin = Z(lambda i: sin ** i, 1, 1)
        if self.conj:
            zb_cos = Z(lambda i: (-cos * 1j) ** i, 1, 1)
        else:
            zb_cos = Z(lambda i: (cos * 1j) ** i, 1, 1)

        beam_splitter = (
            W(2) @ W(2)
            >> Id(1) @ SWAP @ Id(1)
            >> zb_cos @ zb_sin @ zb_sin @ zb_cos
            >> W(2).dagger() @ W(2).dagger()
        )

        return beam_splitter

    def dagger(self):
        return BBS(0.5 - self.bias)

    def lambdify(self, *symbols, **kwargs):
        return lambda *xs: type(self)(
            lambdify(symbols, self.bias, **kwargs)(*xs)
        )


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
    >>> tbs = lambda x: BS >> Id(1) @ Phase(x) >> BS
    >>> assert np.allclose(
    ...     TBS(0.15).to_path().array, tbs(0.15).to_path().array)
    >>> assert np.allclose(
    ...     (TBS(0.25) >> TBS(0.25).dagger()).to_path().array,
    ...     Id(2).to_path().array)
    >>> assert (TBS(0.25).dagger().global_phase() ==\\
    ...         np.conjugate(TBS(0.25).global_phase()))

    """

    def __init__(self, theta, is_dagger=False):
        self.theta = theta
        name = f"TBS({theta})"
        super().__init__(
            name, Mode(2), Mode(2), is_dagger=is_dagger, data=theta
        )

    def conjugate(self):
        return self

    def global_phase(self, dtype=complex):
        backend = sp if dtype is Expr else np
        return (
            -1j * backend.exp(-1j * self.theta * backend.pi)
            if self.is_dagger
            else 1j * backend.exp(1j * self.theta * backend.pi)
        )

    def to_path(self, dtype=complex):
        backend = sp if dtype is Expr else np
        sin = backend.sin(self.theta * backend.pi)
        cos = backend.cos(self.theta * backend.pi)
        array = np.array([sin, cos, cos, -sin])
        array = array * self.global_phase(dtype=dtype)
        matrix = Matrix[dtype](array, len(self.dom), len(self.cod))
        matrix = matrix.dagger() if self.is_dagger else matrix
        return matrix

    def to_zw(self, dtype=complex):
        backend = sp if dtype is Expr else np
        sin = Z(lambda i: (backend.sin(self.theta * backend.pi)) ** i, 1, 1)
        cos = Z(lambda i: (backend.cos(self.theta * backend.pi)) ** i, 1, 1)
        minus_sin = Z(
            lambda i: (-backend.sin(self.theta * backend.pi)) ** i, 1, 1
        )

        beam_splitter = (
            W(2) @ W(2)
            >> Id(1) @ SWAP @ Id(1)
            >> sin @ cos @ cos @ minus_sin
            >> W(2).dagger() @ W(2).dagger()
        )

        return beam_splitter

    def dagger(self):
        return TBS(self.theta, is_dagger=not self.is_dagger)

    def lambdify(self, *symbols, **kwargs):
        return lambda *xs: type(self)(
            lambdify(symbols, self.theta, **kwargs)(*xs),
            is_dagger=self.is_dagger,
        )

    def _decomp(self):
        d = BS >> Id(Mode(1)) @ Phase(self.theta) >> BS
        return d.dagger() if self.is_dagger else d

    def grad(self, var):
        """Gradient with respect to :code:`var`."""
        if var not in self.free_symbols:
            return self.sum_factory((), self.dom, self.cod)
        return self._decomp().grad(var)


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
    >>> assert np.allclose(
    ...     MZI(0.28, 0).to_path().array,
    ...     TBS(0.28).to_path().array)
    >>> assert np.isclose(
    ...    MZI(0.28, 0.3).global_phase(),
    ...    TBS(0.28).global_phase())
    >>> assert np.isclose(
    ...     MZI(0.12, 0.3).global_phase().conjugate(),
    ...     MZI(0.12, 0.3).dagger().global_phase())
    >>> mach = lambda x, y: TBS(x) >> Phase(y) @ Id(1)
    >>> assert np.allclose(
    ...     MZI(0.28, 0.9).to_path().array,
    ...     mach(0.28, 0.9).to_path().array)
    >>> assert np.allclose(
    ...     (MZI(0.28, 0.34) >> MZI(0.28, 0.34).dagger()).to_path().array,
    ...     Id(2).to_path().array)

    """

    def __init__(self, theta, phi, is_dagger=False):
        self.theta, self.phi = theta, phi
        data = {theta, phi}
        super().__init__(
            "MZI", Mode(2), Mode(2), is_dagger=is_dagger, data=data
        )

    def conjugate(self):
        return MZI(self.theta, -self.phi, self.is_dagger)

    def global_phase(self, dtype=complex):
        backend = sp if dtype is Expr else np
        return (
            -1j * backend.exp(-1j * self.theta * backend.pi)
            if self.is_dagger
            else 1j * backend.exp(1j * self.theta * backend.pi)
        )

    def to_path(self, dtype=complex):
        backend = sp if dtype is Expr else np
        cos = backend.cos(backend.pi * self.theta)
        sin = backend.sin(backend.pi * self.theta)
        exp = backend.exp(1j * 2 * backend.pi * self.phi)
        array = np.array([exp * sin, cos, exp * cos, -sin])
        array = array * self.global_phase(dtype=dtype)
        matrix = Matrix[dtype](array, len(self.dom), len(self.cod))
        matrix = matrix.dagger() if self.is_dagger else matrix
        return matrix

    def to_zw(self, dtype=complex):
        backend = sp if dtype is Expr else np
        cos = backend.cos(backend.pi * self.theta)
        sin = backend.sin(backend.pi * self.theta)
        exp = backend.exp(1j * 2 * backend.pi * self.phi)
        mzi = (
            W(2) @ W(2)
            >> Id(1) @ SWAP @ Id(1)
            >> Endo(exp * sin) @ Endo(cos) @ Endo(exp * cos) @ Endo(-sin)
            >> W(2).dagger() @ W(2).dagger()
        )

        return mzi

    def dagger(self):
        return MZI(self.theta, self.phi, is_dagger=not self.is_dagger)

    def lambdify(self, *symbols, **kwargs):
        return lambda *xs: type(self)(
            *lambdify(symbols, [self.theta, self.phi], **kwargs)(*xs),
            is_dagger=self.is_dagger,
        )

    def _decomp(self):
        x, y = self.theta, self.phi
        d = BS >> Id(Mode(1)) @ Phase(x) >> BS >> Phase(y) @ Id(Mode(1))
        return d.dagger() if self.is_dagger else d

    def grad(self, var):
        """Gradient with respect to :code:`var`."""
        if var not in self.free_symbols:
            return self.sum_factory((), self.dom, self.cod)
        return self._decomp().grad(var)


num_op = Split(2) >> Id(1) @ (Select() >> Create()) >> Merge(2)


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

    d = Id(width)
    for i in range(depth):
        n_mzi = (width - 1) // 2 if i % 2 else width // 2
        left = Id(i % 2)
        right = Id(width - len(left.dom) - 2 * n_mzi)
        d >>= left.tensor(*[MZI(*p(i, j)) for j in range(n_mzi)]) @ right

    return d


BS = BBS(0)
