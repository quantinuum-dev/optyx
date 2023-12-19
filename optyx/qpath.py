"""
The category `qpath.Matrix` of matrices with creations and post-selections,
and the syntax `qpath.Diagram`.

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Matrix
    Amplitudes
    Probabilities
    Diagram
    Box
    Swap
    Create
    Select
    Merge
    Split
    Endo
    Gate
    Phase

.. admonition:: Functions

    .. autosummary::
        :template: function.rst
        :nosignatures:
        :toctree:

        npperm
        occupation_numbers

Examples
--------

We can check the Hong-Ou-Mandel effect:

>>> HOM = Create(1, 1) >> BS
>>> HOM.eval()
Amplitudes([0.+0.70710678j, 0.+0.j    , 0.+0.70710678j], dom=1, cod=3)
>>> HOM.prob()
Probabilities[complex]([0.5+0.j, 0. +0.j, 0.5+0.j], dom=1, cod=3)
>>> left = Create(1, 1) >> BS >> Select(2, 0)
>>> left.prob()
Probabilities[complex]([0.5+0.j], dom=1, cod=1)

We can construct a Bell state in dual rail encoding:


>>> plus = Create() >> Split()
>>> state = plus >> Id(1) @ plus @ Id(1)
>>> bell = state @ state\\
...     >> Id(2) @ (BS @ BS.dagger() >> state.dagger()) @ Id(2)
>>> H, V = Select(1, 0), Select(0, 1)
>>> assert np.allclose(
...     (bell >> H @ H).eval().array, (bell >> V @ V).eval().array)
>>> assert np.allclose(
...     (bell >> V @ H).eval().array, (bell >> H @ V).eval().array)

We can define the number operator and compute its expectation.

>>> num_op = Split() >> Id(1) @ Select(1) >> Id(1) @ Create(1) >> Merge()
>>> expectation = lambda n: Create(n) >> num_op >> Select(n)
>>> assert np.allclose(expectation(5).eval().array, np.array([5.]))

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
...     expectation.subs((psi, 0)).eval().array, np.array([2.]))
>>> assert np.allclose(
...     expectation.grad(psi).subs((psi, 1/2)).eval().array, np.array([0.]))
>>> assert np.allclose(
...     expectation.grad(psi).subs((psi, 1/4)).eval().array,
...     np.array([-2*np.pi]))
>>> assert np.allclose(
...     expectation.grad(psi).subs((psi, 0)).eval().array, np.array([0.]))
>>> assert np.allclose(
...     expectation.grad(psi).grad(psi).subs((psi, 1/4)).eval().array,
...     np.array([0.]))
"""

from __future__ import annotations

from math import factorial
import numpy as np

from discopy import symmetric
from discopy.cat import factory, assert_iscomposable
from discopy.monoidal import PRO
from discopy import matrix as underlying
from discopy.utils import unbiased


def npperm(matrix):
    """
    Numpy code for computing the permanent of a matrix,
    from https://github.com/scipy/scipy/issues/7151.
    """
    n = matrix.shape[0]
    d = np.ones(n)
    j = 0
    s = 1
    f = np.arange(n)
    v = matrix.sum(axis=0)
    p = np.prod(v)
    while j < n - 1:
        v -= 2 * d[j] * matrix[j]
        d[j] = -d[j]
        s = -s
        prod = np.prod(v)
        p += s * prod
        f[0] = 0
        f[j] = f[j + 1]
        f[j + 1] = j + 1
        j = f[0]
    return p / 2 ** (n - 1)


def occupation_numbers(n_photons, m_modes):
    """
    Returns vectors of occupation numbers for n_photons in m_modes.

    Example
    -------
    >>> occupation_numbers(3, 2)
    [(3, 0), (2, 1), (1, 2), (0, 3)]
    >>> occupation_numbers(2, 3)
    [(2, 0, 0), (1, 1, 0), (1, 0, 1), (0, 2, 0), (0, 1, 1), (0, 0, 2)]
    """
    if not n_photons:
        return [m_modes * (0,)]
    if not m_modes:
        raise ValueError(f"Can't put {n_photons} photons in zero modes!")
    if m_modes == 1:
        return [(n_photons,)]
    return [
        (head,) + tail
        for head in range(n_photons, -1, -1)
        for tail in occupation_numbers(n_photons - head, m_modes - 1)
    ]


class Matrix(underlying.Matrix):
    """
    Matrix with photon creations and post-selections,
    interpreted as an operator on the Fock space via :class:`Amplitudes`

    >>> array = np.array([[1, 1], [1, 0]])
    >>> matrix = Matrix(array, 1, 1, creations=(1,), selections=(1,))
    >>> matrix.eval(3)
    Amplitudes([3.+0.j], dom=1, cod=1)
    >>> num_op = Split() >> Select() @ Id(1) >> Create() @ Id(1) >> Merge()
    >>> assert np.allclose(num_op.eval(5).array, matrix.eval(5).array)
    >>> num_op2 = Split() @ Create() >> Id(1) @ SWAP >> Merge() @ Select()
    >>> assert (num_op @ Id(1)).eval(2) == (num_op2 @ Id(1)).eval(2)
    >>> assert (num_op @ Id(1)).eval(3) == (num_op2 @ Id(1)).eval(3)
    >>> assert (
    ...     Id(1) @ Create(1) >> num_op @ Id(1) >> Id(1) @ Select(1)
    ...     ).eval(3) == num_op.eval(3)
    >>> assert (num_op @ (Create(1) >> Select(1))).eval(3) == num_op.eval(3)
    >>> assert (
    ...     Create(1) @ Id(1) >> Id(1) @ Split() >> Select(1) @ Id(2)
    ...     ).eval(3) == Split().eval(3)
    """

    dtype = complex

    def __new__(
        cls, array, dom, cod,
        creations=(), selections=(), normalisation=1, scalar=1,
    ):
        return underlying.Matrix.__new__(cls, array, dom, cod)

    def __init__(
        self,
        array,
        dom: int,
        cod: int,
        creations: tuple[int, ...] = (),
        selections: tuple[int, ...] = (),
        normalisation=1,
        scalar=1,
    ):
        self.udom, self.ucod = dom + len(creations), cod + len(selections)
        super().__init__(array, self.udom, self.ucod)
        self.dom, self.cod = dom, cod
        self.creations, self.selections = creations, selections
        self.normalisation = normalisation
        self.scalar = scalar

    @property
    def umatrix(self) -> underlying.Matrix:
        return underlying.Matrix[self.dtype](self.array, self.udom, self.ucod)

    @unbiased
    def then(self, other: Matrix) -> Matrix:
        assert_iscomposable(self, other)
        M = underlying.Matrix[self.dtype]
        left, right = len(self.selections), len(other.creations)
        umatrix = (
            self.umatrix @ right
            >> self.cod @ M.swap(left, right)
            >> other.umatrix @ left
        )
        creations = self.creations + other.creations
        selections = other.selections + self.selections
        scalar = self.scalar * other.scalar
        normalisation = self.normalisation * other.normalisation
        return Matrix[self.dtype](
            umatrix.array,
            self.dom,
            other.cod,
            creations,
            selections,
            normalisation,
            scalar
        )

    @unbiased
    def tensor(self, other: Matrix) -> Matrix:
        M = underlying.Matrix[self.dtype]
        a, b = len(self.creations), len(other.creations)
        c, d = len(self.selections), len(other.selections)
        umatrix = (
            self.dom @ M.swap(other.dom, a) @ b
            >> self.umatrix @ other.umatrix
            >> self.cod @ M.swap(c, other.cod) @ d
        )
        dom, cod = self.dom + other.dom, self.cod + other.cod
        creations = self.creations + other.creations
        selections = self.selections + other.selections
        normalisation = self.normalisation * other.normalisation
        scalar = self.scalar * other.scalar
        return Matrix[self.dtype](
            umatrix.array, dom, cod,
            creations, selections, normalisation, scalar
        )

    def dagger(self) -> Matrix:
        array = self.umatrix.dagger().array
        return Matrix[self.dtype](
            array,
            self.cod,
            self.dom,
            self.selections,
            self.creations,
            self.normalisation.conjugate(),
            self.scalar
        )

    def __repr__(self):
        return (
            super().__repr__()[:-1]
            + f", creations={self.creations}"
            + f", selections={self.selections}"
            + f", normalisation={self.normalisation}"
            + f", scalar={self.scalar})"
        )

    def dilate(self) -> Matrix:
        """
        Returns an equivalent qpath `Matrix` with unitary underlying matrix.

        >>> num_op = Split() >> Select() @ Id(1) >> Create() @ Id(1) >> Merge()
        >>> U = num_op.to_path().dilate()
        >>> assert np.allclose(
        ...     (U.umatrix >> U.umatrix.dagger()).array, np.eye(4))
        >>> assert np.allclose(U.eval(5).array, num_op.eval(5).array)
        >>> M = Matrix(
        ...     [1, 2, 1, 1, 1, 4, 1, 1, 0, 4, 1, 0],
        ...     dom=2, cod=3, creations=(1, ), selections=(2, ))
        >>> U1 = M.dilate()
        >>> assert np.allclose(
        ...     (U1.umatrix >> U1.umatrix.dagger()).array,
        ...     np.eye(U1.umatrix.dom))
        >>> assert np.allclose(U1.eval(5).array, M.eval(5).array)
        """
        dom, cod = self.umatrix.dom, self.umatrix.cod
        A = self.umatrix.array
        U, S, Vh = np.linalg.svd(A)
        s = max(S) if max(S) > 1 else 1
        defect0 = np.concatenate(
            [np.sqrt(1 - (S / s) ** 2), [1 for _ in range(dom - len(S))]]
        )
        defect1 = np.concatenate(
            [np.sqrt(1 - (S / s) ** 2), [1 for _ in range(cod - len(S))]]
        )
        defect_left = U.dot(np.diag(defect0)).dot(U.conj().T)
        defect_right = (Vh.conj().T).dot(np.diag(defect1)).dot(Vh)
        unitary = np.block([[A / s, defect_left],
                            [defect_right, -A.conj().T / s]])
        creations = self.creations + cod * (0,)
        selections = self.selections + dom * (0,)
        return Matrix(
            unitary, self.dom, self.cod, creations, selections, normalisation=s
        )

    def eval(self, n_photons=0, permanent=npperm) -> Amplitudes:
        """Evaluates the :class:`Amplitudes` of a the QPath matrix"""
        dom_basis = occupation_numbers(n_photons, self.dom)
        n_photons_out = n_photons - sum(self.selections) + sum(self.creations)
        if n_photons_out < 0:
            raise ValueError("Expected a positive number of photons out.")
        cod_basis = occupation_numbers(n_photons_out, self.cod)
        result = Amplitudes[self.dtype].zero(len(dom_basis), len(cod_basis))
        normalisation = self.normalisation ** (n_photons + sum(self.creations))
        for i, open_creations in enumerate(dom_basis):
            for j, open_selections in enumerate(cod_basis):
                creations = open_creations + self.creations
                selections = open_selections + self.selections
                matrix = np.stack(
                    [
                        self.array[:, m]
                        for m, n in enumerate(selections)
                        for _ in range(n)
                    ],
                    axis=1,
                )
                matrix = np.stack(
                    [
                        matrix[m]
                        for m, n in enumerate(creations)
                        for _ in range(n)
                    ],
                    axis=0,
                )
                divisor = np.sqrt(
                    np.prod([factorial(n) for n in creations + selections])
                )
                result.array[i, j] = (
                    self.scalar * normalisation * permanent(matrix) / divisor
                )
        return result

    def prob(self, n_photons=0, permanent=npperm) -> Probabilities:
        """ Computes the Born rule of the amplitudes for a given `Matrix`"""
        amplitudes = self.eval(n_photons, permanent=npperm)
        probabilities = np.abs(amplitudes.array) ** 2
        return Probabilities[self.dtype](
            probabilities, amplitudes.dom, amplitudes.cod
        )


class Amplitudes(underlying.Matrix):
    """
    Matrix of amplitudes for given
    input and output Fock states with at most n_photons in the input.

    >>> BS.eval(1)
    Amplitudes([0.    +0.70710678j, 0.70710678+0.j    , 0.70710678+0.j    ,
     0.    +0.70710678j], dom=2, cod=2)
    >>> assert isinstance(BS.eval(2), Amplitudes)
    >>> (BS >> Select(1) @ Id(1)).eval(2)
    Amplitudes([0.+0.70710678j, 0.+0.j    , 0.+0.70710678j], dom=3, cod=1)
    """

    dtype = complex

    def __new__(cls, array, dom, cod):
        return underlying.Matrix.__new__(cls, array, dom, cod)


class Probabilities(underlying.Matrix):
    """
    Stochastic matrix of probabilities for given input and output Fock states.

    >>> BS.prob(1)
    Probabilities[complex]([0.5+0.j, 0.5+0.j, 0.5+0.j, 0.5+0.j], dom=2, cod=2)
    >>> (Create(1, 1) >> BS).prob()
    Probabilities[complex]([0.5+0.j, 0. +0.j, 0.5+0.j], dom=1, cod=3)
    """

    dtype = float

    def __new__(cls, array, dom, cod):
        return underlying.Matrix.__new__(cls, array, dom, cod)


@factory
class Diagram(symmetric.Diagram):
    """
    QPath diagram in the sense of https://arxiv.org/abs/2204.12985.
    """
    ty_factory = PRO

    def to_path(self, dtype=complex) -> Matrix:
        """Returns the :class:`Matrix` normal form of a :class:`Diagram`."""
        return symmetric.Functor(
            ob=len,
            ar=lambda f: f.to_path(dtype=dtype),
            cod=symmetric.Category(int, Matrix[dtype]),
        )(self)

    def eval(self, n_photons=0, permanent=npperm, dtype=complex):
        return self.to_path(dtype).eval(n_photons, permanent)

    def prob(self, n_photons=0, permanent=npperm, dtype=complex):
        return self.to_path(dtype).prob(n_photons, permanent)

    def grad(self, var, **params):
        """Gradient with respect to :code:`var`."""
        if var not in self.free_symbols:
            return self.sum_factory((), self.dom, self.cod)
        left, box, right, tail = tuple(self.inside[0]) + (self[1:],)
        t1 = self.id(left) @ box.grad(var, **params) @ self.id(right) >> tail
        t2 = self.id(left) @ box @ self.id(right) >> tail.grad(var, **params)
        return t1 + t2


class Box(symmetric.Box, Diagram):
    """ Box in a :class:`Diagram`"""
    def to_path(self, dtype=complex):
        raise NotImplementedError

    def lambdify(self, *symbols, **kwargs):
        # Non-symbolic gates can be returned directly
        return lambda *xs: self

    def subs(self, *args) -> Box:
        syms, exprs = zip(*args)
        return self.lambdify(*syms)(*exprs)


class Sum(symmetric.Sum, Box):
    __ambiguous_inheritance__ = (symmetric.Sum,)
    ty_factory = PRO

    def eval(self, n_photons=0, permanent=npperm, dtype=complex):
        return sum(
            term.eval(n_photons, permanent, dtype) for term in self.terms
        )

    def prob(self, n_photons=0, permanent=npperm, dtype=complex):
        amplitudes = self.eval(n_photons, permanent, dtype)
        probabilities = np.abs(amplitudes.array) ** 2
        return Probabilities[dtype](
            probabilities, amplitudes.dom, amplitudes.cod
        )

    def grad(self, var, **params):
        """Gradient with respect to :code:`var`."""
        if var not in self.free_symbols:
            return self.sum_factory((), self.dom, self.cod)
        return sum(term.grad(var, **params) for term in self.terms)


class Swap(symmetric.Swap, Box):
    """ Swap in a :class:`Diagram`"""


class Create(Box):
    """
    Creation of photons on modes given a list of occupation numbers.

    Parameters:
        photons : Occupation numbers.

    Example
    -------
    >>> assert Create() == Create(1)
    >>> Create(1).eval()
    Amplitudes([1.+0.j], dom=1, cod=1)
    """

    def __init__(self, *photons: int):
        self.photons = photons or (1,)
        name = "Create()" if self.photons == (1,) else f"Create({photons})"
        super().__init__(name, 0, len(self.photons))

    def to_path(self, dtype=complex):
        array = np.eye(len(self.photons))
        return Matrix[dtype](
            array, 0, len(self.photons), creations=self.photons
        )

    def dagger(self) -> Diagram:
        return Select(*self.photons)


class Select(Box):
    """
    Post-selection of photons given a list of occupation numbers.

    Parameters:
        photons : Occupation numbers.

    Example
    -------
    >>> assert Select() == Select(1)
    >>> assert Select(2).dagger() == Create(2)
    """

    def __init__(self, *photons: int):
        self.photons = photons or (1,)
        name = "Select()" if self.photons == (1,) else f"Select({photons})"
        super().__init__(name, len(self.photons), 0)

    def to_path(self, dtype=complex):
        array = np.eye(len(self.photons))
        return Matrix[dtype](
            array, len(self.photons), 0, selections=self.photons
        )

    def dagger(self) -> Diagram:
        return Create(*self.photons)


class Merge(Box):
    """
    Merge map with two inputs and one output

    Example
    -------
    >>> sqrt2 = Create() >> Endo(2 ** -.5) >> Select()
    >>> assert (sqrt2 @ Create() @ Create() >> Merge()).eval()\\
    ...     == Create(2).eval()
    >>> assert Merge().dagger() == Split()
    """

    def __init__(self, n=2):
        self.n = n
        name = "Merge()" if n == 2 else f"Merge({n})"
        super().__init__(name, n, 1)

    def to_path(self, dtype=complex):
        array = np.ones(self.n)
        return Matrix[dtype](array, self.n, 1)

    def dagger(self) -> Diagram:
        return Split(n=self.n)


class Split(Box):
    """
    Split map with one input and two outputs.

    Example
    -------
    >>> (Create() >> Split()).eval()
    Amplitudes([1.+0.j, 1.+0.j], dom=1, cod=2)
    >>> assert Split().dagger() == Merge()
    """

    def __init__(self, n=2):
        self.n = n
        name = "Split()" if n == 2 else f"Split({n})"
        super().__init__(name, 1, n)

    def to_path(self, dtype=complex):
        array = np.ones(self.n)
        return Matrix[dtype](array, 1, self.n)

    def dagger(self) -> Diagram:
        return Merge(n=self.n)


class Endo(Box):
    """
    Endomorphism with one input and one output.

    Parameters:
        scalar : complex

    Example
    -------
    >>> assert (Create(2) >> Split() >> Id(1) @ Endo(0.5)).to_path()\\
    ...     == Matrix(
    ...         [1. +0.j, 0.5+0.j], dom=0, cod=2,
    ...         creations=(2,), selections=(), normalisation=1)
    """

    def __init__(self, scalar: complex):
        try:
            scalar = complex(scalar)
        except TypeError:
            pass
        self.scalar = scalar
        super().__init__(f"Endo({scalar})", 1, 1)

    def to_path(self, dtype=complex):
        return Matrix[dtype]([self.scalar], 1, 1)

    def dagger(self) -> Diagram:
        return Endo(self.scalar.conjugate())

    def grad(self, var):
        if var not in self.free_symbols:
            return self.sum_factory((), self.dom, self.cod)
        s = self.scalar.diff(var) / self.scalar
        num_op = (
            Split()
            >> Id(1) @ Endo(s)
            >> Id(1) @ (Select() >> Create())
            >> Merge()
        )
        d = self >> num_op
        return d

    def lambdify(self, *symbols, **kwargs):
        from sympy import lambdify

        return lambda *xs: type(self)(
            lambdify(symbols, self.scalar, **kwargs)(*xs)
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
    """

    def __init__(self, angle: float):
        self.angle = angle
        super().__init__(f"Phase({angle})", 1, 1, data=angle)

    def to_path(self, dtype=complex):
        return Matrix[dtype][dtype]([np.exp(2 * np.pi * 1j * self.angle)], 1, 1)

    def dagger(self) -> Diagram:
        return Phase(-self.angle)

    def grad(self, var):
        if var not in self.free_symbols:
            return self.sum_factory((), self.dom, self.cod)
        s = 2j * np.pi * self.angle.diff(var)
        num_op = (
            Split()
            >> Id(1) @ Endo(s)
            >> Id(1) @ (Select() >> Create())
            >> Merge()
        )
        d = self >> num_op
        return d

    def lambdify(self, *symbols, **kwargs):
        from sympy import lambdify

        return lambda *xs: type(self)(
            lambdify(symbols, self.angle, **kwargs)(*xs)
        )


class Gate(Box):
    """
    Creates an instance of :class:`Box` given array, domain and codomain.

    >>> hbs_array = (1 / 2) ** (1 / 2) * np.array([[1, 1], [1, -1]])
    >>> HBS = Gate("HBS", 2, 2, hbs_array)
    >>> assert np.allclose((HBS.dagger() >> HBS).eval(2).array,
    ...                    Id(2).eval(2).array)
    """
    def __init__(self, name: str, dom: int, cod: int, array, is_dagger=False):
        self.array = array
        # self.is_dagger = is_dagger
        super().__init__(
            f"{name}({array}, is_dagger={is_dagger})",
            dom,
            cod,
            is_dagger=is_dagger,
        )

    def to_path(self, dtype=complex):
        result = Matrix[dtype](self.array, len(self.dom), len(self.cod))
        return result.dagger() if self.is_dagger else result

    def dagger(self) -> Gate:
        return Gate(
            self.name,
            self.dom,
            self.cod,
            self.array,
            is_dagger=not self.is_dagger,
        )


class Scalar(Box):
    """
    Scalar in a QPath diagram

    Example
    -------
    >>> assert Scalar(0.45).to_path() == Matrix(
    ...     [], dom=0, cod=0,
    ...     creations=(), selections=(), normalisation=1, scalar=0.45)
    >>> s = Scalar(- 1j * 2 ** (1/2)) @ Create(1, 1) >> BS >> Select(2, 0)
    >>> assert np.isclose(s.eval().array[0], 1)
    """
    def __init__(self, scalar: complex):
        self.scalar = scalar
        super().__init__(f"Scalar({scalar})", 0, 0)

    def to_path(self, dtype=complex):
        return Matrix([], 0, 0, scalar=self.scalar)


Diagram.swap_factory = Swap
SWAP = Swap(PRO(1), PRO(1))
Id = Diagram.id

bs_array = (1 / 2) ** (1 / 2) * np.array([[1j, 1], [1, 1j]])
BS = Gate("BS", 2, 2, bs_array)

Diagram.sum_factory = Sum
