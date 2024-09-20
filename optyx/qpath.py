"""
The category :class:`Matrix` and the syntax :class:`Diagram`
of matrices with creations and post-selections.


.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Matrix
    Amplitudes
    Probabilities
    Diagram
    Box
    Sum
    Swap
    Create
    Select
    Merge
    Split
    Endo
    Scalar

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
"""

from __future__ import annotations

from math import factorial

import numpy as np
import perceval as pcvl

from discopy import symmetric, tensor
from discopy.cat import factory, assert_iscomposable
from discopy.monoidal import PRO
from discopy.utils import unbiased
from optyx.utils import occupation_numbers
import discopy.matrix as underlying
import optyx.zw as zw


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


class Matrix(underlying.Matrix):
    """
    Matrix with photon creations and post-selections,
    evaluated as :class:`Amplitudes`.

    Parameters:
        array : underlying array
        dom : int
        cod : int
        creations : list of occupation numbers
        selections : list of occupation numbers
        normalisation : normalisation factor dependent on number of photons.
        scalar : global scalar independent of number of photons

    Example
    -------
    >>> array = np.array([[1, 1], [1, 0]])
    >>> matrix = Matrix(array, 1, 1, creations=(1,), selections=(1,))
    >>> matrix.eval(3)
    Amplitudes([3.+0.j], dom=1, cod=1)
    >>> num_op = Split() >> Select() @ Id(1) >> Create() @ Id(1) >> Merge()
    >>> assert np.allclose(num_op.eval(4).array, matrix.eval(4).array)
    """

    dtype = complex

    def __new__(
        cls,
        array,
        dom,
        cod,
        creations=(),
        selections=(),
        normalisation=1,
        scalar=1,
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
        """
        Underlying matrix with `len(creations) + dom` inputs and
        `len(selections) + cod` outputs.
        """
        return underlying.Matrix[self.dtype](self.array, self.udom, self.ucod)

    @unbiased
    def then(self, other: Matrix) -> Matrix:
        """Sequential composition of QPath matrices"""
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
            scalar,
        )

    @unbiased
    def tensor(self, other: Matrix) -> Matrix:
        """Parallel composition of QPath matrices"""
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
            umatrix.array,
            dom,
            cod,
            creations,
            selections,
            normalisation,
            scalar,
        )

    def dagger(self) -> Matrix:
        """Adjoint QPath matrix"""
        array = self.umatrix.dagger().array
        return Matrix[self.dtype](
            array,
            self.cod,
            self.dom,
            self.selections,
            self.creations,
            self.normalisation.conjugate(),
            self.scalar,
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
        Returns an equivalent :class:`Matrix` with unitary underlying matrix.

        Example
        -------
        >>> num_op = Split() >> Select() @ Id(1) >> Create() @ Id(1) >> Merge()
        >>> U = num_op.to_path().dilate()
        >>> assert np.allclose(
        ...     (U.umatrix >> U.umatrix.dagger()).array, np.eye(4))
        >>> assert np.allclose(U.eval(5).array, num_op.eval(5).array)
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
        unitary = np.block(
            [[A / s, defect_left], [defect_right, -A.conj().T / s]]
        )
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

    def prob(
        self, n_photons=0, permanent=npperm, with_perceval=False
    ) -> Probabilities:
        """Computes the Born rule of the amplitudes of the :class:`Matrix`"""
        if with_perceval:
            return self.prob_with_perceval(n_photons)
        amplitudes = self.eval(n_photons, permanent)
        probabilities = np.abs(amplitudes.array) ** 2
        return Probabilities[self.dtype](
            probabilities, amplitudes.dom, amplitudes.cod
        )

    def prob_with_perceval(
        self, n_photons=0, simulator: str = "SLOS"
    ) -> Probabilities:
        """
        Computes the Born rule of the amplitudes of the :class:`Matrix` using
        the perceval library

        Note
        ----
        If the :class:`Matrix` is non-unitary, first :meth:`dilate` is called
        to create a unitary.

        Example
        -------
        >>> import numpy as np
        >>> theta, phi = np.pi / 4, 0
        >>> r = np.exp(1j * phi) * np.sin(theta)
        >>> t = np.cos(theta)
        >>> optyx_bs = Split() @ Split() >> Id(PRO(1)) @ SWAP @ Id(PRO(1)) \\
        ...            >> Endo(r) @ Endo(t) @ Endo(np.conj(t)) \\
        ...            @ Endo(-np.conj(r)) >> Merge() @ Merge()
        >>> assert optyx_bs.prob_with_perceval(n_photons=1).round(1)\\
        ...     == Probabilities[complex](
        ...         [0.5+0.j, 0.5+0.j, 0.5+0.j, 0.5+0.j], dom=2, cod=2)
        >>> z_spider = optyx_bs >> Endo(2) @ 1 >> optyx_bs
        >>> assert z_spider.prob_with_perceval(n_photons=1).round(1)\\
        ...     == Probabilities[complex](
        ...         [0.9+0.j, 0.1+0.j, 0.1+0.j, 0.9+0.j], dom=2, cod=2)
        """
        if not self._umatrix_is_is_unitary():
            self = self.dilate()

        circ = self._umatrix_to_perceval_circuit()
        post = self._to_perceval_post_select()

        proc = pcvl.Processor(simulator)
        proc.set_circuit(circ)
        proc.set_postselection(post)

        states = [
            pcvl.BasicState(o + self.creations)
            for o in occupation_numbers(n_photons, self.dom)
        ]
        analyzer = pcvl.algorithm.Analyzer(proc, states, "*")

        permutation = [
            analyzer.col(pcvl.BasicState(o))
            for o in occupation_numbers(
                sum(self.creations) + n_photons, len(self.creations) + self.dom
            )
            if post(pcvl.BasicState(o))
        ]
        return Probabilities[self.dtype](
            analyzer.distribution[:, permutation],
            dom=len(states),
            cod=len(permutation),
        )

    def _umatrix_to_perceval_circuit(self) -> pcvl.Circuit:
        _mzi_triangle = (
            pcvl.Circuit(2)
            // pcvl.BS()
            // (0, pcvl.PS(phi=pcvl.Parameter("phi_1")))
            // pcvl.BS()
            // (0, pcvl.PS(phi=pcvl.Parameter("phi_2")))
        )

        m = pcvl.MatrixN(self.array)
        return pcvl.Circuit.decomposition(
            m,
            _mzi_triangle,
            phase_shifter_fn=pcvl.PS,
            shape="triangle",
            max_try=1,
        )

    def _to_perceval_post_select(self) -> pcvl.PostSelect:
        post = pcvl.PostSelect()
        for i, p in enumerate(self.selections):
            post.eq(self.cod + i, p)
        return post

    def _umatrix_is_is_unitary(self) -> bool:
        m = self.umatrix.array
        return np.allclose(np.eye(m.shape[0]), m.dot(m.conj().T))


class Amplitudes(underlying.Matrix):
    """
    Operator on the Fock space represented as matrix over `occupation_numbers`.

    Example
    -------
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
    Stochastic matrix of probabilities over `occupation_numbers`.

    Example
    -------
    >>> BS.prob(1).round(1)
    Probabilities[complex]([0.5+0.j, 0.5+0.j, 0.5+0.j, 0.5+0.j], dom=2, cod=2)
    >>> (Create(1, 1) >> BS).prob().round(1)
    Probabilities[complex]([0.5+0.j, 0. +0.j, 0.5+0.j], dom=1, cod=3)
    """

    dtype = float

    def __new__(cls, array, dom, cod):
        return underlying.Matrix.__new__(cls, array, dom, cod)

    def normalise(self) -> Probabilities:
        return self.__class__(
            array=self.array / self.array.sum(axis=1)[:, None],
            dom=self.dom,
            cod=self.cod,
        )


@factory
class Diagram(symmetric.Diagram):
    """
    QPath diagram in the sense of https://arxiv.org/abs/2204.12985.
    """

    ty_factory = PRO

    def to_path(self, dtype: type = complex) -> Matrix:
        """Returns the :class:`Matrix` normal form of a :class:`Diagram`."""
        return symmetric.Functor(
            ob=len,
            ar=lambda f: f.to_path(dtype),
            cod=symmetric.Category(int, Matrix[dtype]),
        )(self)

    def eval(self, n_photons=0, permanent=npperm, dtype=complex):
        return self.to_path(dtype).eval(n_photons, permanent)

    def prob(
        self, n_photons=0, permanent=npperm, dtype=complex, with_perceval=False
    ) -> Probabilities:
        return self.to_path(dtype).prob(n_photons, permanent, with_perceval)

    def prob_with_perceval(
        self, n_photons=0, simulator: str = "SLOS", dtype: type = complex
    ) -> Probabilities:
        return self.to_path(dtype).prob_with_perceval(n_photons, simulator)

    @classmethod
    def from_bosonic_operator(cls, n_modes, operators, scalar=1):
        d = cls.id(n_modes)
        annil = Split() >> Select(1) @ Id(1)
        create = annil.dagger()
        for idx, dagger in operators:
            if not (0 <= idx < n_modes):
                raise ValueError(f"Index {idx} out of bounds.")
            box = create if dagger else annil
            d = d >> Id(idx) @ box @ Id(n_modes - idx - 1)

        if scalar != 1:
            d = Scalar(scalar) @ d
        return d

    def to_zw(self) -> zw.Diagram:
        """Converts a :class:`qpath.Diagram` to a :class:`zw.Diagram`."""

        return symmetric.Functor(
            ob=len,
            ar=lambda ob: ob.to_zw(),
            dom=symmetric.Category(PRO, self),
            cod=symmetric.Category(PRO, zw.Diagram),
        )(self)

    grad = tensor.Diagram.grad


class Box(symmetric.Box, Diagram):
    """Box in a :class:`Diagram`"""

    def to_path(self, dtype=complex):
        if isinstance(self.data, Matrix):
            return self.data
        raise NotImplementedError

    def lambdify(self, *symbols, **kwargs):
        # Non-symbolic gates can be returned directly
        return lambda *xs: self

    def subs(self, *args) -> Diagram:
        syms, exprs = zip(*args)
        return self.lambdify(*syms)(*exprs)


class Sum(symmetric.Sum, Box):
    """
    Formal sum of QPath diagrams.

    Example
    -------
    >>> s0, s1 = 1/2 ** 1/2, 1j * 1/2 ** 1/2
    >>> state0 = Scalar(s0) @ Create(1, 0) + Scalar(s1) @ Create(0, 1)
    >>> state1 = Create(1) >> Split() >> Endo(s0) @ Endo(s1)
    >>> assert np.allclose(state0.eval().array, state1.eval().array)
    """

    __ambiguous_inheritance__ = (symmetric.Sum,)
    ty_factory = PRO

    def eval(self, n_photons=0, permanent=npperm, dtype=complex):
        return sum(
            term.eval(n_photons, permanent, dtype) for term in self.terms
        )

    def prob(
        self, n_photons=0, permanent=npperm, dtype=complex
    ) -> Probabilities:
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
    """Swap in a :class:`Diagram`"""

    def to_path(self, dtype=complex) -> Matrix:
        return Matrix([0, 1, 1, 0], 2, 2)

    def to_zw(self) -> zw.Diagram:
        return zw.Swap()

    def dagger(self):
        return self


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

    def to_zw(self) -> zw.Diagram:
        create = zw.Id()
        for n in self.photons:
            create = create @ zw.Create(n)
        return create

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
        name = "Select(1)" if self.photons == (1,) else f"Select{photons}"
        super().__init__(name, len(self.photons), 0)

    def to_path(self, dtype=complex) -> Matrix:
        array = np.eye(len(self.photons))
        return Matrix[dtype](
            array, len(self.photons), 0, selections=self.photons
        )

    def to_zw(self) -> zw.Diagram:
        select = zw.Id()
        for n in self.photons:
            select = select @ zw.Select(n)
        return select

    def dagger(self) -> Diagram:
        return Create(*self.photons)


class Merge(Box):
    """
    Merge map with 2n inputs and n outputs.

    Parameters:
        n : number of output wires

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

    def to_path(self, dtype=complex) -> Matrix:
        array = np.ones(self.n)
        return Matrix[dtype](array, self.n, 1)

    def to_zw(self) -> zw.Diagram:
        return zw.W(self.n).dagger()

    def dagger(self) -> Diagram:
        return Split(n=self.n)


class Split(Box):
    """
    Split map with n inputs and 2n outputs.

    Parameters:
        n : number of input wires

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

    def to_path(self, dtype=complex) -> Matrix:
        array = np.ones(self.n)
        return Matrix[dtype](array, 1, self.n)

    def to_zw(self) -> zw.Diagram:
        return zw.W(self.n)

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
    >>> from sympy import Expr
    >>> from sympy.abc import psi
    >>> import sympy as sp
    >>> assert Endo(3 * psi ** 2).to_path(Expr)\\
    ...     == Matrix[Expr]([3*psi**2], dom=1, cod=1)
    >>> phase = Endo(sp.exp(1j * psi * 2 * sp.pi))
    >>> derivative = phase.grad(psi).subs((psi, 0.5)).eval(2).array
    >>> assert np.allclose(derivative, 4 * np.pi * 1j)
    """

    def __init__(self, scalar: complex):
        try:
            scalar = complex(scalar)
        except TypeError:
            pass
        self.scalar = scalar
        super().__init__(f"Endo({scalar})", 1, 1, data=scalar)

    def to_path(self, dtype=complex) -> Matrix:
        """Returns an equivalent :class:`Matrix` object"""
        return Matrix[dtype]([self.scalar], 1, 1)

    def to_zw(self) -> zw.Diagram:
        return zw.Z(lambda i: self.scalar**i, 1, 1)

    def dagger(self) -> Diagram:
        return Endo(self.scalar.conjugate())

    def grad(self, var):
        if var not in self.free_symbols:
            return self.sum_factory((), self.dom, self.cod)
        s = self.scalar.diff(var) / self.scalar
        num_op = Split() >> Id(1) @ (Select() >> Create()) >> Merge()
        d = Scalar(s) @ (self >> num_op)
        return d

    def lambdify(self, *symbols, **kwargs):
        from sympy import lambdify

        return lambda *xs: type(self)(
            lambdify(symbols, self.scalar, **kwargs)(*xs)
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
        super().__init__(f"Scalar({scalar})", 0, 0, data=scalar)

    def to_path(self, dtype=complex):
        return Matrix[dtype]([], 0, 0, scalar=self.scalar)

    def to_zw(self) -> zw.Diagram:
        return zw.Z([self.scalar], legs_in=0, legs_out=0)

    def dagger(self) -> Diagram:
        return Scalar(self.scalar.conjugate())

    def lambdify(self, *symbols, **kwargs):
        from sympy import lambdify

        return lambda *xs: type(self)(
            lambdify(symbols, self.scalar, **kwargs)(*xs)
        )


bs_array = (1 / 2) ** (1 / 2) * np.array([[1j, 1], [1, 1j]])
bs_matrix = Matrix(bs_array, 2, 2)
BS = Box("BS", 2, 2, data=bs_matrix)

Zb_i = zw.Z(np.array([1, 1j / (np.sqrt(2))]), 1, 1)
Zb_1 = zw.Z(np.array([1, 1 / (np.sqrt(2))]), 1, 1)
beam_splitter = (
    zw.W(2) @ zw.W(2)
    >> Zb_i @ Zb_1 @ Zb_1 @ Zb_i
    >> zw.Id(1) @ zw.Swap() @ zw.Id(1)
    >> zw.W(2).dagger() @ zw.W(2).dagger()
)

BS.to_zw = lambda: beam_splitter

Diagram.swap_factory = Swap
SWAP = Swap(PRO(1), PRO(1))
Id = Diagram.id
Diagram.sum_factory = Sum
