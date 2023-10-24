"""
The category ``qpath.Matrix'' of matrices with creations and annihilations, and the syntax ``qpath.Diagram''.

Example
-------

We can check the Hong-Ou-Mandel effect:

>>> HOM = Create(1, 1) >> BS
>>> HOM.eval()
Amplitudes([0.+0.70710678j, 0.+0.j    , 0.+0.70710678j], dom=1, cod=3)
>>> HOM.prob()
Probabilities([0.5, 0. , 0.5], dom=1, cod=3)
>>> left = Create(1, 1) >> BS >> Select(2, 0)
>>> left.prob()
Probabilities([0.5], dom=1, cod=1)

We can construct a Bell state in dual rail encoding:
>>> plus = Create() >> Split()
>>> state = plus >> Id(1) @ plus @ Id(1)
>>> bell = state @ state >> Id(2) @ (BS @ BS.dagger() >> state.dagger()) @ Id(2)
>>> H, V = Select(1, 0), Select(0, 1)
>>> assert np.allclose((bell >> H @ H).eval().array, (bell >> V @ V).eval().array)
>>> assert np.allclose((bell >> V @ H).eval().array, (bell >> H @ V).eval().array)
"""

from __future__ import annotations

import numpy as np
from math import factorial

from discopy import symmetric
from discopy.cat import factory, assert_iscomposable
from discopy.monoidal import PRO
from discopy import matrix as underlying
from discopy.utils import unbiased


def permanent(M):
    """
    Numpy code for computing the permanent of a matrix,
    from https://github.com/scipy/scipy/issues/7151.
    """
    n = M.shape[0]
    d = np.ones(n)
    j = 0
    s = 1
    f = np.arange(n)
    v = M.sum(axis=0)
    p = np.prod(v)
    while (j < n - 1):
        v -= 2 * d[j] * M[j]
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
        return [(n_photons, )]
    return [(head,) + tail for head in range(n_photons, -1, -1)
            for tail in occupation_numbers(n_photons - head, m_modes - 1)]


class Matrix(underlying.Matrix):
    """
    >>> num_op = Split() >> Select() @ Id(1) >> Create() @ Id(1) >> Merge()
    >>> num_op2 = Split() @ Create() >> Id(1) @ SWAP >> Merge() @ Select()
    >>> assert (num_op @ Id(1)).eval(2) == (num_op2 @ Id(1)).eval(2)
    >>> assert (num_op @ Id(1)).eval(3) == (num_op2 @ Id(1)).eval(3)
    >>> assert (Id(1) @ Create(1) >> num_op @ Id(1) >> Id(1) @ Select(1)).eval(3) == num_op.eval(3)
    >>> assert (num_op @ (Create(1) >> Select(1))).eval(3) == num_op.eval(3)
    >>> assert (Create(1) @ Id(1) >> Id(1) @ Split() >> Select(1) @ Id(2)).eval(3) == Split().eval(3)
    """
    dtype = complex

    def __new__(cls, array, dom, cod, creations=(), selections=()):
        return underlying.Matrix.__new__(cls, array, dom, cod)

    def __init__(self, array, dom: int, cod: int,
                 creations: tuple[int, ...] = (),
                 selections: tuple[int, ...] = ()):
        self.udom, self.ucod = dom + len(creations), cod + len(selections)
        super().__init__(array, self.udom, self.ucod)
        self.dom, self.cod = dom, cod
        self.creations, self.selections = creations, selections

    @property
    def umatrix(self) -> underlying.Matrix[self.dtype]:
        return underlying.Matrix[self.dtype](self.array, self.udom, self.ucod)

    @unbiased
    def then(self, other: Matrix) -> Matrix:
        assert_iscomposable(self, other)
        M = underlying.Matrix[self.dtype]
        a, b = len(self.creations), len(other.creations)
        c, d = len(self.selections), len(other.selections)
        umatrix = self.umatrix @ b >> self.cod @ M.swap(c, b) >> other.umatrix @ c
        creations = self.creations + other.creations
        selections = other.selections + self.selections
        return Matrix(umatrix.array, self.dom, other.cod, creations, selections)

    @unbiased
    def tensor(self, other: Matrix) -> Matrix:
        M = underlying.Matrix[self.dtype]
        a, b = len(self.creations), len(other.creations)
        c, d = len(self.selections), len(other.selections)
        umatrix = self.dom @ M.swap(other.dom, a) @ b\
            >> self.umatrix @ other.umatrix\
            >> self.cod @ M.swap(c, other.cod) @ d
        dom, cod = self.dom + other.dom, self.cod + other.cod
        creations = self.creations + other.creations
        selections = self.selections + other.selections
        return Matrix(umatrix.array, dom, cod, creations, selections)

    def dagger(self) -> Matrix:
        array = self.umatrix.dagger().array
        return Matrix(array, self.cod, self.dom, self.selections, self.creations)

    def __repr__(self):
        return super().__repr__()[:-1]\
            + f", creations={self.creations}, selections={self.selections})"

    def eval(self, n_photons=0, permanent=permanent):
        """ Evaluates the ``Amplitudes'' of a QPath diagram """
        dom_basis = occupation_numbers(n_photons, self.dom)
        n_photons_out = n_photons - sum(self.selections) + sum(self.creations)
        if n_photons_out < 0:
            raise ValueError("Expected a positive number of photons out.")
        cod_basis = occupation_numbers(n_photons_out, self.cod)
        result = Amplitudes[self.dtype].zero(len(dom_basis), len(cod_basis))
        for i, open_creations in enumerate(dom_basis):
            for j, open_selections in enumerate(cod_basis):
                creations = open_creations + self.creations
                selections = open_selections + self.selections
                matrix = np.stack([
                    self.array[:, m]
                    for m, n in enumerate(selections)
                    for _ in range(n)], axis=1)
                matrix = np.stack([
                    matrix[m]
                    for m, n in enumerate(creations)
                    for _ in range(n)], axis=0)
                divisor = np.sqrt(np.prod([
                    factorial(n) for n in creations + selections]))
                result.array[i, j] = permanent(matrix) / divisor
        return result

    def prob(self, n_photons=0, permanent=permanent):
        amplitudes = self.eval(n_photons, permanent)
        probabilities = np.abs(amplitudes.array) ** 2
        return Probabilities(probabilities, amplitudes.dom, amplitudes.cod)


class Amplitudes(underlying.Matrix):
    """
    Matrix of amplitudes for given input and output Fock states with at most n_photons in the input.

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
    Probabilities([0.5, 0.5, 0.5, 0.5], dom=2, cod=2)
    >>> BS.prob(2)
    Probabilities([0.25, 0.5 , 0.25, 0.5 , 0. , 0.5 , 0.25, 0.5 , 0.25], dom=3, cod=3)
    >>> (Create(1, 1) >> BS).prob()
    Probabilities([0.5, 0. , 0.5], dom=1, cod=3)
    """
    dtype = float

    def __new__(cls, array, dom, cod):
        return underlying.Matrix.__new__(cls, array, dom, cod)

@factory
class Diagram(symmetric.Diagram):
    ty_factory = PRO

    def to_path(self, dtype=complex):
        return symmetric.Functor(
            ob=len, ar=lambda f: f.to_path(dtype=dtype),
            cod=symmetric.Category(int, Matrix[dtype]))(self)

    def eval(self, n_photons=0, permanent=permanent):
        return self.to_path().eval(n_photons, permanent)

    def prob(self, n_photons=0, permanent=permanent):
        return self.to_path().prob(n_photons, permanent)


class Box(symmetric.Box, Diagram):
    def to_path(self):
        raise NotImplementedError


class Gate(Box):
    def __init__(self, name: str, dom: int, cod: int, array, is_dagger=False):
        self.array = array
        # self.is_dagger = is_dagger
        super().__init__(f"{name}({array}, is_dagger={is_dagger})", dom, cod, is_dagger=is_dagger)

    def to_path(self, dtype=complex):
        if self.is_dagger:
            return Matrix(self.array, len(self.dom), len(self.cod)).dagger()
        else:
            return Matrix(self.array, len(self.dom), len(self.cod))

    def dagger(self) -> Gate:
        return Gate(self.name, self.dom, self.cod, self.array, is_dagger= not self.is_dagger)


class Swap(symmetric.Swap, Box):
    pass


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
        return Matrix(array, 0, len(self.photons), creations=self.photons)

    def dagger(self) -> Diagram:
        return Select(*self.photons)


class Select(Box):
    """
    Annihilations of photons given a list of occupation numbers.

    Parameters:
        photons : Occupation numbers.

    Example
    -------
    >>> assert Select() == Select(1)
    """
    def __init__(self, *photons: int):
        self.photons = photons or (1,)
        name = "Select()" if self.photons == (1,) else f"Select({photons})"
        super().__init__(name, len(self.photons), 0)

    def to_path(self, dtype=complex):
        array = np.eye(len(self.photons))
        return Matrix(array, len(self.photons), 0, selections=self.photons)

    def dagger(self) -> Diagram:
        return Create(*self.photons)


class Merge(Box):
    """
    Matrix merging.

    Example
    -------
    >>> sqrt2 = Create() >> Scale(2 ** -.5) >> Select()
    >>> assert (sqrt2 @ Create() @ Create() >> Merge()).eval()\\
    ...     == Create(2).eval()
    """
    def __init__(self, n=2):
        self.n = n
        super().__init__("Merge()" if n == 2 else f"Merge({n})", n, 1)

    def to_path(self, dtype=complex):
        array = np.ones(self.n)
        return Matrix(array, self.n, 1)

    def dagger(self) -> Diagram:
        return Split(n=self.n)

class Split(Box):
    """
    Matrix spliting.

    Example
    -------
    >>> (Create() >> Split()).eval()
    Amplitudes([1.+0.j, 1.+0.j], dom=1, cod=2)
    """
    def __init__(self, n=2):
        self.n = n
        super().__init__("Split()" if n == 2 else f"Split({n})", 1, n)

    def to_path(self, dtype=complex):
        array = np.ones(self.n)
        return Matrix(array, 1, self.n)

    def dagger(self) -> Diagram:
        return Merge(n=self.n)


class Scale(Box):
    """
    Scale endomorphism with one input and one output.

    Example
    -------
    >>> (Create(2) >> Split() >> Id(1) @ Scale(0.5)).to_path()
    Matrix([1. +0.j, 0.5+0.j], dom=0, cod=2, creations=(2,), selections=())
    >>> (Create(2) >> Split() >> Id(1) @ Scale(0.5)).eval()
    Amplitudes([1.    +0.j, 0.70710678+0.j, 0.25   +0.j], dom=1, cod=3)
    """
    def __init__(self, scalar: complex):
        self.scalar = scalar
        super().__init__(f"Scale({scalar})", 1, 1)

    def to_path(self, dtype=complex):
        return Matrix([self.scalar], 1, 1)


class Phase(Box):
    """ 
    Phase shift with angle parameter between 0 and 1

    Example
    -------
    >>> Phase(1/8).to_path()
    Matrix([0.70710678+0.70710678j], dom=1, cod=1, creations=(), selections=())
    >>> Phase(1/2).eval(1).array.round(3)
    array([[-1.+0.j]])
    """
    def __init__(self, angle: float):
        self.angle = angle
        super().__init__(f"Phase({angle})", 1, 1)

    def to_path(self, dtype=complex):
        return Matrix([np.exp(2 * np.pi * 1j * self.angle)], 1, 1)


Diagram.swap_factory = Swap
SWAP = Swap(PRO(1), PRO(1))
Id = Diagram.id

bs_array = (1/2) ** (1/2) * np.array([[1j, 1], [1, 1j]])
BS = Gate('BS', 2, 2, bs_array)
