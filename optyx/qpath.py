"""
The category QPath.

Example
-------
>>> HongOuMandel = Create() @ Create() >> Split() @ Split()\\
...     >> Scale(1j) @ SWAP @ Scale(1j)\\
...     >> Merge() @ Merge() >> Delete() @ Delete()
>>> HongOuMandel.eval()
Matrix[Expr]([0j], dom=1, cod=1)
"""

from __future__ import annotations

import numpy as np
from math import factorial
import sympy as sp

from discopy import symmetric
from discopy.cat import factory, assert_iscomposable
from discopy.monoidal import PRO
from discopy.matrix import Matrix
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


class Path(Matrix[sp.Expr]):
    """
    >>> num_op = Split() >> Delete() @ Id(1) >> Create() @ Id(1) >> Merge()
    >>> num_op2 = Split() @ Create() >> Id(1) @ SWAP >> Merge() @ Delete()
    >>> assert (num_op @ Id(1)).eval(2) == (num_op2 @ Id(1)).eval(2)
    >>> assert (num_op @ Id(1)).eval(3) == (num_op2 @ Id(1)).eval(3)
    >>> assert (Create(3, 0) >> number_op @ Id(1) >> Delete(3, 0)).eval() ==\
    ...     (Create(3) >> number_op >> Delete(3)).eval()
    """
    def __new__(cls, array, dom, cod, creations=(), deletions=()):
        return Matrix[sp.Expr].__new__(cls, array, dom, cod)

    def __init__(self, array, dom: int, cod: int,
                 creations: tuple[int, ...] = (),
                 deletions: tuple[int, ...] = ()):
        self.udom, self.ucod = len(creations) + dom, len(deletions) + cod
        super().__init__(array, self.udom, self.ucod)
        self.dom, self.cod = dom, cod
        self.creations, self.deletions = creations, deletions

    @property
    def umatrix(self) -> Matrix[sp.Expr]:
        return Matrix[sp.Expr](self.array, self.udom, self.ucod)

    @unbiased
    def then(self, other: Path) -> Path:
        assert_iscomposable(self, other)
        M = Matrix[sp.Expr]
        a, b = len(self.creations), len(other.creations)
        c, d = len(self.deletions), len(other.deletions)
        umatrix = a @ M.swap(b, self.dom) >> self.umatrix @ b\
            >> c @ M.swap(self.cod, b) >> c @ other.umatrix
        creations = self.creations + other.creations
        deletions = self.deletions + other.deletions
        return Path(umatrix.array, self.dom, other.cod, creations, deletions)

    @unbiased
    def tensor(self, other: Path) -> Path:
        M = Matrix[sp.Expr]
        a, b = len(self.creations), len(other.creations)
        c, d = len(self.deletions), len(other.deletions)
        umatrix = a @ M.swap(b, self.dom) @ other.dom\
            >> self.umatrix @ other.umatrix\
            >> c @ M.swap(self.cod, d) @ other.cod
        dom, cod = self.dom + other.dom, self.cod + other.cod
        creations = self.creations + other.creations
        deletions = self.deletions + other.deletions
        return Path(umatrix.array, dom, cod, creations, deletions)

    def dagger(self) -> Path:
        array = self.umatrix.dagger().array
        return Path(array, self.cod, self.dom, self.deletions, self.creations)

    def __repr__(self):
        return super().__repr__()[:-1]\
            + f", creations={self.creations}, deletions={self.deletions})"

    def make_square(self):
        diff = self.udom - self.ucod
        if diff < 0:
            creations = self.creations + -diff * (0,)
            array = np.concatenate([
                self.array[:len(self.creations)],
                np.zeros((-diff, self.ucod), complex),
                self.array[len(self.creations):]], axis=0)
            return Path(array, self.dom, self.cod, creations, self.deletions)
        if diff > 0:
            deletions = self.deletions + diff * (0,)
            array = np.concatenate([
                self.array[:, :len(self.deletions)],
                np.zeros((self.udom, diff), complex),
                self.array[:, len(self.deletions):]], axis=1)
            return Path(array, self.dom, self.cod, self.creations, deletions)
        return self

    def eval(self, n_photons=0, permanent=permanent):
        if self.udom != self.ucod:
            return self.make_square().eval(n_photons, permanent)
        dom_basis = occupation_numbers(n_photons, self.dom)
        n_photons_out = n_photons - sum(self.deletions) + sum(self.creations)
        if n_photons_out < 0:
            raise ValueError("Expected a positive number of photons out.")
        cod_basis = occupation_numbers(n_photons_out, self.cod)
        result = Matrix[sp.Expr].zero(len(dom_basis), len(cod_basis))
        for i, open_creations in enumerate(dom_basis):
            for j, open_deletions in enumerate(cod_basis):
                creations = self.creations + open_creations
                deletions = self.deletions + open_deletions
                matrix = np.stack([
                    self.array[:, m]
                    for m, n in enumerate(deletions)
                    for _ in range(n)], axis=1)
                matrix = np.stack([
                    matrix[m]
                    for m, n in enumerate(creations)
                    for _ in range(n)], axis=0)
                divisor = np.sqrt(np.prod([
                    factorial(n) for n in creations + deletions]))
                result.array[i, j] = permanent(matrix) / divisor
        return result

    def get_amplitudes(self, n_photons=0, permanent=permanent):
        if self.dom != 0:
            raise ValueError("Expected a state, got a process.")
        n_photons_out = n_photons - sum(self.deletions) + sum(self.creations)
        basis = occupation_numbers(n_photons_out, self.cod)
        counts = self.eval(n_photons, permanent).array[0]
        return {key: value for key, value in zip(basis, counts) if value}

    def get_probabilities(self, n_photons=0, permanent=permanent):
        amplitudes = self.get_amplitudes(n_photons, permanent).items()
        return {key: abs(value) ** 2 for key, value in amplitudes}


@factory
class Diagram(symmetric.Diagram):
    ty_factory = PRO

    def to_path(self):
        return symmetric.Functor(
            ob=len, ar=lambda f: f.to_path(),
            cod=symmetric.Category(int, Path))(self)

    def eval(self, n_photons=0, permanent=permanent):
        return self.to_path().eval(n_photons, permanent)

    def get_amplitudes(self, n_photons=0, permanent=permanent):
        return self.to_path().get_amplitudes(n_photons, permanent)

    def get_probabilities(self, n_photons=0, permanent=permanent):
        return self.to_path().get_probabilities(n_photons, permanent)


class Box(symmetric.Box, Diagram):
    def to_path(self):
        raise NotImplementedError


class Swap(symmetric.Swap, Box):
    pass


class Create(Box):
    """
    Creation of photons given a list of occupation numbers.

    Parameters:
        photons : Occupation numbers.

    Example
    -------
    >>> assert Create() == Create(1)
    >>> Create(1, 2, 3).get_probabilities()
    {(1, 2, 3): 1.0}
    """
    def __init__(self, *photons: int):
        self.photons = photons or (1,)
        name = "Create()" if self.photons == (1,) else f"Create({photons})"
        super().__init__(name, 0, len(self.photons))

    def to_path(self):
        array = Matrix[sp.Expr].id(len(self.photons)).array
        return Path(array, 0, len(self.photons), creations=self.photons)


class Delete(Box):
    """
    Annihilation of photons given a list of occupation numbers.

    Parameters:
        photons : Occupation numbers.

    Example
    -------
    >>> assert Delete() == Delete(1)
    """
    def __init__(self, *photons: int):
        self.photons = photons or (1,)
        name = "Delete()" if self.photons == (1,) else f"Delete({photons})"
        super().__init__(name, len(self.photons), 0)

    def to_path(self):
        array = Matrix[sp.Expr].id(len(self.photons)).array
        return Path(array, len(self.photons), 0, deletions=self.photons)


class Merge(Box):
    """
    Path merging.

    Example
    -------
    >>> sqrt2 = Create() >> Scale(2 ** -.5) >> Delete()
    >>> assert (sqrt2 @ Create() @ Create() >> Merge()).eval()\\
    ...     == Create(2).eval()
    """
    def __init__(self, n=2):
        self.n = n
        super().__init__("Merge()" if n == 2 else f"Merge({n})", n, 1)

    def to_path(self):
        return Path.merge(1, self.n)


class Split(Box):
    """
    Path spliting.

    Example
    -------
    >>> (Create() >> Split()).get_probabilities()
    {(1, 0): 1.0, (0, 1): 1.0}
    """
    def __init__(self, n=2):
        self.n = n
        super().__init__("Split()" if n == 2 else f"Split({n})", 1, n)

    def to_path(self):
        return Path.copy(1, self.n)


class Scale(Box):
    def __init__(self, scalar: complex):
        self.scalar = scalar
        super().__init__(f"Scalar({scalar})", 1, 1)

    def to_path(self):
        return Path([self.scalar], 1, 1)


Diagram.swap_factory = Swap
SWAP = Swap(PRO(1), PRO(1))
Id = Diagram.id
