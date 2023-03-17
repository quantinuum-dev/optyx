"""
The category QPath.

Example
-------
>>> HongOuMandel = Create() @ Create() >> Split() @ Split()\\
...     >> Scale(1j) @ SWAP @ Scale(1j)\\
...     >> Merge() @ Merge() >> Delete() @ Delete()
>>> HongOuMandel.to_path()
"""

from __future__ import annotations

from discopy import symmetric
from discopy.cat import factory, assert_iscomposable
from discopy.monoidal import PRO
from discopy.matrix import Matrix
from discopy.utils import mmap


class Path(Matrix[complex]):
    def __init__(self, array, dom: int, cod: int,
                 creations: tuple[int, ...] = (),
                 deletions: tuple[int, ...] = ()):
       self.udom, self.ucod = len(creations) + dom, len(deletions) + cod
       super().__init__(array, self.udom, self.ucod)
       self.dom, self.cod = dom, cod
       self.creations, self.deletions = creations, deletions

    @property
    def umatrix(self) -> Matrix[complex]:
        return Matrix[complex](self.array, self.udom, self.ucod)

    @mmap
    def then(self, other: Path) -> Path:
        assert_iscomposable(self, other)
        M = Matrix[complex]
        a, b = len(self.creations), len(other.creations)
        c, d = len(self.deletions), len(other.deletions)
        umatrix = a @ M.swap(c, self.dom) >> self.umatrix @ c\
            >> b @ M.swap(self.cod, c) >> b @ other.umatrix
        creations = self.creations + other.creations
        deletions = self.deletions + other.deletions
        return Path(umatrix.array, self.dom, other.cod, creations, deletions)

    @mmap
    def tensor(self, other: Path) -> Path:
        M = Matrix[complex]
        a, b = len(self.creations), len(other.creations)
        c, d = len(self.deletions), len(other.deletions)
        umatrix = a @ M.swap(self.dom, b) @ other.dom\
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


@factory
class Diagram(symmetric.Diagram):
    ty_factory = PRO

    def to_path(self):
        return symmetric.Functor(
            ob=len, ar=lambda f: f.to_path(),
            cod=symmetric.Category(int, Path))(self)

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
    >>> Create(1, 2, 3).to_path()
    """
    def __init__(self, *photons: int):
        name = "Create()" if not photons else f"Create({photons})"
        self.photons = photons or (1, )
        super().__init__(name, 0, len(self.photons))

    def to_path(self):
        array = Matrix[complex].id(len(self.photons))
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
        name = "Delete()" if not photons else f"Delete({photons})"
        self.photons = photons or (1, )
        super().__init__(name, len(self.photons), 0)

    def to_path(self):
        array = Matrix[complex].id(len(self.photons))
        return Path(array, len(self.photons), 0, deletions=self.photons)


class Merge(Box):
    def __init__(self, n=2):
        self.n = n
        super().__init__("Merge()" if n == 2 else f"Merge({n})", n, 1)

    def to_path(self):
        return Path.merge(1, self.n)


class Split(Box):
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
