"""
Optyx diagrams have wires of type `code:bit` or `code:mode`
"""
from __future__ import annotations

import numpy as np
from discopy import symmetric, monoidal
from discopy.cat import factory

_mode = monoidal.Ty('mode')

class Mode(monoidal.Ty):
    def __init__(self, n: int):
        assert isinstance(n, int)
        self.n = n
        super().__init__(*['mode' for _ in range(n)])

    # def tensor(self, *others: Mode) -> Mode:
    #     for other in others:
    #         if not isinstance(other, monoidal.Ty):
    #             return NotImplemented  # This allows whiskering on the left.
    #         assert self.factory == other.factory
    #     return self.factory(self.n + sum(other.n for other in others))

@factory
class Diagram(symmetric.Diagram):

    # def to_optyx_diagram(self) -> Diagram:
    #     from optyx import zx, zw, qpath, circuit
    #     if isinstance(self, qpath.Diagram):
    #         return Diagram(MODE ** len(self.dom), MODE ** len(self.cod), self.inside)

    def tensor(self, other: Diagram = None, *others: Diagram) -> Diagram:
        """
        >>> from optyx.qpath import Split
        >>> from optyx.zw import Z
        >>> s = Split()
        >>> box = Z(lambda x: x, 2, 1)
        >>> box @ s
        optyx.diagram.Diagram(inside=(monoidal.Layer(monoidal.PRO(0), Z(0, 1, ...), monoidal.PRO(1)), monoidal.Layer(monoidal.PRO(1), optyx.qpath.Split('Split()', monoidal.PRO(1), monoidal.PRO(2)), monoidal.PRO(0))), dom=monoidal.PRO(3), cod=monoidal.PRO(3))
        >>> box @ box
        optyx.zw.Diagram(inside=(monoidal.Layer(monoidal.PRO(0), Z(0, 1, ...), monoidal.PRO(2)), monoidal.Layer(monoidal.PRO(1), Z(0, 1, ...), monoidal.PRO(0))), dom=monoidal.PRO(4), cod=monoidal.PRO(2))

        """
        if other is not None and all(o.factory == self.factory for o in (other, ) + others):
            return super().tensor(other, *others)
        left = Diagram(self.inside, self.dom, self.cod)
        right = Diagram(other.inside, other.dom, other.cod)
        rest = [Diagram(o.inside, o.dom, o.cod) for o in others]
        return left.tensor(right, *rest)

    # def to_tensor(self, dims: list[int]):
    #     pass
    #
    # def conjugate(self):
    #     pass
    #
    # def eval(self):
    #     pass

class Box(symmetric.Box):
    pass
    # def to_array(self, dims: list[int]) -> np.ndarray:
    #     pass
    #
    # def to_tensor(self):
    #     pass


class Sum(symmetric.Sum):
    pass


class Swap(symmetric.Swap):
    pass

