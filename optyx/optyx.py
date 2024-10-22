from __future__ import annotations

import numpy as np
from discopy import symmetric, frobenius, tensor
from discopy.cat import factory
from discopy.frobenius import Dim

class Ty(frobenius.Ty):
    pass

mode = Ty('mode')
bit = Ty('bit')

class Mode(Ty):
    def __init__(self, n):
        self.n = n
        super().__init__(*['mode' for _ in range(n)])


@factory
class Diagram(frobenius.Diagram):
    """ Optyx diagram """

    grad = tensor.Diagram.grad

    def to_path(self, dtype: type = complex) -> Matrix:
        """Returns the :class:`Matrix` normal form of a :class:`Diagram`."""
        from optyx import qpath
        return symmetric.Functor(
            ob=len,
            ar=lambda f: f.to_path(dtype),
            cod=symmetric.Category(int, qpath.Matrix[dtype]),
        )(self)

    def to_tensor(self, input_dims: list = None) -> tensor.Diagram:
        """Returns a a tensor.Diagram for evaluation"""

        def f_ob(dims: np.ndarray | list) -> Dim:
            """Converts a list of dimensions to a Dim object"""
            return Dim(*[int(i) for i in dims])

        def f_ar(box: Box, dims_in: list, dims_out: list) -> tensor.Box:
            """Converts a box to a tensor.Box object
            with the correct dimensions and array"""
            arr = box.truncated_array(np.array(dims_in))
            return tensor.Box(
                box.name, f_ob(dims_in), f_ob(dims_out), arr
            )

        if input_dims is None:
            layer_dims = [2 for _ in range(len(self.dom))]
        else:
            layer_dims = input_dims

        right_dim = len(self.dom)

        for i, (box, off) in enumerate(zip(self.boxes, self.offsets)):
            dims_in = layer_dims[off: off + len(box.dom)]

            dims_out = box.determine_dimensions(
                dims_in
            )

            left = Dim()
            if off > 0:
                left = f_ob(layer_dims[0:off])
            right = Dim()
            if off + len(box.dom) < right_dim:
                right = f_ob(
                    layer_dims[off + len(box.dom): right_dim]
                )

            cod_right_dim = right_dim - len(box.dom) + len(box.cod)
            cod_layer_dims = (
                layer_dims[0:off]
                + dims_out
                + layer_dims[off + len(box.dom):]
            )

            diagram_ = left @ f_ar(box, dims_in, dims_out) @ right

            if i == 0:
                diagram = diagram_
            else:
                diagram = diagram >> diagram_

            right_dim = cod_right_dim
            layer_dims = cod_layer_dims
        return diagram


class Box(frobenius.Box, Diagram):
    """A box in an optyx diagram"""

    __ambiguous_inheritance__ = (frobenius.Box,)

    def lambdify(self, *symbols, **kwargs):
        # Non-symbolic gates can be returned directly
        return lambda *xs: self

    def subs(self, *args) -> Diagram:
        syms, exprs = zip(*args)
        return self.lambdify(*syms)(*exprs)


class Sum(symmetric.Sum, Box):
    """
    Formal sum of optyx diagrams
    """

    __ambiguous_inheritance__ = (symmetric.Sum,)

    def to_path(self, dtype=complex):
        return sum(term.to_path(dtype) for term in self.terms)

    def grad(self, var, **params):
        """Gradient with respect to :code:`var`."""
        if var not in self.free_symbols:
            return self.sum_factory((), self.dom, self.cod)
        return sum(term.grad(var, **params) for term in self.terms)


class Swap(frobenius.Swap, Box):
    """Swap in optyx diagram"""

    def to_path(self, dtype=complex) -> Matrix:
        return Matrix([0, 1, 1, 0], 2, 2)


class Permutation(Box):
    """Permute wires in an optyx diagram"""

    def __init__(self, dom: Ty,
                 permutation: list[int],
                 is_dagger: bool = False):
        """
        Args:
            dom: The input type
            permutation: List of indices representing the permutation.
                         Each entry indicates where the
                         corresponding input goes in the output.
        """
        assert len(permutation) == len(dom)

        cod = Ty.tensor(*[dom[i] for i in permutation])
        self.is_dagger = is_dagger
        super().__init__(str(permutation), dom, cod)
        self.permutation = permutation

    def truncated_array(self, input_dims: list[int]) -> np.ndarray:
        """Create an array that permutes the occupation
        numbers based on the input dimensions."""

        input_total_dim = int(np.prod(input_dims))

        perm_matrix = np.zeros((input_total_dim, input_total_dim),
                               dtype=complex)

        output_dims = [input_dims[self.permutation[i]]
                       for i in range(len(self.permutation))]

        for input_index in np.ndindex(*input_dims):
            permuted_index = tuple(input_index[self.permutation[i]]
                                   for i in range(len(self.permutation)))
            input_flat_index = np.ravel_multi_index(input_index, input_dims)
            permuted_flat_index = np.ravel_multi_index(permuted_index,
                                                       output_dims)
            perm_matrix[permuted_flat_index, input_flat_index] = 1

        return perm_matrix.T

    def determine_dimensions(self, input_dims: list[int]) -> list[int]:
        """Determine the output dimensions based on the permutation."""
        return [input_dims[i] for i in self.permutation]

    def dagger(self) -> Diagram:
        n = len(self.permutation)
        inverse_permutation = [0] * n
        for i, j in enumerate(self.permutation):
            inverse_permutation[j] = i

        return Swap(int(np.sum(self.dom.inside)),
                    int(np.sum(self.cod.inside)),
                    inverse_permutation, 
                    not self.is_dagger)

Diagram.swap_factory = Swap
Diagram.swap = Swap
Diagram.sum_factory = Sum
Id = Diagram.id