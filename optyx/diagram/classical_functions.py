import numpy as np
from typing import Callable, List
import optyx.diagram.optyx as optyx
from discopy import tensor
from discopy.frobenius import Dim

class ClassicalFunctionBox(optyx.Box):

    def __init__(
        self,
        function: Callable[[List[int]], List[int]],
        dom: optyx.Mode | optyx.Bit,
        cod: optyx.Mode | optyx.Bit,
        is_dagger: bool = False,
    ):

        assert all(
            d == cod[0] for d in cod
        ), "cod must be either all Mode(n) or all Bit(n)"
        assert all(
            d == dom[0] for d in dom
        ), "dom must be either all Mode(n) or all Bit(n)"

        super().__init__("F", dom, cod)

        self.function = function
        self.input_size = len(dom)
        self.output_size = len(cod)
        self.is_dagger = is_dagger

    def to_zw(self):
        return self

    def truncation(
        self, input_dims: List[int], output_dims: List[int]
    ) -> tensor.Box:

        if self.is_dagger:
            input_dims, output_dims = output_dims, input_dims

        array = np.zeros((*input_dims, *output_dims), dtype=complex)
        input_ranges = [range(i) for i in input_dims]
        input_combinations = np.array(np.meshgrid(*input_ranges)).T.reshape(
            -1, len(input_dims)
        )

        outputs = [
            (i, self.function(i))
            for i in input_combinations
            if self.function(i) != 0
        ]

        full_indices = np.array(
            [tuple(input_) + tuple(output) for input_, output in outputs]
        )
        array[tuple(full_indices.T)] = 1

        input_dims = [int(d) for d in input_dims]
        output_dims = [int(d) for d in output_dims]

        if self.is_dagger:
            return tensor.Box(
                self.name, Dim(*input_dims), Dim(*output_dims), array
            ).dagger()

        return tensor.Box(
            self.name, Dim(*input_dims), Dim(*output_dims), array
        )

    def determine_output_dimensions(self, input_dims: List[int]) -> List[int]:
        if self.cod == optyx.Mode(self.output_size):
            return [optyx.MAX_DIM] * self.output_size

        elif self.cod == optyx.Bit(self.output_size):
            return [2] * self.output_size

        else:
            return [int(max(input_dims))] * self.output_size

    def dagger(self):
        return ClassicalFunctionBox(
            self.function, self.cod, self.dom, not self.is_dagger
        )


class BinaryMatrixBox(optyx.Box):
    """
    Represents a linear transformation over
    GF(2) using matrix multiplication.

    Example
    -------
    >>> from optyx.zx import X
    >>> from optyx.optyx import Scalar
    >>> xor = X(2, 1) @ Scalar(np.sqrt(2))
    >>> matrix = [[1, 1]]
    >>> m_res = BinaryMatrixBox(matrix).to_tensor().eval().array
    >>> xor_res = xor.to_zw().to_tensor().eval().array
    >>> assert np.allclose(m_res, xor_res)

    """

    def __init__(self, matrix: np.ndarray, is_dagger: bool = False):

        matrix = np.array(matrix)
        if len(matrix.shape) == 1:
            matrix = matrix.reshape(1, -1)

        cod = optyx.Bit(len(matrix[0])) if is_dagger else optyx.Bit(len(matrix))
        dom = optyx.Bit(len(matrix)) if is_dagger else optyx.Bit(len(matrix[0]))

        super().__init__("LogicalMatrix", dom, cod)

        self.matrix = matrix
        self.is_dagger = is_dagger

    def to_zw(self):
        return self

    def truncation(
        self, input_dims: List[int], output_dims: List[int]
    ) -> tensor.Box:

        if self.is_dagger:
            input_dims, output_dims = output_dims, input_dims

        def f(x):
            if not isinstance(x, np.ndarray):
                x = np.array(x, dtype=np.uint8)
            if len(x.shape) == 1:
                x = x.reshape(-1, 1)
            A = np.array(self.matrix, dtype=np.uint8)

            return list(((A @ x) % 2).reshape(1, -1)[0])

        classical_function = ClassicalFunctionBox(f, self.dom, self.cod)

        if self.is_dagger:
            return classical_function.truncation(
                input_dims, output_dims
            ).dagger()
        return classical_function.truncation(input_dims, output_dims)

    def determine_output_dimensions(self,
                                    input_dims: List[int]) -> List[int]:
        return ClassicalFunctionBox(
            None, self.dom, self.cod
        ).determine_output_dimensions(input_dims)

    def dagger(self):
        return BinaryMatrixBox(self.matrix, not self.is_dagger)