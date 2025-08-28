from optyx.core.zw import *
from optyx.core.diagram import mode, Swap
import numpy as np
import pytest

@pytest.mark.skip(reason="Helper function for testing")
def kron_truncation_swap(input_dims: list[int]) -> np.ndarray[complex]:
    input_total_dim = (input_dims[0]) * (input_dims[1])   # Total input dimension

    swap = np.zeros((input_total_dim, input_total_dim), dtype=complex)

    for i in range(input_dims[0]):
        for j in range(input_dims[1]):
            i_ = np.eye(input_dims[0])[i]
            j_ = np.eye(input_dims[1])[j]

            ij = np.kron(i_, j_)
            ji = np.kron(j_, i_)
            swap += np.outer(ij, ji)
    return swap

test_pairs = [(i, j) for i in range(2, 10) for j in range(2, 10, 2)]

@pytest.mark.parametrize("i, j", test_pairs)
def test_swap(i, j):
    assert np.allclose(kron_truncation_swap([i, j]).flatten(), Swap(mode, mode).to_tensor([i, j]).eval().array.flatten())

class W_test(ZWBox):
    """
    W node from the infinite ZW calculus - one input and n outputs
    """

    draw_as_spider = False
    color = "white"

    def __init__(self, n_legs: int, is_dagger: bool = False):
        dom = diagram.Mode(n_legs) if is_dagger else diagram.Mode(1)
        cod = diagram.Mode(1) if is_dagger else diagram.Mode(n_legs)
        super().__init__("W", dom, cod)
        self.n_legs = n_legs
        self.is_dagger = is_dagger
        self.shape = "triangle_up" if not is_dagger else "triangle_down"

    def conjugate(self):
        return self

    def truncation(
        self, input_dims: list[int] = None, output_dims: list[int] = None
    ) -> tensor.Box:
        """Create a truncated array like in 2306.02114."""
        if input_dims is None:
            raise ValueError("Input dimensions must be provided.")

        if output_dims is None:
            output_dims = self.determine_output_dimensions(input_dims)

        max_dim = output_dims[0] if self.is_dagger else input_dims[0]
        shape = (
            (np.prod(input_dims), output_dims[0])
            if self.is_dagger
            else (np.prod(output_dims), input_dims[0])
        )
        result_matrix = np.zeros(shape, dtype=complex)

        for n in range(max_dim):
            allowed_configs = occupation_numbers(n, self.n_legs)
            if self.is_dagger:
                allowed_configs = filter_occupation_numbers(
                    allowed_configs, np.array(input_dims) - 1
                )

            for config in allowed_configs:
                coef = np.sqrt(multinomial(config))

                if self.is_dagger:
                    row_idx = sum(
                        s * np.prod(input_dims[i + 1:], dtype=int)
                        for i, s in enumerate(config)
                    )
                else:
                    row_idx = sum(
                        s * np.prod(output_dims[i + 1:], dtype=int)
                        for i, s in enumerate(config)
                    )

                result_matrix[row_idx, n] += coef

        out_dims = Dim(*[int(i) for i in output_dims])
        in_dims = Dim(*[int(i) for i in input_dims])

        if self.is_dagger:
            return tensor.Box(self.name, in_dims, out_dims, result_matrix)
        return tensor.Box(
            self.name, in_dims, out_dims, result_matrix.conj().T
        )

    def determine_output_dimensions(self, input_dims: list[int]) -> list[int]:
        """Determine the output dimensions based on the input dimensions."""
        if self.is_dagger:
            dims_out = np.sum(np.array(input_dims) - 1) + 1
            return [dims_out]
        return [input_dims[0] for _ in range(len(self.cod))]

    def to_path(self, dtype=complex) -> Matrix:
        array = np.ones(self.n_legs)
        if self.is_dagger:
            return Matrix[dtype](array, self.n_legs, 1)
        return Matrix[dtype](array, 1, int(self.n_legs))

    def dagger(self) -> diagram.Diagram:
        return W(self.n_legs, not self.is_dagger)

    def __repr__(self):
        attr = ", dagger=True" if self.is_dagger else ""
        return f"W({self.n_legs}{attr})"

    def __eq__(self, other: "W") -> bool:
        if not isinstance(other, W):
            return False
        return (self.n_legs, self.is_dagger) == (
            other.n_legs,
            other.is_dagger,
        )

test_pairs = [[i] for i in range(1, 10, 2)]
test_pairs += [[i, j] for i in range(1, 10, 2) for j in range(1, 10, 2)]
test_pairs += [[i, j, k] for i in range(1, 10, 2) for j in range(1, 10, 2) for k in range(1, 10, 2)]

@pytest.mark.parametrize("comb", test_pairs)
def test_W(comb):
    assert np.allclose(W_test(len(comb)).to_tensor(input_dims=[comb[0]]).eval().array,
                       W(len(comb)).to_tensor(input_dims=[comb[0]]).eval().array)

@pytest.mark.parametrize("comb", test_pairs)
def test_W_dagger(comb):
    d = Id(0)
    for i in comb:
        d = d @ Create(i)
    assert np.allclose(
        (d >> W(len(comb)).dagger()).to_tensor().eval().array,
        (d >> W_test(len(comb)).dagger()).to_tensor().eval().array
    )
@pytest.mark.skip(reason="Helper function for testing")
def kron_truncation_Z(diagram, input_dims: list[int]) -> np.ndarray[complex]:
    max_occupation_num = min(input_dims)

    result_matrix = np.zeros(
        (max_occupation_num**diagram.legs_out, np.prod(np.array(input_dims))), dtype=complex
    )

    if diagram.legs_in == 0 and diagram.legs_out == 0:
        if not isinstance(diagram.amplitudes, IndexableAmplitudes):
            return np.array([diagram.amplitudes], dtype=complex)
        return np.array([diagram.amplitudes[0]], dtype=complex)

    for i in range(max_occupation_num):
        vec_in = 1
        for j in range(diagram.legs_in):
            vec_in = np.kron(vec_in, np.eye(input_dims[j])[i])
        vec_out = 1
        for j in range(diagram.legs_out):
            vec_out = np.kron(vec_out, np.eye(max_occupation_num)[i])
        if not isinstance(diagram.amplitudes, IndexableAmplitudes):
            if i >= len(diagram.amplitudes):
                pass
            else:
                result_matrix += np.outer(vec_out, vec_in) * diagram.amplitudes[i]
        else:
            result_matrix += np.outer(vec_out, vec_in) * diagram.amplitudes[i]
    return result_matrix
