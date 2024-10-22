"""
ZW diagrams and their mapping to :class:`tensor.Diagram`.

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    W
    Z

.. admonition:: Functions

    .. autosummary::
        :template: function.rst
        :nosignatures:
        :toctree:

        tn_output_2_perceval_output


Example
-------
We check the axioms of the ZW calculus.

W commutativity

>>> from optyx.utils import compare_arrays_of_different_sizes
>>> bSym_l = W(2)
>>> bSym_r = W(2) >> swap
>>> assert compare_arrays_of_different_sizes(\\
...             bSym_l.to_tensor().eval().array,\\
...             bSym_r.to_tensor().eval().array)

W associativity

>>> bAso_l = W(2) >> W(2) @ Id(1)
>>> bAso_r = W(2) >> Id(1) @ W(2)
>>> assert compare_arrays_of_different_sizes(\\
...             bAso_l.to_tensor().eval().array,\\
...             bAso_r.to_tensor().eval().array)

W unit

>>> bId_l = W(2) >> Select(0) @ Id(1)
>>> bId_r = W(2) >> Id(1) @ Select(0)
>>> assert compare_arrays_of_different_sizes(\\
...             bId_l.to_tensor().eval().array,\\
...             bId_r.to_tensor().eval().array)

W bialgebra

>>> bBa_l = W(2) @ W(2) >>\\
...             Id(1) @ swap @ Id(1) >>\\
...             W(2).dagger() @ W(2).dagger()
>>> bBa_r = W(2).dagger() >> W(2)
>>> assert compare_arrays_of_different_sizes(\\
...             bBa_l.to_tensor().eval().array,\\
...             bBa_r.to_tensor().eval().array)

ZW bialgebra

>>> from math import factorial
>>> N = [float(np.sqrt(factorial(i))) for i in range(5)]
>>> frac_N = [float(1/np.sqrt(factorial(i))) for i in range(5)]
>>> bZBA_l = Z(N, 1, 2) @ Z(N, 1, 2) >>\\
...             Id(1) @ swap @ Id(1) >>\\
...             W(2).dagger() @ W(2).dagger() >>\\
...             Id(1) @ Z(frac_N, 1, 1)
>>> bZBA_r = W(2).dagger() >> Z([1, 1, 1, 1, 1], 1, 2)
>>> assert compare_arrays_of_different_sizes(\\
...             bZBA_l.to_tensor().eval().array,\\
...             bZBA_r.to_tensor().eval().array)

Z copies n-photon states

>>> K0_infty_l = Create(4) >> Z([1, 1, 1, 1, 1], 1, 2)
>>> K0_infty_r = Create(4) @ Create(4)
>>> assert compare_arrays_of_different_sizes(\\
...             K0_infty_l.to_tensor().eval().array,\\
...             K0_infty_r.to_tensor().eval().array)


Check the Hong-Ou-Mandel interference

>>> Zb_i = Z(np.array([1, 1j/(np.sqrt(2))]), 1, 1)
>>> Zb_1 = Z(np.array([1, 1/(np.sqrt(2))]), 1, 1)
>>> beam_splitter = W(2) @ W(2) >> \\
...               Zb_i @ Zb_1 @ Zb_1 @ Zb_i >> \\
...               Id(1) @ swap @ Id(1) >> \\
...               W(2).dagger() @ W(2).dagger()
>>> Hong_Ou_Mandel = Create(1) @ Create(1) >> \\
...                beam_splitter >> \\
...                Select(1) @ Select(1)
>>> assert compare_arrays_of_different_sizes(\\
...             Hong_Ou_Mandel.to_tensor().eval().array,\\
...             np.array([0]))

Check Lemma B7 from 2306.02114

>>> lemma_B7_l = Id(1) @ W(2).dagger() >> \\
...             Z(lambda i: 1, 2, 0)
>>> lemma_B7_r = W(2) @ Id(2) >>\\
...             Id(1) @ Id(1) @ swap >>\\
...             Id(1) @ swap @ Id(1) >>\\
...             Z(lambda i: 1, 2, 0) @ Z(lambda i: 1, 2, 0)
>>> assert compare_arrays_of_different_sizes(\\
...             lemma_B7_l.to_tensor().eval().array,\\
...             lemma_B7_r.to_tensor().eval().array)
"""

from typing import Union
import numpy as np
from discopy import tensor, frobenius
from optyx.optyx import Box, Diagram, Mode, Dim, Permutation, mode
from optyx.utils import occupation_numbers, multinomial, get_index_from_list

swap = Permutation(mode @ mode, [1, 0])
Id = lambda n: Diagram.id(mode ** n)

class W(Box):
    """
    W gate from the ZW calculus - one input and n outputs
    """

    draw_as_spider = False
    color = "white"

    def __init__(self, n_legs: int, is_dagger: bool = False):
        dom = Mode(n_legs) if is_dagger else Mode(1)
        cod = Mode(1) if is_dagger else Mode(n_legs)
        super().__init__("W", dom, cod)
        self.n_legs = n_legs
        self.is_dagger = is_dagger
        self.shape = "triangle_up" if not is_dagger else "triangle_down"

    def truncated_array(self, input_dims: list[int]) -> np.ndarray[complex]:
        """Create an array like in 2306.02114"""

        if self.is_dagger:
            max_dimension = np.sum(np.array(input_dims) - 1) + 1

            total_map = np.zeros(
                (np.prod(np.array(input_dims)), max_dimension),
                dtype=complex,
            )

            for n in range(max_dimension):

                # get all allowed occupation configurations for n photons
                # (symmetric Fock space basis states)
                allowed_occupation_configurations = occupation_numbers(
                    n, self.n_legs
                )

                allowed_occupation_configurations = filter_occupation_numbers(
                    allowed_occupation_configurations, np.array(input_dims) - 1
                )

                for configuration in allowed_occupation_configurations:

                    # get the coefficient for the configuration
                    coef = np.sqrt(multinomial(configuration))

                    # find idx of the matrix where to put the coefficient
                    row_index = 0

                    for i, s_i in enumerate(configuration):
                        row_index += s_i * (
                            np.prod(np.array(input_dims[i + 1:]), dtype=int)
                        )
                    col_index = n

                    total_map[row_index, col_index] += coef
            return total_map

        max_dimension = input_dims[0]

        total_map = np.zeros(
            ((max_dimension) ** self.n_legs, max_dimension),
            dtype=complex,
        )

        for n in range(max_dimension):

            # get all allowed occupation configurations for n photons
            # (symmetric Fock space basis states)
            allowed_occupation_configurations = occupation_numbers(
                n, self.n_legs
            )

            for configuration in allowed_occupation_configurations:

                # get the coefficient for the configuration
                coef = np.sqrt(multinomial(configuration))

                # find idx of the matrix where to put the coefficient
                row_index = 0

                for i, s_i in enumerate(configuration):
                    row_index += s_i * (max_dimension) ** (
                        self.n_legs - i - 1
                    )
                col_index = n
                total_map[row_index, col_index] += coef

        return total_map.conj().T

    def determine_dimensions(self, input_dims: list[int]) -> list[int]:
        """Determine the output dimensions based on the input dimensions."""
        if self.is_dagger:
            dims_out = np.sum(np.array(input_dims) - 1) + 1
            return [dims_out for _ in range(len(self.cod))]
        return [input_dims[0] for _ in range(len(self.cod))]

    def dagger(self) -> Diagram:
        return W(self.n_legs, not self.is_dagger)

    def __repr__(self):
        attr = ", dagger=True" if self.is_dagger else ""
        return f"W({self.n_legs}{attr})"

    def __eq__(self, other: "W") -> bool:
        if not isinstance(other, W):
            return False
        return (self.n_legs, self.is_dagger) == (other.n_legs, other.is_dagger)


class IndexableAmplitudes:
    """Since the amplitudes can be an infinite list,
    we can specify them as a function instead of an explicit list.
    The class is a wrapper for the function which allows to
    index the function as if it was a list.

    >>> f = lambda i: i
    >>> amplitudes = IndexableAmplitudes(f)
    >>> assert amplitudes[0] == 0
    """

    def __init__(self, func):
        self.func = func
        self.conj = False
        self.name = "Z(func)"

    def __getitem__(self, i):
        if not self.conj:
            return self.func(i)
        return np.conj(self.func(i))

    def __str__(self) -> str:
        return "function"

    def conjugate(self):
        """Conjugate the amplitudes"""
        self.conj = not self.conj
        return self

    def __eq__(self, other: "IndexableAmplitudes") -> bool:
        if not isinstance(other, IndexableAmplitudes):
            return False
        return self.func.__code__.co_code == other.func.__code__.co_code


class Z(Box):
    """
    Z gate from the ZW calculus.
    """

    def __init__(
        self,
        amplitudes: Union[np.ndarray, list, callable, IndexableAmplitudes],
        legs_in: int,
        legs_out: int,
    ):
        # if amplitudes are a function then make it indexable, "conjugable"
        if callable(amplitudes):
            self.amplitudes = IndexableAmplitudes(amplitudes)
        else:
            self.amplitudes = amplitudes
        super().__init__(f"Z{self.amplitudes}", Mode(legs_in), Mode(legs_out))
        self.name = self.__repr__()
        self.legs_in = legs_in
        self.legs_out = legs_out
        self.drawing_name = self.__repr__()
        self.shape = "rectangle"
        self.color = "green"

    def truncated_array(self, input_dims: list[int]) -> np.ndarray[complex]:
        """Create an array like in 2306.02114"""
        if len(input_dims) == 0:
            max_dimension = 2
        else:
            max_dimension = min(input_dims)

        result_matrix = np.zeros(
            (
                int(max_dimension**self.legs_out),
                int(np.prod(np.array(input_dims))),
            ),
            dtype=complex,
        )

        if self.legs_in == 0 and self.legs_out == 0:
            if not isinstance(self.amplitudes, IndexableAmplitudes):
                return np.array([self.amplitudes], dtype=complex)
            return np.array([self.amplitudes[0]], dtype=complex)

        for i in range(max_dimension):
            row_index = 0
            col_index = 0

            for j in range(self.legs_out):
                row_index += i * max_dimension ** (self.legs_out - j - 1)
            for j in range(self.legs_in):
                col_index += i * (
                    np.prod(np.array(input_dims[j + 1:]), dtype=int)
                )
            if not isinstance(self.amplitudes, IndexableAmplitudes):
                if i >= len(self.amplitudes):
                    result_matrix[row_index, col_index] = 0
                else:
                    result_matrix[row_index, col_index] = self.amplitudes[i]
            else:
                result_matrix[row_index, col_index] = self.amplitudes[i]

        return result_matrix

    def determine_dimensions(self, input_dims: list[int]) -> list[int]:
        """Determine the output dimensions based on the input dimensions."""
        return [min(input_dims) for _ in range(len(self.cod))]

    def __repr__(self):
        if isinstance(self.amplitudes, IndexableAmplitudes):
            s = ", ".join(str(self.amplitudes[i]) for i in range(2)) + ", ..."
        else:
            s = ", ".join(str(a) for a in self.amplitudes)
        return f"Z({s})"

    __str__ = __repr__

    def __eq__(self, other: "Z") -> bool:
        if (
            not isinstance(other, Z)
            or self.legs_in != other.legs_in
            or self.legs_out != other.legs_out
        ):
            return False
        if isinstance(self.amplitudes, IndexableAmplitudes):
            return self.amplitudes == other.amplitudes
        return np.allclose(self.amplitudes, other.amplitudes)

    def dagger(self) -> Diagram:
        return Z(np.conj(self.amplitudes), self.legs_out, self.legs_in)


class Create(Box):
    """
    n-photon initialisation map from the ZW calculus.
    """

    draw_as_spider = True
    color = "blue"

    def __init__(self, n_photons: int):
        super().__init__(str(n_photons), Mode(0), Mode(1))
        self.n_photons = n_photons

    def truncated_array(self, _) -> np.ndarray[complex]:
        """Create an array like in 2306.02114"""
        if self.n_photons == 0:
            dims_out = 2
        else:
            dims_out = self.n_photons + 1
        result_matrix = np.zeros((dims_out, 1), dtype=complex)
        result_matrix[self.n_photons, 0] = 1.0
        return result_matrix

    def determine_dimensions(self, _: list[int]) -> list[int]:
        """Determine the output dimensions based on the input dimensions."""
        return [
            2 if self.n_photons == 0 else self.n_photons + 1
        ]

    def __repr__(self):
        return f"Create({self.n_photons})"

    def __eq__(self, other: "Create") -> bool:
        if not isinstance(other, Create):
            return False
        return self.n_photons == other.n_photons

    def dagger(self) -> Diagram:
        return Select(self.n_photons)


class Select(Box):
    """
    n-photon postselection map from the ZW calculus.
    """

    draw_as_spider = True
    color = "blue"

    def __init__(self, n_photons: int):
        super().__init__(str(n_photons), Mode(1), Mode(0))
        self.n_photons = n_photons

    def __repr__(self):
        return f"Select({self.n_photons})"

    def __eq__(self, other: "Select") -> bool:
        if not isinstance(other, Select):
            return False
        return self.n_photons == other.n_photons

    def truncated_array(self, input_dims: list) -> np.ndarray[complex]:
        """Create an array like in 2306.02114"""

        result_matrix = np.zeros((1, input_dims[0]), dtype=complex)
        result_matrix[0, self.n_photons] = 1.0
        return result_matrix

    def determine_dimensions(self, _: list[int]) -> list[int]:
        """Determine the output dimensions based on the input dimensions."""
        return []

    def dagger(self) -> Diagram:
        return Create(self.n_photons)


def tn_output_2_perceval_output(
    tn_output: list | np.ndarray, diagram: Diagram, n_extra_photons: int = 0
) -> np.ndarray:
    """Convert the prob output of the tensor
    network to the perceval prob output"""

    n_selections, n_creations = calculate_num_creations_selections(diagram)

    wires_out = len(diagram.cod)

    n_photons_out = n_extra_photons - n_selections + n_creations

    cod = list(diagram.to_tensor().cod.inside)

    idxs = list(occupation_numbers(n_photons_out, wires_out))

    ix = [get_index_from_list(i, cod) for i in idxs]
    res_ = []
    for i in ix:
        if i < len(tn_output):
            res_.append(tn_output[i])
        else:
            res_.append(0.0)

    return np.array(res_)


def calculate_num_creations_selections(diagram: Diagram) -> tuple:
    """Calculate the number of creations and selections in the diagram"""
    terms = list(diagram)

    n_selections = 0
    n_creations = 0

    if isinstance(terms[0], frobenius.Layer):
        for box, _ in zip(diagram.boxes, diagram.offsets):
            if isinstance(box, Create):
                n_creations += box.n_photons
            elif isinstance(box, Select):
                n_selections += box.n_photons

    else:
        arr_selections_creations = []
        for term in diagram:
            arr_selections_creations.append(
                term.calculate_num_creations_selections(diagram)
            )
        n_selections = max(i[0] for i in arr_selections_creations)
        n_creations = max(i[1] for i in arr_selections_creations)
    return n_selections, n_creations


def filter_occupation_numbers(
    allowed_occupation_configurations: list[list[int]], input_dims: list[int]
) -> list[list[int]]:
    """Filter the occupation numbers based on the input dimensions"""
    return [
        config
        for config in allowed_occupation_configurations
        if all(list(config[i] <= input_dims[i] for i in
                    range(len(input_dims))))
    ]

