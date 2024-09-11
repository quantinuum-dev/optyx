"""
ZW diagrams and their mapping to :class:`tensor.Diagram`.

.. admonition:: Functions
    .. autosummary::
        :template: function.rst
        :nosignatures:
        :toctree:
    
Example
-------
>>> lemma_B7_l = Id(1) @ W(2).dagger() >> \\
...             Z(lambda i: 1, 2, 0)
>>> lemma_B7_r = W(2) @ Id(2) >>\\
...             Id(1) @ Id(1) @ Swap() >>\\
...             Id(1) @ Swap() @ Id(1) >>\\
...             Z(lambda i: 1, 2, 0) @ Z(lambda i: 1, 2, 0)
>>> assert compare_arrays_of_different_sizes(\\
...             lemma_B7_l.to_tensor(print_max_occupation_number=False)\\
...                     .eval().array,\\
...             lemma_B7_r.to_tensor(print_max_occupation_number=False)\\
...                     .eval().array)

>>> Zb_i = Z(np.array([1, 1j/(np.sqrt(2))]), 1, 1)
>>> Zb_1 = Z(np.array([1, 1/(np.sqrt(2))]), 1, 1)
>>> beam_splitter = W(2) @ W(2) >> \\
...               Zb_i @ Zb_1 @ Zb_1 @ Zb_i >> \\
...               Id(1) @ Swap() @ Id(1) >> \\
...               W(2).dagger() @ W(2).dagger()
>>> Hong_Ou_Mandel = Create(1) @ Create(1) >> \\
...                beam_splitter >> \\
...                Select(1) @ Select(1) 
>>> assert compare_arrays_of_different_sizes(\\
...             Hong_Ou_Mandel.to_tensor(print_max_occupation_number=False)\\
...                     .eval().array,\\
...             np.array([0]))

Axioms

>>> bSym_l = W(2)
>>> bSym_r = W(2) >> Swap()
>>> assert compare_arrays_of_different_sizes(\\
...             bSym_l.to_tensor(print_max_occupation_number=False)\\
...                     .eval().array,\\
...             bSym_r.to_tensor(print_max_occupation_number=False)\\
...                     .eval().array) 


>>> bAso_l = W(2) >> W(2) @ Id(1)
>>> bAso_r = W(2) >> Id(1) @ W(2)
>>> assert compare_arrays_of_different_sizes(\\
...             bAso_l.to_tensor(print_max_occupation_number=False)\\
...                     .eval().array,\\
...             bAso_r.to_tensor(print_max_occupation_number=False)\\
...                     .eval().array)


>>> bBa_l = ProjectionMap(2) >> W(2) @ W(2) >>\\
...             Id(1) @ Swap() @ Id(1) >>\\
...             W(2).dagger() @ W(2).dagger()
>>> bBa_r = W(2).dagger() >> W(2)
>>> assert compare_arrays_of_different_sizes(\\
...             bBa_l.to_tensor(print_max_occupation_number=False)\\
...                     .eval().array,\\ 
...             bBa_r.to_tensor(print_max_occupation_number=False)\\
...                     .eval().array)


>>> bId_l = W(2) >> Select(0) @ Id(1)
>>> bId_r = Id(1)
>>> assert compare_arrays_of_different_sizes(\\
...             bId_l.to_tensor(print_max_occupation_number=False)\\
...                     .eval().array,\\
...             bId_r.to_tensor(print_max_occupation_number=False)\\
...                     .eval().array)


>>> from math import factorial
>>> N = [float(np.sqrt(factorial(i))) for i in range(5)]
>>> frac_N = [float(1/np.sqrt(factorial(i))) for i in range(5)]
>>> bZBA_l = Z(N, 1, 2) @ Z(N, 1, 2) >>\\
...             Id(1) @ Swap() @ Id(1) >>\\
...             W(2).dagger() @ W(2).dagger() >>\\
...             Id(1) @ Z(frac_N, 1, 1) 
>>> bZBA_r = W(2).dagger() >> Z([1, 1, 1, 1, 1], 1, 2) 
>>> assert compare_arrays_of_different_sizes(\\
...             bZBA_l.to_tensor(print_max_occupation_number=False)\\
...                     .eval().array,\\
...             bZBA_r.to_tensor(print_max_occupation_number=False)\\
...                     .eval().array)

>>> K0_infty_l = Create(4) >> Z([1, 1, 1, 1, 1], 1, 2)
>>> K0_infty_r = Create(4) @ Create(4) 
>>> assert compare_arrays_of_different_sizes(\\
...             K0_infty_l.to_tensor(print_max_occupation_number=False)\\
...                     .eval().array,\\
...             K0_infty_r.to_tensor(print_max_occupation_number=False)\\
...                     .eval().array)
"""

import logging
from typing import Union
import numpy as np
from discopy.rigid import Layer
from discopy.quantum import zx
from discopy.cat import factory, Category
from discopy.rigid import PRO
from discopy import tensor
from discopy.frobenius import Dim, Functor


@factory
class Diagram(zx.Diagram):
    """
    ZW diagram
    """

    def get_max_occupation_num(self):
        """Returns a maximum occupation number to perform
        truncation of the generators: a method from 2306.02114: Lemma 4.1"""
        max_occupation_num = 0

        occupation_numbers = [0]
        terms = list(self)

        # check if the diagram is not a sum of diagrams
        # - otherwise we need to run this function for all terms in the sum
        if isinstance(terms[0], Layer):
            inputs = [0 for _ in range(len(self.dom.inside))]
            scan = [(t, 1) for t in inputs]

            for box, off in zip(self.boxes, self.offsets):
                if isinstance(box, tensor.Swap):
                    scan[off], scan[off + 1] = scan[off + 1], scan[off]
                    continue

                current_occupation_num = 0
                if isinstance(box, Z):
                    previous_occupation_nums = [0]
                    for j in range(len(box.dom)):
                        other_t, _ = scan[off + j]
                        previous_occupation_nums.append(other_t)

                    max_previous_occupation_num = max(previous_occupation_nums)

                    current_occupation_num = max_previous_occupation_num * (
                        box.legs_out + 1
                    )
                elif isinstance(box, Create):
                    current_occupation_num = box.n_photons
                elif isinstance(box, Select):
                    current_occupation_num = box.n_photons

                scan[off:off + len(box.dom)] = [
                    (current_occupation_num, len(box.dom) + ind)
                    for ind in range(len(box.cod))
                ]
                occupation_numbers.append(current_occupation_num)
        else:
            for term in self:
                occupation_numbers.append(term.get_max_occupation_num())

        max_occupation_num = max(occupation_numbers)
        return max_occupation_num

    def to_tensor(
        self, max_occupation_num: int = 2, print_max_occupation_number=True
    ) -> tensor.Diagram:
        """Returns tensor.Diagram based on the ZW diagram for a
        given max_occupation_num which is used to truncate
        the array to be used in the tensor.Diagram.

        :param max_occupation_num: maximum occupation number for
        truncation of the generators
        :param print_max_occupation_number: show the max_occupation_number
        """

        max_occupation_num_ = self.get_max_occupation_num()
        if print_max_occupation_number:
            print(f"detected max_occupation_num: {max_occupation_num_}")

        # a user can specify a custom max_occupation_num
        # throw a warning if the number is too low
        if max_occupation_num < max_occupation_num_:
            if max_occupation_num != 5:
                logging.info(
                    "max_occupation_num is too low, "
                    "setting it to %s", max_occupation_num_
                )
            max_occupation_num = max_occupation_num_

        def f_ob(ob: PRO) -> Dim:
            return Dim(max_occupation_num + 1) ** len(ob)

        def f_ar(box: zx.Box) -> tensor.Box:
            arr = box.truncated_array(max_occupation_num)
            return tensor.Box(box.name, f_ob(box.dom), f_ob(box.cod), arr)

        return Functor(
            ob=f_ob,
            ar=f_ar,
            dom=Category(PRO, self),
            cod=Category(Dim, tensor.Diagram),
        )(self)

    def __add__(self, other: "Diagram") -> "Diagram":
        return Sum([self, other])


class Box(zx.Box, Diagram):
    """A ZW box"""

    __ambiguous_inheritance__ = (zx.Box,)


class Sum(zx.Sum, Box, Diagram):
    """A sum of ZW diagrams"""

    __ambiguous_inheritance__ = (zx.Sum,)


class ProjectionMap(Box):
    """
    ProjectionMap map from the ZXW calculus.
    """

    def __init__(self, wires: int):
        super().__init__("Id", PRO(wires), PRO(wires))
        self.wires = wires

    def truncated_array(self, max_occupation_num: int) -> np.ndarray[complex]:
        """Create a truncated array line in 2306.02114 -
        this array projects the state to the subspace
        of the symmetric Fock space with
        all occupation numbers <= max_occupation_num
        """

        d = max_occupation_num + 1
        arr = np.zeros((d**self.wires, d**self.wires), dtype=complex)

        # for each occupation number
        for i in range(d):

            # get all allowed occupation configurations for n photons
            # (symmetric Fock space basis states)
            combs = get_allowed_occupation_numbers(i, self.wires)
            for comb in combs:
                index = 0

                # for each wire/mode find the index in the matrix
                for j, s_i in enumerate(comb):
                    index += s_i * (max_occupation_num + 1) ** (
                        self.wires - j - 1
                    )

                arr[index, index] += 1
        return arr

    def __repr__(self):
        return f"ProjectionMap({self.wires})"

    def __eq__(self, other: "ProjectionMap") -> bool:
        if not isinstance(other, ProjectionMap):
            return False
        return self.wires == other.wires


class Swap(zx.Swap, Diagram):
    """A ZW Swap"""

    def __init__(self):
        super().__init__(PRO(1), PRO(1))

    # create an array like in 2306.02114
    def truncated_array(self, max_occupation_num: int) -> np.ndarray[complex]:
        """Create an array like in 2306.02114 - this
        array swaps the occupation numbers"""
        max_occupation_num = max_occupation_num + 1

        swap = np.zeros(
            (max_occupation_num**2, max_occupation_num**2), dtype=complex
        )

        for i in range(max_occupation_num):
            for j in range(max_occupation_num):
                swap[
                    i * max_occupation_num + j, j * max_occupation_num + i
                ] = 1

        return swap

    def dagger(self) -> Diagram:
        return Swap()


class Id(Box):
    """An identity wire"""

    def __init__(self, n_wires: int = 0):
        super().__init__("Id", PRO(n_wires), PRO(n_wires))
        self.n_wires = n_wires
        self.draw_as_wires = True

    def truncated_array(self, max_occupation_num: int) -> np.ndarray[complex]:
        """Create an array like in 2306.02114"""
        return np.eye((max_occupation_num + 1) ** self.n_wires, dtype=complex)

    def dagger(self) -> Diagram:
        return self

    def __repr__(self):
        return f"Id({self.n_wires})"

    def __eq__(self, other: "Id") -> bool:
        if not isinstance(other, Id):
            return False
        return self.n_wires == other.n_wires


class W(Box):
    """
    W gate from the ZW calculus - one input and n outputs
    """

    draw_as_spider = False
    color = "white"

    def __init__(self, n_legs: int, is_dagger: bool = False):
        dom = PRO(n_legs) if is_dagger else PRO(1)
        cod = PRO(1) if is_dagger else PRO(n_legs)
        super().__init__("W", dom, cod)
        self.n_legs = n_legs
        self.is_dagger = is_dagger
        self.shape = "triangle_up" if not is_dagger else "triangle_down"

    def truncated_array(self, max_occupation_num: int) -> np.ndarray[complex]:
        """Create an array like in 2306.02114"""

        total_map = np.zeros(
            ((max_occupation_num + 1) ** self.n_legs, max_occupation_num + 1),
            dtype=complex,
        )

        # the default - non dagger - map is 1 -> n wires
        for n in range(max_occupation_num + 1):

            # get all allowed occupation configurations for n photons
            # (symmetric Fock space basis states)
            allowed_occupation_configurations = get_allowed_occupation_numbers(
                n, self.n_legs
            )

            for configuration in allowed_occupation_configurations:

                # get the coefficient for the configuration
                coef = np.sqrt(multinomial(configuration))

                # find idx of the matrix where to put the coefficient
                row_index = 0
                for i, s_i in enumerate(configuration):
                    row_index += s_i * (max_occupation_num + 1) ** (
                        self.n_legs - i - 1
                    )

                col_index = n
                total_map[row_index, col_index] += coef

        if self.is_dagger:
            return total_map
        return total_map.conj().T

    def dagger(self) -> Diagram:
        return W(self.n_legs, not self.is_dagger)

    def __repr__(self):
        attr = ", dagger=True" if self.is_dagger else ""
        return f"W({self.n_legs}{attr})"

    def __eq__(self, other: "W") -> bool:
        if not isinstance(other, W):
            return False
        return (self.n_legs, self.is_dagger) == (other.n_legs, other.is_dagger)


class Z(Box):
    """
    Z gate from the ZW calculus.
    """

    class IndexableAmplitudes:
        """Since the amplitudes can be an infinite list,
        we can specify them as a function instead of a explicit list
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

        def __eq__(self, other: "Z.IndexableAmplitudes") -> bool:
            if not isinstance(other, Z.IndexableAmplitudes):
                return False
            return self.func.__code__.co_code == other.func.__code__.co_code

    def __init__(
        self,
        amplitudes: Union[np.ndarray, list, callable, IndexableAmplitudes],
        legs_in: int,
        legs_out: int,
    ):
        # if amplitudes are a function then make it indexable, "conjugable"
        if callable(amplitudes):
            self.amplitudes = self.IndexableAmplitudes(amplitudes)
        else:
            self.amplitudes = amplitudes
        super().__init__(f"B{self.amplitudes}", PRO(legs_in), PRO(legs_out))
        self.name = self.__repr__()
        self.legs_in = legs_in
        self.legs_out = legs_out
        self.drawing_name = self.__repr__()
        self.shape = "rectangle"
        self.color = "green"

    def truncated_array(self, max_occupation_num: int) -> np.ndarray[complex]:
        """Create an array like in 2306.02114"""

        dim = max_occupation_num + 1
        result_matrix = np.zeros(
            (dim**self.legs_out, dim**self.legs_in), dtype=complex
        )

        if self.legs_in == 0 and self.legs_out == 0:
            if not isinstance(self.amplitudes, self.IndexableAmplitudes):
                return np.array([self.amplitudes], dtype=complex)
            return np.array([self.amplitudes[0]], dtype=complex)

        for i in range(dim):
            row_index = 0
            col_index = 0

            for j in range(self.legs_out):
                row_index += i * dim ** (self.legs_out - j - 1)
            for j in range(self.legs_in):
                col_index += i * dim ** (self.legs_in - j - 1)

            if not isinstance(self.amplitudes, self.IndexableAmplitudes):
                if i >= len(self.amplitudes):
                    result_matrix[row_index, col_index] = 0
                else:
                    result_matrix[row_index, col_index] = self.amplitudes[i]
            else:
                result_matrix[row_index, col_index] = self.amplitudes[i]

        return result_matrix

    def __repr__(self):
        if isinstance(self.amplitudes, self.IndexableAmplitudes):
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
        return self.amplitudes == other.amplitudes

    def dagger(self) -> Diagram:
        return Z(np.conj(self.amplitudes), self.legs_out, self.legs_in)


class Create(Box):
    """
    n-photon initialisation map from the ZW calculus.
    """

    draw_as_spider = True
    color = "blue"

    def __init__(self, n_photons: int):
        super().__init__(str(n_photons), PRO(0), PRO(1))
        self.n_photons = n_photons

    def truncated_array(self, max_occupation_num: int) -> np.ndarray[complex]:
        """Create an array like in 2306.02114"""
        result_matrix = np.zeros((max_occupation_num + 1, 1), dtype=complex)
        result_matrix[self.n_photons, 0] = 1.0
        return result_matrix

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
        super().__init__(str(n_photons), PRO(1), PRO(0))
        self.n_photons = n_photons

    def __repr__(self):
        return f"Select({self.n_photons})"

    def __eq__(self, other: "Select") -> bool:
        if not isinstance(other, Select):
            return False
        return self.n_photons == other.n_photons

    def truncated_array(self, max_occupation_num: int) -> np.ndarray[complex]:
        """Create an array like in 2306.02114"""

        result_matrix = np.zeros((1, max_occupation_num + 1), dtype=complex)
        result_matrix[0, self.n_photons] = 1.0
        return result_matrix

    def dagger(self) -> Diagram:
        return Create(self.n_photons)


def get_allowed_occupation_numbers(n: int, n_legs: int):
    """Returns all allowed occupation numbers for n photons for n_legs modes
    (symmetric Fock space basis states)
    """

    def generate_combinations(n, n_legs, current_list):
        if n_legs == 0:
            if n == 0:
                result.append(current_list)
            return
        for i in range(n + 1):
            generate_combinations(n - i, n_legs - 1, current_list + [i])

    result = []
    generate_combinations(n, n_legs, [])
    return result


def multinomial(lst: list) -> int:
    """Returns the multinomial coefficient for a given list of numbers"""
    # https://stackoverflow.com/questions/46374185/does-python-have-a-function-which-computes-multinomial-coefficients
    res, i = 1, sum(lst)
    i0 = lst.index(max(lst))
    for a in lst[:i0] + lst[i0 + 1:]:
        for j in range(1, a + 1):
            res *= i
            res //= j
            i -= 1
    return res

def compare_arrays_of_different_sizes(array_1, array_2):
    """ZW diagrams which are equal in infinite dimensions
    might be intrepreted as arrays of different dimensions
    if we truncate them to a finite number of dimensions"""
    if not isinstance(array_1, np.ndarray):
        array_1 = np.array([array_1])
    if not isinstance(array_2, np.ndarray):
        array_2 = np.array([array_2])
    if len(array_1.flatten()) < len(array_2.flatten()):
        ax_0 = array_1.shape[0]
        if len(array_1.shape) == 1:
            array_2 = array_2[:ax_0]
        else:
            ax_1 = array_1.shape[1]
            array_2 = array_2[:ax_0,:ax_1]
    elif len(array_1.flatten()) > len(array_2.flatten()):
        ax_0 = array_2.shape[0]
        if len(array_2.shape) == 1:
            array_1 = array_1[:ax_0]
        else:
            ax_1 = array_2.shape[1]
            array_1 = array_1[:ax_0,:ax_1]
    else:
        pass
    return np.allclose(array_1, array_2)
