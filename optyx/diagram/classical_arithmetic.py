"""
Overview
--------

Classical arithmetic and logical operations on `Mode` wires.

This module implements classical gates such as AND, ADD,
MOD2, and multiplicative operations, following a categorical approach
compatible with :class:`ZW` diagrams.

Each gate supports truncation to tensor semantics and dagger (adjoint)
operations to allow classical computing.

Classes
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    And
    Add
    Multiply
    Divide
    Mod2

Examples
--------

We can simulate classical modular arithmetic:

>>> from optyx.feed_forward.classical_arithmetic import Mod2
>>> mod2_gate = Mod2().to_zw().to_tensor(input_dims=[5]).eval().array
>>> assert list(map(int, map(np.argmax, mod2_gate))) == [0, 1, 0, 1, 0]

Or implement addition:

>>> from optyx.feed_forward.classical_arithmetic import Add
>>> add_box = Add(2)
>>> tensor = add_box.to_zw().to_tensor(input_dims=[2, 2]).eval().array
>>> assert np.allclose(tensor.sum(), 4)
"""

from optyx.diagram.optyx import (
    Box,
    Bit,
    Swap,
    Mode,
    Scalar,
)
from optyx.diagram.zw import W, Create
from optyx.diagram.optyx import Spider
from optyx.diagram.zx import Z, X
from optyx.diagram.optyx import truncation_tensor
from discopy import tensor
from discopy.frobenius import Dim
import numpy as np
from typing import List


class And(Box):
    """
    Reversible classical AND gate on n bits.

    This gate acts as a classical operation that maps:
        - (1, 1) ↦ 1
        - Otherwise ↦ 0

    Example
    -------
    >>> from optyx.feed_forward.classical_arithmetic import And
    >>> and_box = And().to_zw().to_tensor(input_dims=[2, 2]).eval().array
    >>> import numpy as np
    >>> assert np.isclose(and_box[1, 1, 1], 1.0)
    >>> for a in [0, 1]:
    ...     for b in [0, 1]:
    ...         expected = int(a & b)
    ...         assert np.isclose(and_box[a, b, expected], 1.0)
    """

    def __init__(self, n=2, is_dagger: bool = False):
        super().__init__("And", Bit(n), Bit(1))
        self.is_dagger = is_dagger

    def truncation(
        self, input_dims: List[int], output_dims: List[int]
    ) -> tensor.Box:

        if self.is_dagger:
            input_dims, output_dims = output_dims, input_dims

        array = np.zeros((*input_dims, *output_dims), dtype=complex)
        array[..., 0] = 1
        all_ones = (1,) * len(input_dims)
        array[all_ones + (0,)] = 0
        array[all_ones + (1,)] = 1

        if self.is_dagger:
            return tensor.Box(
                self.name, Dim(*input_dims), Dim(*output_dims), array
            ).dagger()
        return tensor.Box(
            self.name, Dim(*input_dims), Dim(*output_dims), array
        )

    def determine_output_dimensions(self, input_dims):
        if self.is_dagger:
            return [2]*len(input_dims)
        return [2]

    def to_zw(self):
        return self

    def dagger(self):
        return And(not self.is_dagger)


class Or(Box):
    """
    Reversible classical OR gate on *n* bits.

    This gate acts as a classical operation that maps
        - (0, 0, …, 0) ↦ 0
        - Otherwise    ↦ 1

    Example
    -------
    >>> from optyx.feed_forward.classical_arithmetic import Or
    >>> or_box = Or().to_zw().to_tensor(input_dims=[2, 2]).eval().array
    >>> import numpy as np
    >>> assert np.isclose(or_box[0, 0, 0], 1.0)   # only all-zeros→0
    >>> for a in [0, 1]:
    ...     for b in [0, 1]:
    ...         expected = int(a | b)
    ...         assert np.isclose(or_box[a, b, expected], 1.0)
    """

    def __init__(self, n: int = 2, is_dagger: bool = False):
        super().__init__("Or", Bit(n), Bit(1))
        self.is_dagger = is_dagger

    def truncation(
        self,
        input_dims: List[int],
        output_dims: List[int],
    ) -> tensor.Box:
        """
        Build the (broadcast-sized) truth-table tensor for the OR gate.
        """
        if self.is_dagger:
            input_dims, output_dims = output_dims, input_dims

        array = np.zeros((*input_dims, *output_dims), dtype=complex)

        array[..., 1] = 1

        all_zeros = (0,) * len(input_dims)
        array[all_zeros + (1,)] = 0
        array[all_zeros + (0,)] = 1

        box = tensor.Box(self.name, Dim(*input_dims), Dim(*output_dims), array)
        return box.dagger() if self.is_dagger else box

    def determine_output_dimensions(self, input_dims: List[int]) -> List[int]:
        return [2] * len(input_dims) if self.is_dagger else [2]

    def to_zw(self):
        return self

    def dagger(self):
        return Or(n=self.dom.n, is_dagger=not self.is_dagger)



class Add(Box):
    """
    Adds multiple classical values using a W-dagger operation.
    Acts by adding the basis vectors without the binomial coefficient.
    Takes `n` mode inputs and returns a single summed mode output
    (or vice versa if daggered).

    Example
    -------
    >>> from optyx.feed_forward.classical_arithmetic import Add
    >>> add_box = Add(2)
    >>> tensor = add_box.to_zw().to_tensor(input_dims=[2, 2]).eval().array
    >>> import numpy as np
    >>> # Expect 4 one-hot outputs for all input combinations
    >>> assert np.allclose(tensor.sum(), 4)
    """

    def __init__(self, n: int, is_dagger: bool = False):
        dom = Mode(1) if is_dagger else Mode(n)
        cod = Mode(n) if is_dagger else Mode(1)

        super().__init__("Add", dom, cod)
        self.n = n
        self.is_dagger = is_dagger

    def truncation(
        self, input_dims: List[int], output_dims: List[int]
    ) -> tensor.Box:

        input_dims = [int(i) for i in input_dims]
        output_dims = [int(i) for i in output_dims]

        if self.is_dagger:
            input_dims, output_dims = output_dims, input_dims

        diag = W(self.n).dagger().to_tensor(input_dims)
        array = np.sign(
            (diag >> truncation_tensor(diag.cod.inside, output_dims))
            .eval()
            .array
        )
        if self.is_dagger:
            return tensor.Box(
                "Add", Dim(*input_dims), Dim(*output_dims), array
            ).dagger()

        return tensor.Box("Add", Dim(*input_dims), Dim(*output_dims), array)

    def determine_output_dimensions(self, input_dims: List[int]) -> List[int]:
        if self.is_dagger:
            return [int(input_dims[0])] * self.n
        return [int(sum(input_dims))]

    def to_zw(self):
        return self

    def dagger(self):
        return Add(self.n, not self.is_dagger)


class Multiply(Box):
    """
    Multiplies two classical integers.

    Example
    -------
    >>> from optyx.feed_forward.classical_arithmetic import Multiply
    >>> mbox = Multiply()
    >>> result = mbox.to_zw().to_tensor(input_dims=[3, 3]).eval().array
    >>> import numpy as np
    >>> assert result.shape == (3, 3, 9)
    >>> nonzero = np.nonzero(result)
    >>> assert len(nonzero[0]) > 0
    """

    def __init__(self, is_dagger: bool = False):
        dom = Mode(1) if is_dagger else Mode(2)
        cod = Mode(2) if is_dagger else Mode(1)

        super().__init__("Multiply", dom, cod)

        self.is_dagger = is_dagger

    def truncation(
        self, input_dims: List[int], output_dims: List[int]
    ) -> tensor.Box:

        if self.is_dagger:
            input_dims, output_dims = output_dims, input_dims

        array = np.zeros((*input_dims, *output_dims), dtype=complex)

        for i in range(input_dims[0]):
            if i > 0:
                def multiply_diagram(n): return (Spider(1, n, Mode(1)) >>
                                                 add_N(n))
            else:
                def multiply_diagram(n): return (Spider(1, 0, Mode(1)) >>
                                                 Create(0))

            d = multiply_diagram(i).to_tensor([input_dims[1]])
            d = d >> truncation_tensor(d.cod.inside, output_dims)

            array[i, :] = d.eval().array.reshape(array[i, :].shape)

        if self.is_dagger:
            return tensor.Box(
                self.name, Dim(*input_dims), Dim(*output_dims), array
            ).dagger()
        return tensor.Box(
            self.name, Dim(*input_dims), Dim(*output_dims), array
        )

    def determine_output_dimensions(self, input_dims: List[int]) -> List[int]:
        if self.is_dagger:
            return [int(input_dims[0])]
        return [int(np.prod(input_dims))]

    def to_zw(self):
        return self

    def dagger(self):
        return Multiply(not self.is_dagger)


class Divide(Box):
    """
    Inverse of multiplication: decomposes a product into factors if possible.

    Example
    -------
    >>> from optyx.feed_forward.classical_arithmetic import Divide
    >>> dbox = Divide()
    >>> result = dbox.to_zw().to_tensor(input_dims=[3, 3]).eval().array
    >>> import numpy as np
    >>> assert result.shape == (3, 3, 9)
    >>> assert np.all(result >= 0)
    """

    def __init__(self, is_dagger: bool = False):
        dom = Mode(1) if is_dagger else Mode(2)
        cod = Mode(2) if is_dagger else Mode(1)

        super().__init__("Divide", dom, cod)

        self.is_dagger = is_dagger

    def truncation(
        self, input_dims: List[int], output_dims: List[int]
    ) -> tensor.Box:

        if self.is_dagger:
            input_dims, output_dims = output_dims, input_dims

        array = np.zeros((*input_dims, *output_dims), dtype=complex)

        for i in range(input_dims[1]):
            if i > 0:
                def divide_diagram(n): return (Spider(1, n, Mode(1)) >>
                                               add_N(n)).dagger()

                d = divide_diagram(i).to_tensor([input_dims[0]])
                d = d >> truncation_tensor(d.cod.inside, output_dims)

                array[:, i, :] = d.eval().array.reshape(array[:, i, :].shape)

        if self.is_dagger:
            return tensor.Box(
                self.name, Dim(*input_dims), Dim(*output_dims), array
            ).dagger()
        return tensor.Box(
            self.name, Dim(*input_dims), Dim(*output_dims), array
        )

    def determine_output_dimensions(self, input_dims: List[int]) -> List[int]:
        if self.is_dagger:
            return [int(input_dims[0])]
        return [int(np.prod(input_dims))]

    def to_zw(self):
        return self

    def dagger(self):
        return Divide(not self.is_dagger)


class Mod2(Box):
    """
    Reduces a classical mode to its parity (even/odd), i.e., modulo 2.

    Example
    -------
    >>> from optyx.feed_forward.classical_arithmetic import Mod2
    >>> m2 = Mod2()
    >>> array = m2.to_zw().to_tensor(input_dims=[5]).eval().array
    >>> import numpy as np
    >>> assert np.allclose([np.argmax(array[i]) for i in range(5)],
    ...    [i % 2 for i in range(5)])
    """

    def __init__(self, is_dagger: bool = False):
        super().__init__("Mod2", Mode(1), Mode(1))
        self.is_dagger = is_dagger

    def truncation(
        self, input_dims: List[int], output_dims: List[int]
    ) -> tensor.Box:

        if self.is_dagger:
            input_dims, output_dims = output_dims, input_dims

        array = np.zeros((*input_dims, *output_dims), dtype=complex)

        for i in range(input_dims[0]):
            array[i, i % 2] = 1

        if self.is_dagger:
            return tensor.Box(
                self.name, Dim(*input_dims), Dim(*output_dims), array
            ).dagger()
        return tensor.Box(
            self.name, Dim(*input_dims), Dim(*output_dims), array
        )

    def determine_output_dimensions(self, input_dims: List[int]) -> List[int]:
        if self.is_dagger:
            return [input_dims[0]]
        return [2]

    def to_zw(self):
        return self

    def dagger(self):
        return Mod2(not self.is_dagger)


# natural number arithmetic
divide_N = Divide() # divide first input by second
multiply_N = Multiply() # multiply two inputs
copy_N = lambda n: Spider(1, n, Mode(1))
def add_N(n): return Add(n) # add multiple inputs
subtract_N = add_N(2).dagger() @ Mode(1) >> Mode(1) @ Spider(2, 0, Mode(1)) # subtract first input by second
mod2 = Mod2() # modulo 2
swap_N = Swap(Mode(1), Mode(1))

# binary arithmetic
postselect_1 = X(1, 0, 0.5) @ Scalar(1 / np.sqrt(2))
postselect_0 = X(1, 0) @ Scalar(1 / np.sqrt(2))
init_1 = postselect_1.dagger()
init_0 = postselect_0.dagger()
xor_bits = lambda n: X(n, 1) @ Scalar(np.sqrt(n))
not_bit = X(1, 1, 0.5)
copy_bit = lambda n: Z(1, n)
swap_bits = Swap(Bit(1), Bit(1))
and_bit = lambda n: And(n)
or_bit = lambda n: Or(n)