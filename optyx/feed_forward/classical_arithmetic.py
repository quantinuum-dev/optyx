"""
Overview
--------

Classical arithmetic and logical operations on `Mode` wires.

This module implements reversible classical gates such as AND, ADD,
MOD2, and multiplicative operations, following a categorical approach
compatible with :class:`ZW` diagrams.

Each gate supports truncation to tensor semantics and dagger (adjoint)
operations to allow reversible classical computing.

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

from optyx.optyx import (
    Box,
    Bit,
    Swap,
    Mode,
    Scalar,
)
from optyx.zw import W, Create
from optyx.optyx import Spider
from optyx.zx import Z, X
from optyx.feed_forward.controlled_gates import truncation_tensor
from discopy import tensor
from discopy.frobenius import Dim
import numpy as np
from typing import List


class And(Box):
    """
    Reversible classical AND gate on two bits.

    This gate acts as a Toffoli-style classical operation that maps:
        - (1, 1) ↦ 1
        - Otherwise ↦ 0

    All outputs are reversible — i.e., the transformation is
    injective over classical states.

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

    def __init__(self, is_dagger: bool = False):
        super().__init__("And", Bit(2), Bit(1))
        self.is_dagger = is_dagger

    def truncation(
        self, input_dims: List[int], output_dims: List[int]
    ) -> tensor.Box:

        if self.is_dagger:
            input_dims, output_dims = output_dims, input_dims

        array = np.zeros((*input_dims, *output_dims), dtype=complex)
        array[1, 1, 1] = 1
        array[1, 0, 0] = 1
        array[0, 1, 0] = 1
        array[0, 0, 0] = 1

        if self.is_dagger:
            return tensor.Box(
                self.name, Dim(*input_dims), Dim(*output_dims), array
            ).dagger()
        return tensor.Box(
            self.name, Dim(*input_dims), Dim(*output_dims), array
        )

    def determine_output_dimensions(self, input_dims):
        if self.is_dagger:
            return [2, 2]
        return [2]

    def to_zw(self):
        return self

    def dagger(self):
        return And(not self.is_dagger)


class Add(Box):
    """
    Adds multiple classical values using a W-dagger operation.

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
    Performs repeated addition as multiplication.

    Multiplies two classical integers by repeated addition using Z/W diagrams.

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
                                                 add(n))
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

    This operation is approximate and non-injective in general.

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
                                               add(n)).dagger()

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


divide = Divide()
multiply = Multiply()
copy_mode = Spider(1, 2, Mode(1))
def add(n): return Add(n)


subtract = add(2).dagger() @ Mode(1) >> Mode(1) @ Spider(2, 0, Mode(1))
mod2 = Mod2()

postselect_1 = X(1, 0, 0.5) @ Scalar(1 / np.sqrt(2))
postselect_0 = X(1, 0) @ Scalar(1 / np.sqrt(2))
xor = X(2, 1) @ Scalar(np.sqrt(2))
not_ = X(1, 1, 0.5)
copy = Z(1, 2)
swap = Swap(Bit(1), Bit(1))
and_ = And()
