from optyx.optyx import (
    Box,
    Bit,
    Swap,
    Mode,
    Scalar,
)
from optyx.zw import W, ZBox, Create
from optyx.zx import Z, X
from optyx.feed_forward.controlled_gates import truncation_tensor
from discopy import tensor
from discopy.frobenius import Dim
import numpy as np


class And(Box):
    def __init__(self, is_dagger=False):
        super().__init__("And", Bit(2), Bit(1))
        self.is_dagger = is_dagger

    def truncation(self,
                   input_dims,
                   output_dims):

        if self.is_dagger:
            input_dims, output_dims = output_dims, input_dims

        array = np.zeros((*input_dims, *output_dims), dtype=complex)
        array[1, 1, 1] = 1
        array[1, 0, 0] = 1
        array[0, 1, 0] = 1
        array[0, 0, 0] = 1

        if self.is_dagger:
            return tensor.Box(
                self.name,
                Dim(*input_dims),
                Dim(*output_dims),
                array
            ).dagger()
        return tensor.Box(
            self.name,
            Dim(*input_dims),
            Dim(*output_dims),
            array
        )

    def determine_output_dimensions(self,
                                    input_dims):
        if self.is_dagger:
            return [2, 2]
        return [2]

    def to_zw(self):
        return self

    def dagger(self):
        return And(not self.is_dagger)

    def conjugate(self):
        return self


class Add(Box):
    def __init__(self, n, is_dagger=False):
        dom = Mode(1) if is_dagger else Mode(n)
        cod = Mode(n) if is_dagger else Mode(1)

        super().__init__("Add", dom, cod)
        self.n = n
        self.is_dagger = is_dagger

    def truncation(self,
                   input_dims,
                   output_dims):

        if self.is_dagger:
            input_dims, output_dims = output_dims, input_dims

        diag = W(self.n).dagger().to_tensor(input_dims)
        array = np.sign(
            (
                diag >> truncation_tensor(
                    diag.cod.inside,
                    output_dims
                    )
            ).eval().array
        )
        if self.is_dagger:
            return tensor.Box(
                "Add",
                Dim(*input_dims),
                Dim(*output_dims),
                array
            ).dagger()
        return tensor.Box(
            "Add",
            Dim(*input_dims),
            Dim(*output_dims),
            array
        )

    def determine_output_dimensions(self,
                                    input_dims):
        if self.is_dagger:
            return [input_dims[0]]*self.n
        return [sum(input_dims)]

    def to_zw(self):
        return self

    def dagger(self):
        return Add(self.n, not self.is_dagger)

    def conjugate(self):
        return self


class Multiply(Box):
    def __init__(self, is_dagger=False):
        dom = Mode(1) if is_dagger else Mode(2)
        cod = Mode(2) if is_dagger else Mode(1)

        super().__init__("Multiply", dom, cod)

        self.is_dagger = is_dagger

    def truncation(self,
                   input_dims,
                   output_dims):

        if self.is_dagger:
            input_dims, output_dims = output_dims, input_dims

        array = np.zeros((*input_dims, *output_dims), dtype=complex)

        for i in range(input_dims[0]):
            if i > 0:
                multiply_diagram = lambda n: (
                    ZBox(1, n) >> add(n)
                )
            else:
                multiply_diagram = lambda n : ZBox(1, 0) >> Create(0)

            d = multiply_diagram(i).to_tensor([input_dims[1]])
            d = d >> truncation_tensor(d.cod.inside, output_dims)

            array[i, :] = d.eval().array.reshape(array[i, :].shape)

        if self.is_dagger:
            return tensor.Box(
                self.name,
                Dim(*input_dims),
                Dim(*output_dims),
                array
            ).dagger()
        return tensor.Box(
            self.name,
            Dim(*input_dims),
            Dim(*output_dims),
            array
        )

    def determine_output_dimensions(self,
                                    input_dims):
        if self.is_dagger:
            return [int(input_dims[0])]
        return [int(np.prod(input_dims))]

    def to_zw(self):
        return self

    def dagger(self):
        return Multiply(not self.is_dagger)

    def conjugate(self):
        return self


class Divide(Box):
    def __init__(self, is_dagger=False):
        dom = Mode(1) if is_dagger else Mode(2)
        cod = Mode(2) if is_dagger else Mode(1)

        super().__init__("Divide", dom, cod)

        self.is_dagger = is_dagger

    def truncation(self,
                   input_dims,
                   output_dims):

        if self.is_dagger:
            input_dims, output_dims = output_dims, input_dims

        array = np.zeros((*input_dims, *output_dims), dtype=complex)

        for i in range(input_dims[1]):
            if i > 0:
                divide_diagram = lambda n: (
                    ZBox(1, n) >> add(n)
                ).dagger()

                d = divide_diagram(i).to_tensor([input_dims[0]])
                d = d >> truncation_tensor(d.cod.inside, output_dims)

                array[:, i, :] = d.eval().array.reshape(array[:, i, :].shape)

        if self.is_dagger:
            return tensor.Box(
                self.name,
                Dim(*input_dims),
                Dim(*output_dims),
                array
            ).dagger()
        return tensor.Box(
            self.name,
            Dim(*input_dims),
            Dim(*output_dims),
            array
        )

    def determine_output_dimensions(self,
                                    input_dims):
        if self.is_dagger:
            return [int(input_dims[0])]
        return [int(np.prod(input_dims))]

    def to_zw(self):
        return self

    def dagger(self):
        return Divide(not self.is_dagger)

    def conjugate(self):
        return self


class Mod2(Box):
    def __init__(self, is_dagger=False):
        super().__init__("Mod2", Mode(1), Mode(1))
        self.is_dagger = is_dagger

    def truncation(self,
                   input_dims,
                   output_dims):

        if self.is_dagger:
            input_dims, output_dims = output_dims, input_dims

        array = np.zeros((*input_dims, *output_dims), dtype=complex)

        for i in range(input_dims[0]):
            array[i, i % 2] = 1

        if self.is_dagger:
            return tensor.Box(
                self.name,
                Dim(*input_dims),
                Dim(*output_dims),
                array
            ).dagger()
        return tensor.Box(
            self.name,
            Dim(*input_dims),
            Dim(*output_dims),
            array
        )

    def determine_output_dimensions(self,
                                    input_dims):
        if self.is_dagger:
            return [input_dims[0]]
        return [2]

    def to_zw(self):
        return self

    def dagger(self):
        return Mod2(not self.is_dagger)

    def conjugate(self):
        return self

divide = Divide()
multiply = Multiply()
copy_mode = ZBox(1, 2)
add = lambda n: Add(n)
subtract = (
    add(2).dagger() @ Mode(1) >>
    Mode(1) @ ZBox(2, 0)
)
mod2 = Mod2()

postselect_1 = X(1, 0, 0.5) @ Scalar(1/np.sqrt(2))
postselect_0 = X(1, 0) @ Scalar(1/np.sqrt(2))
xor = X(2, 1) @ Scalar(np.sqrt(2))
not_ = X(1, 1, 0.5)
copy =  Z(1, 2)
swap = Swap(Bit(1), Bit(1))
and_ = And()