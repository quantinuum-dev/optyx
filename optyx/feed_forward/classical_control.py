from optyx.channel import Channel, CQMap,  Ob, Ty
from optyx.optyx import (
    Box,
    Bit,
    Diagram,
    Swap,
    Mode,
    Scalar
)
from optyx.zw import W,ZBox
from optyx.zx import Z, X
from discopy import tensor
from discopy.frobenius import Dim
import numpy as np


class ClassicalFunctionBox(Box):
    def __init__(self,
                 function,
                 dom,
                 cod,
                 is_dagger=False):

        assert cod == Bit(len(cod)), "cod must binary Bit(n)"
        assert all([d == dom[0] for d in dom]), "dom must be either all Mode(n) or all Bit(n)"

        dom = cod if is_dagger else dom
        cod = dom if is_dagger else cod

        super().__init__("F", dom, cod)

        self.function = function
        self.input_size = len(dom)
        self.output_size = len(cod)
        self.is_dagger = is_dagger

    def to_zw(self):
        return self

    def truncation(self,
                   input_dims,
                   output_dims):
        array = np.zeros((*input_dims, *output_dims), dtype=complex)
        input_ranges = [range(i) for i in input_dims]
        input_combinations = np.array(
            np.meshgrid(*input_ranges)
        ).T.reshape(-1, len(input_dims))

        outputs = [(i, self.function(i)) for i in input_combinations if self.function(i) != 0]
        full_indices = np.array([tuple(input_) + tuple(output) for input_, output in outputs])
        array[tuple(full_indices.T)] = 1

        if self.is_dagger:
            return tensor.Box(self.name,
                              Dim(*output_dims),
                              Dim(*input_dims),
                              array).dagger()

        return tensor.Box(self.name,
                          Dim(*input_dims),
                          Dim(*output_dims),
                          array)

    def determine_output_dimensions(self,
                                    input_dims):
        return [2]*self.output_size

    def dagger(self):
        return ClassicalFunctionBox(self.function,
                                 self.input_size,
                                 self.output_size,
                                 not self.is_dagger)

    def conjugate(self):
        return self


class LogicalMatrixBox(Box):
    '''
    Matrix multiplication in GF(2)
    '''
    def __init__(self,
                 matrix,
                 is_dagger=False):
        if len(matrix.shape) == 1:
            matrix = matrix.reshape(1, -1)
        cod = Bit(len(matrix[0])) if is_dagger else Bit(len(matrix))
        dom = Bit(len(matrix)) if is_dagger else Bit(len(matrix[0]))
        super().__init__("LogicalMatrix", dom, cod)
        self.matrix = matrix
        self.is_dagger = is_dagger

    def to_zw(self):
        return self

    def truncation(self,
                   input_dims,
                   output_dims):
        def f(x):
            if len(x.shape) == 1:
                x = x.reshape(-1, 1)
            A = np.array(self.matrix, dtype=np.uint8)
            x = np.array(x, dtype=np.uint8)

            return list(((A @ x) % 2).reshape(1, -1)[0])
        classical_function = ClassicalFunctionBox(f, self.dom, self.cod)
        return classical_function.truncation(input_dims, output_dims)

    def determine_output_dimensions(self, input_dims):
        return [2]*len(self.cod)

    def dagger(self):
        return LogicalMatrixBox(self.matrix, not self.is_dagger)

    def conjugate(self):
        return self


class ClassicalCircuitBox(Diagram):
    def __new__(self,
                diagram):
        return diagram


class ControlChannel(Channel):
    def __new__(self,
                control_box):
        assert isinstance(
                control_box,
                (
                    Diagram,
                    ClassicalFunctionBox,
                    LogicalMatrixBox
                )
        ), "control_box needs to be an instance of optyx.Diagram, ClassicalFunctionBox, LogicalMatrixBox"

        return CQMap(
            "Classical Control",
            control_box,
            Ty(*[Ob._classical[ob.name] for ob in control_box.dom.inside]),
            Ty(*[Ob._classical[ob.name] for ob in control_box.cod.inside])
        )

#boxes:
# - and
# - or
# - not
# - xor
# - copy
# - swap
# - add
# - subtract
# - multiply
# - divide
# - mod 2
# - copy_mode

postselect_1 = X(1, 0, 0.5) @ Scalar(1/np.sqrt(2))
postselect_0 = X(1, 0) @ Scalar(1/np.sqrt(2))
xor = X(2, 1) @ Scalar(np.sqrt(2))
copy =  Z(1, 2)
swap = Swap(Bit(1), Bit(1))

add = W(2).dagger()
subtract = (
    W(2) @ Mode(1) >>
    Mode(1) @ ZBox(2, 0)
)
multiply = lambda x: (
    ZBox(1, x) >> W(x).dagger()
)
divide = lambda x: multiply(x).dagger()
copy_mode = ZBox(1, 2)