from optyx.channel import Channel, CQMap, Ob, Ty
from optyx.optyx import (
    Box,
    Bit,
    Diagram,
    Swap,
    Mode,
    Scalar,
    Id,
    MAX_DIM
)
from optyx.zw import W,ZBox
from optyx.zx import Z, X
from optyx.feed_forward.controlled_gates import truncation_tensor
from discopy import tensor
from discopy.frobenius import Dim
import numpy as np
from typing import List, Callable

class ClassicalFunctionBox(Box):
    def __init__(self,
                 function : Callable[[List[int]], List[int]],
                 dom : Mode | Bit,
                 cod : Mode | Bit,
                 is_dagger : bool = False):

        #assert cod == Bit(len(cod)), "cod must binary Bit(n)"
        #assert all([d == dom[0] for d in dom]), "dom must be either all Mode(n) or all Bit(n)"

        #dom = cod if is_dagger else dom
        #cod = dom if is_dagger else cod

        super().__init__("F", dom, cod)

        self.function = function
        self.input_size = len(dom)
        self.output_size = len(cod)
        self.is_dagger = is_dagger

    def to_zw(self):
        return self

    def truncation(self,
                   input_dims : List[int],
                   output_dims : List[int]) -> tensor.Box:

        if self.is_dagger:
            input_dims, output_dims = output_dims, input_dims

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
                              Dim(*input_dims),
                              Dim(*output_dims),
                              array).dagger()

        return tensor.Box(self.name,
                          Dim(*input_dims),
                          Dim(*output_dims),
                          array)

    def determine_output_dimensions(self,
                                    input_dims : List[int]) -> List[int]:
        if (self.dom == Bit(self.input_size) and
            self.cod == Mode(self.output_size)):
            return [MAX_DIM]*self.output_size
        elif (self.dom == Mode(self.input_size) and
              self.cod == Bit(self.output_size)):
            return [2]*self.output_size
        else:
            return [max(input_dims)]*self.output_size

    def dagger(self):
        return ClassicalFunctionBox(self.function,
                                    self.cod,
                                    self.dom,
                                    not self.is_dagger)

    def conjugate(self):
        return self


class LogicalMatrixBox(Box):
    '''
    Matrix multiplication in GF(2)
    '''
    def __init__(self,
                 matrix : np.ndarray,
                 is_dagger : bool = False):

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
                   input_dims : List[int],
                   output_dims : List[int]) -> tensor.Box:

        if self.is_dagger:
            input_dims, output_dims = output_dims, input_dims

        def f(x):
            if len(x.shape) == 1:
                x = x.reshape(-1, 1)
            A = np.array(self.matrix, dtype=np.uint8)
            x = np.array(x, dtype=np.uint8)

            return list(((A @ x) % 2).reshape(1, -1)[0])

        classical_function = ClassicalFunctionBox(f, self.dom, self.cod)

        if self.is_dagger:
            return classical_function.truncation(input_dims, output_dims).dagger()
        return classical_function.truncation(input_dims, output_dims)

    def determine_output_dimensions(self,
                                    input_dims : List[int]) -> List[int]:
        return ClassicalFunctionBox(
            None,
            self.dom,
            self.cod
        ).determine_output_dimensions(input_dims)

    def dagger(self):
        return LogicalMatrixBox(self.matrix, not self.is_dagger)

    def conjugate(self):
        return self


class ClassicalCircuitBox(Diagram):
    def __new__(self,
                diagram : Diagram) -> Diagram:
        return diagram


class PhaseShiftParamControl(Box):
    def __init__(self,
                 function : Callable[[int], List[float]],
                 dom : Mode,
                 cod : Mode,
                 is_dagger : bool = False):

        #assert dom == Mode(1), "dom must be Mode(1)"
        #assert cod == Mode(len(cod)), "cod must be Mode(n)"

        super().__init__("PhaseShiftParamControl", dom, cod)

        self.function = function
        self.is_dagger = is_dagger

    def to_zw(self):
        return self

    def truncation(self,
                     input_dims : List[int],
                     output_dims : List[int]) -> tensor.Box:

        if self.is_dagger:
            input_dims, output_dims = output_dims, input_dims

        #assert len(input_dims) == 1, "input_dims must be of length 1"
        array = np.zeros((*input_dims, *output_dims), dtype=complex)

        for i in range(input_dims[0]):
            fx = self.function(i)
            zbox = Id(Mode(0))
            for y in fx:
                exp = np.exp(2 * np.pi * 1j * y)
                zbox @= ZBox(0, 1, lambda i: exp ** i)

            zbox = zbox.to_tensor(input_dims)
            array[i, :] = (zbox >>
                           truncation_tensor(zbox.cod.inside,
                                             output_dims)).eval().array.reshape(array[i, :].shape)

        if self.is_dagger:
            return tensor.Box(self.name,
                              Dim(*input_dims),
                              Dim(*output_dims),
                              array).dagger()

        return tensor.Box(self.name,
                            Dim(*input_dims),
                            Dim(*output_dims),
                            array)

    def determine_output_dimensions(self,
                                    input_dims : List[int]) -> List[int]:
        if self.is_dagger:
            return [MAX_DIM]*len(self.cod)
        return [max(input_dims)]*len(self.cod)

    def dagger(self):
        return PhaseShiftParamControl(self.function,
                                       self.cod,
                                       self.dom,
                                       not self.is_dagger)

class ControlChannel(Channel):
    def __new__(self,
                control_box : Diagram | ClassicalFunctionBox | LogicalMatrixBox) -> CQMap:
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