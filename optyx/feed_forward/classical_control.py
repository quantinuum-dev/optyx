"""
Overview
--------

This module defines classical control primitives used in hybrid quantum-classical
circuits.

Classical control is defined using callable boxes that act on classical types
(:class:`Bit` or :class:`Mode`) via user-defined logic or matrix transformations.
These control boxes can be embedded into quantum-classical circuits using the
:class:`ControlChannel` to convert them into :class:`CQMap` channels.

Classes
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    ClassicalFunctionBox
    LogicalMatrixBox
    ClassicalCircuitBox
    ControlChannel

Examples
--------

A classical function can be embedded into a circuit:

>>> from optyx.feed_forward.classical_control import ClassicalFunctionBox
>>> box = ClassicalFunctionBox(lambda bits: [sum(bits) % 2], Bit(2), Bit(1))
>>> box.to_zw().draw(path='docs/_static/classical_func.svg')

A classical matrix transformation (e.g. stabilizer logic) can be applied:

>>> import numpy as np
>>> from optyx.feed_forward.classical_control import LogicalMatrixBox
>>> matrix = np.array([[1, 1], [0, 1]], dtype=np.uint8)
>>> logical = LogicalMatrixBox(matrix)
>>> logical.to_zw().draw(path='docs/_static/logical_matrix.svg')
"""


from optyx.channel import Channel, CQMap, Ob, Ty
from optyx.optyx import (
    Box,
    Bit,
    Diagram,
    Mode,
    MAX_DIM
)
from discopy import tensor
from discopy.frobenius import Dim
import numpy as np
from typing import List, Callable


class ClassicalFunctionBox(Box):
    """
    A classical function box mode -> bit or bit -> bit, mapping an input list of integers
    to an output bit via a user-defined function.

    Example
    -------
    >>> from optyx.feed_forward.classical_arithmetic import xor
    >>> f_res = ClassicalFunctionBox(lambda x: [x[0] ^ x[1]],
    ...         Bit(2),
    ...         Bit(1)).to_zw().to_tensor().eval().array
    >>> xor_res = xor.to_zw().to_tensor().eval().array
    >>> assert np.allclose(f_res, xor_res)

    """
    def __init__(self,
                 function : Callable[[List[int]], List[int]],
                 dom : Mode | Bit,
                 cod : Mode | Bit,
                 is_dagger : bool = False):

        if is_dagger:
            assert dom == Bit(len(dom)), "dom must be Bit(n)"
            assert all([d == cod[0] for d in cod]), "cod must be either all Mode(n) or all Bit(n)"
        else:
            assert cod == Bit(len(cod)), "cod must be Bit(n)"
            assert all([d == dom[0] for d in dom]), "dom must be either all Mode(n) or all Bit(n)"

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
    Represents a linear transformation over GF(2) using matrix multiplication.

    Example
    -------
    >>> from optyx.feed_forward.classical_arithmetic import xor
    >>> matrix = [[1, 1]]
    >>> m_res = LogicalMatrixBox(matrix).to_tensor().eval().array
    >>> xor_res = xor.to_zw().to_tensor().eval().array
    >>> assert np.allclose(m_res, xor_res)

    '''
    def __init__(self,
                 matrix : np.ndarray,
                 is_dagger : bool = False):

        matrix = np.array(matrix)
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
    """
    Identity wrapper for classical diagrams. Provides a unified interface for
    definiting control boxes from optyx diagrams.
    """
    def __new__(self,
                diagram : Diagram) -> Diagram:
        return diagram


class ControlChannel(Channel):
    """
    Syntactic sugar.
    Converts a classical circuit (Diagram or Box) into a CQMap channel, allowing
    it to be used as a control channel in hybrid quantum-classical systems.

    Accepts ClassicalFunctionBox, LogicalMatrixBox, or raw optyx Diagrams.
    """
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
