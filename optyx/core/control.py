"""
Overview
--------

Controlled gates with classical inputs for feed-forward control.

This module implements classical control over quantum gates by defining boxes
that apply actions conditionally based on classical values. This includes
generic controlled boxes, controlled phase shifts.

Classes
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    BitControlledBox
    ControlledPhaseShift

Functions
---------

.. autosummary::
    :template: function.rst
    :nosignatures:
    :toctree:

    truncation_tensor

Examples
--------

We can construct a controlled gate acting on a quantum mode:

>>> f = lambda x: [x * 0.1, x * 0.2]
>>> box = ControlledPhaseShift(f, n_modes=2)
>>> box.draw(path='docs/_static/controlled_phase.svg')

A bit-controlled gate can be composed as:

>>> from optyx.core.zw import ZBox
>>> control = BitControlledBox(ZBox(1, 1, lambda x: x))
>>> control.draw(path='docs/_static/binary_control.svg')
"""

from typing import Callable, List
from discopy import tensor
from discopy.frobenius import Dim
import numpy as np

from optyx.core import diagram, zw


class ControlDiagram(diagram.Diagram):
    pass


class ControlBox(diagram.Box, ControlDiagram):
    pass


class ClassicalDiagram(diagram.Diagram):
    pass


class ClassicalBox(diagram.Box, ClassicalDiagram):
    def conjugate(self):
        return self


class _BitControlledBox(ControlBox):
    def __new__(cls, diag, default_box=None, is_dagger=False):
        if default_box is not None:
            return _BitControlledBox(
                diag, default_box, is_dagger
            )
        from optyx.core import diagram
        from optyx.core.zx import Z

        boxes = []
        for i in range(len(diag)):
            layer = diag[i]
            box = layer.inside[0][1]
            if box.cod == box.dom:# and not isinstance(box, diagram.Swap):
                left = layer.inside[0][0]
                right = layer.inside[0][2]
                copy = Z(1, 2)

                layers = [
                    copy @ left @ box.dom @ right,
                    diagram.Diagram.permutation(
                        [
                            0, *range(2, 2+len(left)), 1,
                            *range(2+len(left), 2+len(left)+len(box.dom))
                        ], diagram.Bit(2) @ left @ box.dom
                    ) @ right,
                    diagram.Bit(1) @ left @ _BitControlledBox(box, ) @ right
                ]

                l = diagram.Diagram.then(*layers)
                boxes.append(l)
            else:
                l = diagram.Bit(1) @ layer
                boxes.append(l)

        boxes.append(Z(1, 0) @ layer.cod)
        if is_dagger:
            return diagram.Diagram.then(*boxes).dagger()
        return diagram.Diagram.then(*boxes)

class BitControlledBox(ControlBox):
    """
    A box controlled by a bit that switches between two boxes:
    - action_box: the box that is applied when the control bit is 1
    - default_box: the box that is applied when
    the control bit is 0 (default is Id)

    Example
    -------
    >>> from optyx.photonic import Phase
    >>> from optyx.core.diagram import PhotonThresholdDetector, Mode
    >>> from optyx.core.zw import Create, ZBox
    >>> action = Phase(0.1).get_kraus()
    >>> default = ZBox(1, 1, lambda x: 1)
    >>> action_result = action.to_tensor().eval().array
    >>> default_result = default.to_tensor().eval().array
    >>> action_test = ((Create(1) >> PhotonThresholdDetector()) @
    ...         Mode(len(action.cod)) >>
    ...         BitControlledBox(action)).to_tensor().eval().array
    >>> default_test = ((Create(0) >> PhotonThresholdDetector()) @
    ...         Mode(len(default.cod)) >>
    ...         BitControlledBox(default)).to_tensor().eval().array
    >>> assert np.allclose(action_result, action_test)
    >>> assert np.allclose(default_result, default_test)
    """

    def __init__(
        self,
        action_box: diagram.Box,
        default_box: diagram.Box = None,
        is_dagger: bool = False,
    ):

        if default_box is None:
            default_box = diagram.Id(action_box.dom)

        assert (
            action_box.dom == default_box.dom
            and action_box.cod == default_box.cod
        ), "action_box and default_box must have the same domain and codomain"
        assert len(action_box.dom) == len(
            action_box.cod
        ), "action_box must have the same number of inputs and outputs"

        dom = action_box.cod if is_dagger else diagram.Bit(1) @ action_box.dom
        cod = diagram.Bit(1) @ action_box.cod if is_dagger else action_box.cod

        if hasattr(action_box, "name"):
            box_name = action_box.name + "_controlled"
        else:
            box_name = "controlled_box"

        super().__init__(box_name, dom, cod)

        self.action_box = action_box
        self.default_box = default_box
        self.is_dagger = is_dagger

    def determine_output_dimensions(self, input_dims: List[int]) -> List[int]:

        action_box_dims = (
            self.action_box.to_tensor(input_dims).cod.inside
            if self.is_dagger
            else self.action_box.to_tensor(input_dims[1:]).cod.inside
        )

        default_box_dims = (
            self.default_box.to_tensor(input_dims).cod.inside
            if self.is_dagger
            else self.default_box.to_tensor(input_dims[1:]).cod.inside
        )

        dims = [max(a, b) for a, b in zip(action_box_dims, default_box_dims)]

        if self.is_dagger:
            return [2, *dims]
        return dims

    def truncation(
        self, input_dims: List[int], output_dims: List[int]
    ) -> tensor.Box:

        if self.is_dagger:
            input_dims, output_dims = output_dims, input_dims

        action_in_dim = input_dims[1:]

        array = np.zeros(
            (input_dims[0], *input_dims[1:], *output_dims), dtype=complex
        )

        default_box_tensor = self.default_box.to_tensor(action_in_dim)
        action_box_tensor = self.action_box.to_tensor(action_in_dim)

        array[0, :, :] = (
            (
                default_box_tensor
                >> diagram.truncation_tensor(
                    default_box_tensor.cod.inside, output_dims
                )
            )
            .eval()
            .array.reshape(array[0, :, :].shape)
        )

        array[1, :, :] = (
            (
                action_box_tensor
                >> diagram.truncation_tensor(
                    action_box_tensor.cod.inside, output_dims
                )
            )
            .eval()
            .array.reshape(array[1, :, :].shape)
        )

        if self.is_dagger:
            return tensor.Box(
                self.name, Dim(*input_dims), Dim(*output_dims), array
            ).dagger()
        return tensor.Box(
            self.name,
            Dim(*[int(d) for d in input_dims]),
            Dim(*[int(d) for d in output_dims]), array
        )

    def dagger(self):
        return BitControlledBox(
            self.action_box, self.default_box, not self.is_dagger
        )

    def conjugate(self):
        return BitControlledBox(
            self.action_box.conjugate(),
            self.default_box.conjugate(),
            self.is_dagger,
        )


class ControlledPhaseShift(ControlBox):
    """
    A controlled phase shift on modes, where the control
    is a natural number and
    the phase applied is determined by a user-defined function.

    The function maps each control value to a list
    of real values (interpreted as 2π multiples of phase shifts).

    Example
    -------
    >>> from optyx.core.diagram import Id, Mode
    >>> from optyx.core.zw import Create, ZBox
    >>> f = lambda x: [x[0]*0.1, x[0]*0.2, x[0]*0.3]
    >>> n = len(f([0]))
    >>> controlled_phase = (Create(2) @ Mode(n) >>
    ...                     ControlledPhaseShift(f, n_modes=n))
    >>> zbox = Id(Mode(0))
    >>> for y in f([2]):
    ...     zbox @= ZBox(1, 1,
    ...         lambda i, y=y: np.exp(2 * np.pi * 1j * y) ** i)
    >>> assert np.allclose(controlled_phase.to_tensor().eval().array,
    ...                    zbox.to_tensor().eval().array)
    """

    def __init__(
        self,
        function: Callable[[List[int]], List[int]],
        n_modes: int = 1,
        n_control_modes: int = 1,
        is_dagger: bool = False,
    ):

        dom = diagram.Mode(n_modes) if is_dagger else diagram.Mode(
            n_modes + n_control_modes
            )
        cod = diagram.Mode(
            n_modes + n_control_modes
            ) if is_dagger else diagram.Mode(n_modes)

        super().__init__("ControlledPhase", dom, cod)
        self.n_modes = n_modes
        self.function = function
        self.is_dagger = is_dagger
        self.n_control_modes = n_control_modes

    def truncation(
        self, input_dims: List[int], output_dims: List[int]
    ) -> tensor.Box:

        if self.is_dagger:
            input_dims, output_dims = output_dims, input_dims

        array = np.zeros((*input_dims, *output_dims), dtype=complex)

        input_combinations = np.array(
            np.meshgrid(*[range(i) for i
                          in input_dims[:self.n_control_modes]]),
        ).T.reshape(-1, len(input_dims[:self.n_control_modes]))

        for i in input_combinations:
            fx = self.function(i)
            zbox = diagram.Id(diagram.Mode(0))
            for y in fx:
                zbox @= zw.ZBox(
                    1, 1, lambda x, y=y: np.exp(2 * np.pi * 1j * y) ** x
                )

            zbox = zbox.to_tensor(input_dims[self.n_control_modes:])
            array[i, :] = (
                (zbox >> diagram.truncation_tensor(zbox.cod.inside,
                                                   output_dims))
                .eval()
                .array.reshape(array[i, :].shape)
            )

        if self.is_dagger:
            return tensor.Box(
                self.name, Dim(*input_dims), Dim(*output_dims), array
            ).dagger()
        return tensor.Box(
            self.name, Dim(*input_dims), Dim(*output_dims), array
        )

    def determine_output_dimensions(self, input_dims: List[int]) -> List[int]:
        if self.is_dagger:
            return [diagram.MAX_DIM]*self.n_control_modes + input_dims
        return input_dims[self.n_control_modes:]

    def dagger(self):
        return ControlledPhaseShift(
            self.function, self.n_modes,
            self.n_control_modes, not self.is_dagger
        )

    def conjugate(self):
        return ControlledPhaseShift(
            self.function, self.n_modes, self.n_control_modes, self.is_dagger
        )


class ClassicalFunctionBox(ClassicalBox):

    def __init__(
        self,
        function: Callable[[List[int]], List[int]],
        dom: diagram.Mode | diagram.Bit,
        cod: diagram.Mode | diagram.Bit,
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

    def truncation_specificaton(
        self, input_dims: list[int] = None, output_dims: list[int] = None
    ):
        input_ranges = [range(d) for d in input_dims]
        input_combinations = np.array(np.meshgrid(*input_ranges)).T.reshape(
            -1, len(input_dims)
        )

        outputs = [
            (i, self.function(i))
            for i in input_combinations
            if self.function(i) != 0
        ]

        return [(tuple(tuple(output) + tuple(input_)), 1.0) for input_, output in outputs]

    def determine_output_dimensions(self, input_dims: List[int]) -> List[int]:
        if self.cod == diagram.Mode(self.output_size):
            return [diagram.MAX_DIM] * self.output_size

        elif self.cod == diagram.Bit(self.output_size):
            return [2] * self.output_size

        else:
            return [int(max(input_dims))] * self.output_size

    def dagger(self):
        return ClassicalFunctionBox(
            self.function, self.cod, self.dom, not self.is_dagger
        )


class BinaryMatrixBox(ClassicalBox):
    """
    Represents a linear transformation over
    GF(2) using matrix multiplication.

    Example
    -------
    >>> from optyx.core.zx import X
    >>> from optyx.core.diagram import Scalar
    >>> xor = X(2, 1) @ Scalar(np.sqrt(2))
    >>> matrix = [[1, 1]]
    >>> m_res = BinaryMatrixBox(matrix).to_tensor().eval().array
    >>> xor_res = xor.to_tensor().eval().array
    >>> assert np.allclose(m_res, xor_res)

    """

    def __init__(self, matrix: np.ndarray, is_dagger: bool = False):

        matrix = np.array(matrix)
        if len(matrix.shape) == 1:
            matrix = matrix.reshape(1, -1)

        cod = diagram.Bit(
            len(matrix[0])
            ) if is_dagger else diagram.Bit(len(matrix))
        dom = diagram.Bit(
            len(matrix)
            ) if is_dagger else diagram.Bit(len(matrix[0]))

        super().__init__("LogicalMatrix", dom, cod)

        self.matrix = matrix
        self.is_dagger = is_dagger

    def truncation_specificaton(
        self, input_dims: list[int] = None, output_dims: list[int] = None
    ):
        def f(x):
            if not isinstance(x, np.ndarray):
                x = np.array(x, dtype=np.uint8)
            if len(x.shape) == 1:
                x = x.reshape(-1, 1)
            A = np.array(self.matrix, dtype=np.uint8)

            return list(((A @ x) % 2).reshape(1, -1)[0])

        return ClassicalFunctionBox(
            f, self.dom, self.cod
        ).truncation_specificaton(input_dims, output_dims)

    def determine_output_dimensions(self,
                                    input_dims: List[int]) -> List[int]:
        return ClassicalFunctionBox(
            None, self.dom, self.cod
        ).determine_output_dimensions(input_dims)

    def dagger(self):
        return BinaryMatrixBox(self.matrix, not self.is_dagger)

    def conjugate(self):
        return self
