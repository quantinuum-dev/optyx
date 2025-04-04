"""
Overview
--------

Controlled gates with classical inputs for feed-forward control.

This module implements classical control over quantum gates by defining boxes
that apply actions conditionally based on classical values. This includes
generic controlled boxes, controlled phase shifts, and utility functions
for truncating tensors.

Classes
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    BinaryControlledBox
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

>>> from optyx.feed_forward.controlled_gates import ControlledPhaseShift
>>> f = lambda x: [x * 0.1, x * 0.2]
>>> box = ControlledPhaseShift(f, n_modes=2)
>>> box.to_zw().draw(path='docs/_static/controlled_phase.svg')

A binary-controlled gate can be composed as:

>>> from optyx.feed_forward.controlled_gates import BinaryControlledBox
>>> from optyx.zw import ZBox
>>> control = BinaryControlledBox(ZBox(1, 1, lambda x: x))
>>> control.to_zw().draw(path='docs/_static/binary_control.svg')
"""


from typing import Callable, List
from optyx.optyx import (
    Box,
    Id,
    Bit,
    Mode,
    EmbeddingTensor,
    MAX_DIM,
)
from optyx.zw import ZBox
from discopy import tensor
from discopy.frobenius import Dim

import numpy as np


class BinaryControlledBox(Box):
    """
    A box controlled by a bit that chooses between two boxes:
    - action_box: the box that is applied when the control bit is 1
    - default_box: the box that is applied when the control bit is 0

    Example
    -------
    >>> from optyx.lo import Phase
    >>> from optyx.optyx import PhotonThresholdDetector, Mode
    >>> from optyx.zw import Create
    >>> action = Phase(0.1)
    >>> default = ZBox(1, 1)
    >>> action_result = action.to_zw().to_tensor().eval().array
    >>> default_result = default.to_zw().to_tensor().eval().array
    >>> action_test = ((Create(1) >> PhotonThresholdDetector()) @
    ...         Mode(len(action.cod)) >>
    ...         BinaryControlledBox(action)).to_zw().to_tensor().eval().array
    >>> default_test = ((Create(0) >> PhotonThresholdDetector()) @
    ...         Mode(len(default.cod)) >>
    ...         BinaryControlledBox(default)).to_zw().to_tensor().eval().array
    >>> assert np.allclose(action_result, action_test)
    >>> assert np.allclose(default_result, default_test)
    """

    def __init__(self,
                 action_box : Box,
                 default_box : Box = None,
                 is_dagger : bool = False):

        if default_box is None:
            default_box = Id(action_box.dom)

        assert (action_box.dom == default_box.dom and
                action_box.cod == default_box.cod), \
            "action_box and default_box must have the same domain and codomain"
        assert (len(action_box.dom) == len(action_box.cod)), \
            "action_box must have the same number of inputs and outputs"

        dom = action_box.cod if is_dagger else Bit(1) @ action_box.dom
        cod = Bit(1) @ action_box.cod if is_dagger else action_box.cod

        if hasattr(action_box, "name"):
            box_name = action_box.name + "_controlled"
        else:
            box_name = "controlled_box"

        action_box = action_box.to_zw()
        default_box = default_box.to_zw()

        super().__init__(box_name,
                         dom,
                         cod)

        self.action_box = action_box
        self.default_box = default_box
        self.is_dagger = is_dagger

    def determine_output_dimensions(self,
                                    input_dims : List[int]) -> List[int]:

        action_box_dims = self.action_box.to_tensor(
            input_dims
        ).cod.inside if self.is_dagger \
        else self.action_box.to_tensor(
            input_dims[1:]
        ).cod.inside

        default_box_dims = self.default_box.to_tensor(
            input_dims
        ).cod.inside if self.is_dagger \
        else self.default_box.to_tensor(
            input_dims[1:]
        ).cod.inside

        dims = [max(a, b) for a, b in zip(action_box_dims,
                                          default_box_dims)]

        if self.is_dagger:
            return [2, *dims]
        return dims

    def truncation(self,
                   input_dims : List[int],
                   output_dims : List[int]) -> tensor.Box:

        if self.is_dagger:
            input_dims, output_dims = output_dims, input_dims

        action_in_dim = input_dims[1:]

        array = np.zeros((input_dims[0],
                            *input_dims[1:],
                            *output_dims), dtype=complex)

        default_box_tensor = self.default_box.to_tensor(action_in_dim)
        action_box_tensor = self.action_box.to_tensor(action_in_dim)

        array[0, :, :] = (
            default_box_tensor >>
            truncation_tensor(
                default_box_tensor.cod.inside,
                output_dims
            )
        ).eval().array.reshape(array[0, :, :].shape)

        array[1, :, :] = (
            action_box_tensor >>
            truncation_tensor(
                action_box_tensor.cod.inside,
                output_dims
            )
        ).eval().array.reshape(array[1, :, :].shape)

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

    def to_zw(self):
        return self

    def dagger(self):
        return BinaryControlledBox(self.action_box,
                                   self.default_box,
                                   not self.is_dagger)


class ControlledPhaseShift(Box):
    """
    A controlled phase shift on modes, where the control
    is a classical integer and
    the phase applied is determined by a user-defined function.

    The function maps each control value to a list
    of real values (interpreted as 2π multiples of phase shifts).

    Example
    -------
    >>> from optyx.optyx import Id
    >>> from optyx.zw import Create
    >>> f = lambda x: [x*0.1, x*0.2, x*0.3]
    >>> n = len(f(0))
    >>> controlled_phase = (Create(2) @ Mode(n) >>
    ...                     ControlledPhaseShift(f, n_modes=n))
    >>> zbox = Id(Mode(0))
    >>> for y in f(2):
    ...     zbox @= ZBox(1, 1,
    ...         lambda i, y=y: np.exp(2 * np.pi * 1j * y) ** i)
    >>> assert np.allclose(controlled_phase.to_tensor().eval().array,
    ...                    zbox.to_tensor().eval().array)
    """
    def __init__(self,
                 function : Callable[[List[int]], List[int]],
                 n_modes : int = 1,
                 is_dagger : bool = False):

        dom = Mode(n_modes) if is_dagger else Mode(n_modes + 1)
        cod = Mode(n_modes + 1) if is_dagger else Mode(n_modes)

        super().__init__("ControlledPhase", dom, cod)
        self.n_modes = n_modes
        self.function = function
        self.is_dagger = is_dagger

    def truncation(self,
                   input_dims : List[int],
                   output_dims : List[int]) -> tensor.Box:

        if self.is_dagger:
            input_dims, output_dims = output_dims, input_dims

        array = np.zeros((*input_dims, *output_dims), dtype=complex)

        for i in range(input_dims[0]):
            fx = self.function(i)
            zbox = Id(Mode(0))
            for y in fx:
                zbox @= ZBox(1, 1,
                             lambda i, y=y: np.exp(
                                 2 * np.pi * 1j * y
                             ) ** i)

            zbox = zbox.to_tensor(input_dims[1:])
            array[i, :] = (
                zbox >>
                truncation_tensor(
                    zbox.cod.inside,
                    output_dims
                )
            ).eval().array.reshape(array[i, :].shape)

        if self.is_dagger:
            return tensor.Box(
                self.name,
                Dim(*input_dims),
                Dim(*output_dims),
                array
            ).dagger()
        return tensor.Box(self.name,
                          Dim(*input_dims),
                          Dim(*output_dims),
                          array
        )

    def determine_output_dimensions(self,
                                    input_dims : List[int]) -> List[int]:
        if self.is_dagger:
            return [MAX_DIM] + input_dims
        return input_dims[1:]

    def to_zw(self):
        return self

    def dagger(self):
        return ControlledPhaseShift(self.function,
                                    self.n_modes,
                                    not self.is_dagger)

    def conjugate(self):
        return ControlledPhaseShift(self.function,
                                    self.n_modes,
                                    self.is_dagger)


def truncation_tensor(input_dims : List[int],
                      output_dims : List[int]) -> tensor.Box:

    assert len(input_dims) == len(output_dims), \
        "input_dims and output_dims must have the same length"

    tensor = EmbeddingTensor(input_dims[0],
                             output_dims[0])

    for i in zip(input_dims[1:], output_dims[1:]):

        tensor = tensor @ EmbeddingTensor(i[0], i[1])
    return tensor