# channels
# detectors
    # photon threshold detectors (qmode -> bit)
    # photon number resolving detectors (qmode -> mode)
    # dual rail postselections
    # fusion measurements (made of the above)
# classically controlled gates
#  three classes taking any (suitable) gates as inputs and returning a new controlled gate:
    # classically controlled gates are gates that are controlled by classical bits
    # classically controlled gates with continuous parameters
    # classically controlled gates with discrete parameters
# classical functions on the outputs of detectors
#  input are the functions between natural numbers, bits and real numbers
    # to be used as an input to classically controlled gates
from optyx.Channel import Channel, Measure, bit
from optyx.optyx import PhotonThresholdDetector, Box, Id, bit, EmbeddingTensor
from discopy import tensor, Dim
import numpy as np

class PhotonThresholdMeasurement(Channel):
    """
    Ideal non-photon resolving detector from qmode to bit.
    Detects whether one or more photons are present.
    """

    def __init__(self):
        super().__init__("PhotonThresholdMeasurement",
                         PhotonThresholdDetector(), cod=bit)


#Ideal photon number resolving detector from qmode to mode.
NumberResolvingMeasurement = Measure


def truncation_tensor(input_dims, output_dims):

    assert len(input_dims) == len(output_dims), \
        "input_dims and output_dims must have the same length"

    tensor = EmbeddingTensor(input_dims[0], output_dims[0])

    for i in zip(input_dims[1:], output_dims[1:]):

        tensor = tensor @ EmbeddingTensor(i[0], i[1])
    return tensor

class BinaryControlledBox(Box):
    """
    A box controlled by a bit that chooses between two boxes:
    - action_box: the box that is applied when the control bit is 1
    - default_box: the box that is applied when the control bit is 0
    """

    def __init__(self, action_box, default_box=None, is_dagger=False):
        if default_box is None:
            default_box = Id(action_box.dom)

        assert (action_box.dom == default_box.dom and
                action_box.cod == default_box.cod), \
            "action_box and default_box must have the same domain and codomain"
        assert (len(action_box.dom) == len(action_box.cod)), \
            "action_box must have the same number of inputs and outputs"

        dom = bit @ action_box.dom
        cod = action_box.cod

        if hasattr(action_box, "name"):
            box_name = action_box.name + "_controlled"
        else:
            box_name = "controlled_box"

        action_box = action_box.to_zw()
        default_box = default_box.to_zw()

        super().__init__(box_name,
                         dom,
                         cod)

        self.action_box = action_box.dagger() if is_dagger else action_box
        self.default_box = default_box.dagger() if is_dagger else default_box
        self.is_dagger = is_dagger

    def determine_output_dimensions(self, input_dims):
        action_box_dims = self.action_box.to_tensor(input_dims).cod.inside if self.is_dagger else \
                            self.action_box.to_tensor(input_dims[1:]).cod.inside

        default_box_dims = self.default_box.to_tensor(input_dims).cod.inside if self.is_dagger else \
                            self.default_box.to_tensor(input_dims[1:]).cod.inside

        dims = [max(a, b) for a, b in zip(action_box_dims, default_box_dims)]

        return [2, *dims] if self.is_dagger else dims

    def truncation(self, input_dims, output_dims):
        action_in_dim = input_dims if self.is_dagger else input_dims[1:]

        array = np.zeros((2,*output_dims[1:],
                            *input_dims), dtype=complex) if self.is_dagger else \
                    np.zeros((input_dims[0],
                            *input_dims[1:],
                            *output_dims), dtype=complex)

        default_box_tensor = self.default_box.to_tensor(action_in_dim)
        action_box_tensor = self.action_box.to_tensor(action_in_dim)

        array[0, :, :] = (default_box_tensor >>
                          truncation_tensor(default_box_tensor.cod.inside,
                                            output_dims)).eval().array.reshape(array[0, :, :].shape)
        array[1, :, :] = (action_box_tensor >>
                          truncation_tensor(action_box_tensor.cod.inside,
                                            output_dims)).eval().array.reshape(array[1, :, :].shape)

        if self.is_dagger:
            return tensor.Box(self.name, Dim(*output_dims), Dim(*input_dims), array).dagger()
        return tensor.Box(self.name, Dim(*input_dims), Dim(*output_dims), array)

    def to_zw(self):
        return self

    def dagger(self):
        return BinaryControlledBox(self.action_box.dagger(),
                                   self.default_box.dagger(),
                                   not self.is_dagger)