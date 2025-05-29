from typing import Callable, List

import numpy as np
from discopy import tensor
from discopy.frobenius import Dim

import optyx.channel as channel
import optyx.zw as zw
import optyx.zx as zx
import optyx.diagram.optyx as optyx
import optyx.classical_arithmetic as classical_arithmetic


class BitControlledGate(channel.Channel):
    """
    Represents a gate that is
    controlled by a classical bit.
    It uses a `BitControlledBox` to define
    the Kraus operators for the gate.
    """
    def __init__(self,
                 control_gate,
                 default_gate=None):
        if isinstance(control_gate, (channel.Circuit, channel.Channel)):
            assert control_gate.is_pure, \
                 "The input gates must be pure quantum channels"
            control_gate_single = control_gate.get_kraus()
        if isinstance(default_gate, (channel.Circuit, channel.Channel)):
            assert default_gate.is_pure, \
                 "The input gates must be pure quantum channels"
            default_gate = default_gate.get_kraus()
        kraus = _BitControlledBoxKraus(control_gate_single, default_gate)
        super().__init__(
            f"BitControlledGate({control_gate}, {default_gate})",
            kraus,
            channel.bit @ control_gate.dom,
            control_gate.cod
        )


class BitControlledPhaseShift(channel.Channel):
    """
    Represents a phase shift operation
    that is controlled by classical bits.
    It uses a `ControlledPhaseShiftBox` to
    define the Kraus operators for the phase shift.
    """
    def __init__(self,
                 function: Callable[[List[int]], List[int]],
                 n_modes: int = 1,
                 n_control_modes: int = 1):
        kraus = _ControlledPhaseShiftKraus(function, n_modes, n_control_modes)
        super().__init__(
            "BitControlledPhaseShift",
            kraus,
            channel.mode**n_control_modes @ channel.qmode**n_modes,
            channel.mode**n_modes,
        )


DiscardBit = lambda n: channel.Discard(channel.bit**n)
DiscardMode = lambda n: channel.Discard(channel.mode**n)


class ClassicalBox(channel.CQMap):
    pass


class AddN(ClassicalBox):
    """
    Classical addition of n natural numbers.
    The domain of the map is n modes.
    The map will perform addition on the basis states.
    """
    def __init__(self, n):
        super().__init__(
            f"AddInt({n})",
            classical_arithmetic.add_N(n),
            channel.mode**n,
            channel.mode
        )


class SubN(ClassicalBox):
    """
    Classical subtraction: subtract the first number from the second.
    The domain of the map is 2 modes.
    The map will perform subtraction on the basis states.
    If the result is negative, it will return a 0 map.
    """
    def __init__(self):
        super().__init__(
            "SubInt",
            classical_arithmetic.ubtract_N,
            channel.mode**2,
            channel.mode
        )


class MultiplyN(ClassicalBox):
    """
    Classical multiplication of 2 natural numbers.
    The domain of the map is 2 modes.
    The map will perform multiplication on the basis states.
    """
    def __init__(self):
        super().__init__(
            "MultiplyInt",
            classical_arithmetic.multiply_N,
            channel.mode**2,
            channel.mode
        )


class DivideN(ClassicalBox):
    """
    Classical division: divide the first number by the second.
    The domain of the map is 2 modes.
    The map will perform division on the basis states.
    If the result is not an integer, it will return a 0 map.
    """
    def __init__(self):
        super().__init__(
            "DivideInt",
            classical_arithmetic.divide_N,
            channel.mode**2,
            channel.mode
        )


class ModN(ClassicalBox):
    """
    Classical modulo 2.
    The domain of the map is a mode.
    The codomain of the map is a bit.
    The map will perform modulo 2 on the basis states.
    """
    def __init__(self):
        super().__init__(
            "ModInt",
            classical_arithmetic.mod2,
            channel.mode,
            channel.bit
        )


class CopyN(ClassicalBox):
    """
    Classical copy of n natural numbers.
    The domain of the map is a mode.
    The codomain of the map is n modes.
    The map will perform copy on the basis states.
    """
    def __init__(self, n):
        super().__init__(
            f"CopyInt({n})",
            classical_arithmetic.copy_N(n),
            channel.mode,
            channel.mode**n
        )


class SwapN(ClassicalBox):
    """
    Classical swap of 2 natural numbers.
    The domain of the map is 2 modes.
    The codomain of the map is 2 modes.
    The map will perform swap on the basis states.
    """
    def __init__(self):
        super().__init__(
            "SwapInt",
            classical_arithmetic.swap_N,
            channel.mode**2,
            channel.mode**2
        )


class PostselectBit(ClassicalBox):
    """
    Postselect on a bit result.
    The domain of the map is a bit.
    """
    def __init__(self, result):

        if result not in (0, 1):
            raise ValueError("Result must be 0 or 1.")
        if result == 0:
            super().__init__(
                f"PostselectBit(0)",
                classical_arithmetic.postselect_0,
                channel.bit,
                channel.bit**0
            )
        else:
            super().__init__(
                f"PostselectBit(1)",
                classical_arithmetic.postselect_1,
                channel.bit,
                channel.bit**0
            )


class InitBit(ClassicalBox):
    """
    Initialize a bit to 0 or 1.
    The domain of the map is a bit.
    The codomain of the map is a bit.
    The map will perform initialization on the basis states.
    """
    def __init__(self, value):
        if value not in (0, 1):
            raise ValueError("Value must be 0 or 1.")
        if value == 0:
            super().__init__(
                f"InitBit(0)",
                classical_arithmetic.init_0,
                channel.bit**0,
                channel.bit
            )
        else:
            super().__init__(
                f"InitBit(1)",
                classical_arithmetic.init_1,
                channel.bit**0,
                channel.bit
            )


class NotBit(ClassicalBox):
    """
    Classical NOT gate.
    The domain of the map is a bit.
    The codomain of the map is a bit.
    The map will perform NOT on the basis states.
    """
    def __init__(self):
        super().__init__(
            "NotBit",
            classical_arithmetic.not_bit,
            channel.bit,
            channel.bit
        )


class XorBit(ClassicalBox):
    """
    Classical XOR gate.
    The domain of the map is n bits.
    The codomain of the map is a bit.
    The map will perform XOR on the basis states.
    """
    def __init__(self, n=2):
        super().__init__(
            f"XorBit({n})",
            classical_arithmetic.xor_bits(n),
            channel.bit**n,
            channel.bit
        )


class AndBit(ClassicalBox):
    """
    Classical AND gate.
    The domain of the map is 2 bits.
    The codomain of the map is a bit.
    The map will perform AND on the basis states.
    """
    def __init__(self, n=2):
        super().__init__(
            "AndBit",
            classical_arithmetic.and_bit(n),
            channel.bit**2,
            channel.bit
        )


class CopyBit(ClassicalBox):
    """
    Classical copy of a bit.
    The domain of the map is a bit.
    The codomain of the map is n bits.
    The map will perform copy on the basis states.
    """
    def __init__(self, n=2):
        super().__init__(
            f"CopyBit({n})",
            classical_arithmetic.copy_bit(n),
            channel.bit,
            channel.bit**n
        )


class SwapBit(ClassicalBox):
    """
    Classical swap of 2 bits.
    The domain of the map is 2 bits.
    The codomain of the map is 2 bits.
    The map will perform swap on the basis states.
    """
    def __init__(self):
        super().__init__(
            "SwapBit",
            classical_arithmetic.swap_bits,
            channel.bit**2,
            channel.bit**2
        )


class OrBit(ClassicalBox):
    """
    Classical OR gate.
    The domain of the map is n bits.
    The codomain of the map is a bit.
    The map will perform OR on the basis states.
    """
    def __init__(self, n=2):
        super().__init__(
            f"OrBit({n})",
            classical_arithmetic.or_bit(n),
            channel.bit**n,
            channel.bit
        )


class Z(ClassicalBox):
    """Z spider."""
    tikzstyle_name = "Z"
    color = "green"
    draw_as_spider = True

    def __init__(self, n_legs_in, n_legs_out, phase=0):
        kraus = zx.Z(n_legs_in, n_legs_out, phase)
        super().__init__(
            f"Z({phase})",
            kraus,
            channel.bit**n_legs_in,
            channel.bit**n_legs_out,
        )


class X(ClassicalBox):
    """X spider."""
    tikzstyle_name = "X"
    color = "red"
    draw_as_spider = True

    def __init__(self, n_legs_in, n_legs_out, phase=0):
        kraus = zx.X(n_legs_in, n_legs_out, phase)
        super().__init__(
            f"X({phase})",
            kraus,
            channel.bit**n_legs_in,
            channel.bit**n_legs_out,
        )


class H(ClassicalBox):
    """Hadamard spider."""
    tikzstyle_name = "H"
    color = "blue"
    draw_as_spider = True

    def __init__(self):
        kraus = zx.H()
        super().__init__(
            f"H",
            kraus,
            channel.bit,
            channel.bit,
        )


class Scalar(ClassicalBox):
    def __init__(self, value: float):
        super().__init__(
            f"Scalar({value})",
            zw.Scalar(value),
            channel.bit**0,
            channel.bit**0,
        )


class ControlChannel(ClassicalBox):
    """
    Syntactic sugar.
    Converts a classical circuit (Diagram or Box)
    into a CQMap, allowing
    it to be used as a control channel in hybrid quantum-classical systems.
    """
    pass


class ClassicalFunction(ControlChannel):
    """
    A classical function box between modes or bits,
    mapping an input list of natural numbers or
    a list of bits to a list of
    natural numbers or a list of bits.

    Example
    -------
    >>> from optyx.zx import X
    >>> from optyx.optyx import Scalar
    >>> xor = X(2, 1) @ Scalar(np.sqrt(2))
    >>> f_res = ClassicalFunctionBox(lambda x: [x[0] ^ x[1]],
    ...         Bit(2),
    ...         Bit(1)).to_zw().to_tensor().eval().array
    >>> xor_res = xor.to_zw().to_tensor().eval().array
    >>> assert np.allclose(f_res, xor_res)
    """

    def __init__(self, function, dom, cod):
        box = _ClassicalFunctionKraus(
            function,
            dom,
            cod
        )
        return super().__init__(
            box.name,
            box,
            channel.Ty(
                *[channel.Ob._classical[ob.name] for ob in box.dom.inside]
            ),
            channel.Ty(
                *[channel.Ob._classical[ob.name] for ob in box.cod.inside]
            ),
        )


class BinaryMatrix(ControlChannel):
    """
    Represents a linear transformation over
    GF(2) using matrix multiplication.

    Example
    -------
    >>> from optyx.zx import X
    >>> from optyx.optyx import Scalar
    >>> xor = X(2, 1) @ Scalar(np.sqrt(2))
    >>> matrix = [[1, 1]]
    >>> m_res = BinaryMatrixBox(matrix).to_tensor().eval().array
    >>> xor_res = xor.to_zw().to_tensor().eval().array
    >>> assert np.allclose(m_res, xor_res)
    """

    def __init__(self, matrix):
        box = _BinaryMatrixBoxKraus(self, matrix)
        return super().__new__(
            box.name,
            box,
            channel.Ty(
                *[channel.Ob._classical[ob.name] for ob in box.dom.inside]
            ),
            channel.Ty(
                *[channel.Ob._classical[ob.name] for ob in box.cod.inside]
            ),
        )


class Select(channel.Channel):
    def __init__(self, *n_photons: int):
        super().__init__(
            f"Select({n_photons})",
            zw.Scalar(*n_photons)
        )


class _ClassicalFunctionKraus(optyx.Box):

    def __init__(
        self,
        function: Callable[[List[int]], List[int]],
        dom: optyx.Mode | optyx.Bit,
        cod: optyx.Mode | optyx.Bit,
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

    def to_zw(self):
        return self

    def truncation(
        self, input_dims: List[int], output_dims: List[int]
    ) -> tensor.Box:

        if self.is_dagger:
            input_dims, output_dims = output_dims, input_dims

        array = np.zeros((*input_dims, *output_dims), dtype=complex)
        input_ranges = [range(i) for i in input_dims]
        input_combinations = np.array(np.meshgrid(*input_ranges)).T.reshape(
            -1, len(input_dims)
        )

        outputs = [
            (i, self.function(i))
            for i in input_combinations
            if self.function(i) != 0
        ]

        full_indices = np.array(
            [tuple(input_) + tuple(output) for input_, output in outputs]
        )
        array[tuple(full_indices.T)] = 1

        input_dims = [int(d) for d in input_dims]
        output_dims = [int(d) for d in output_dims]

        if self.is_dagger:
            return tensor.Box(
                self.name, Dim(*input_dims), Dim(*output_dims), array
            ).dagger()

        return tensor.Box(
            self.name, Dim(*input_dims), Dim(*output_dims), array
        )

    def determine_output_dimensions(self, input_dims: List[int]) -> List[int]:
        if self.cod == optyx.Mode(self.output_size):
            return [optyx.MAX_DIM] * self.output_size

        elif self.cod == optyx.Bit(self.output_size):
            return [2] * self.output_size

        else:
            return [int(max(input_dims))] * self.output_size

    def dagger(self):
        return _ClassicalFunctionKraus(
            self.function, self.cod, self.dom, not self.is_dagger
        )


class _BinaryMatrixBoxKraus(optyx.Box):
    """
    Represents a linear transformation over
    GF(2) using matrix multiplication.

    Example
    -------
    >>> from optyx.zx import X
    >>> from optyx.optyx import Scalar
    >>> xor = X(2, 1) @ Scalar(np.sqrt(2))
    >>> matrix = [[1, 1]]
    >>> m_res = BinaryMatrixBox(matrix).to_tensor().eval().array
    >>> xor_res = xor.to_zw().to_tensor().eval().array
    >>> assert np.allclose(m_res, xor_res)

    """

    def __init__(self, matrix: np.ndarray, is_dagger: bool = False):

        matrix = np.array(matrix)
        if len(matrix.shape) == 1:
            matrix = matrix.reshape(1, -1)

        cod = optyx.Bit(len(matrix[0])) if is_dagger else optyx.Bit(len(matrix))
        dom = optyx.Bit(len(matrix)) if is_dagger else optyx.Bit(len(matrix[0]))

        super().__init__("LogicalMatrix", dom, cod)

        self.matrix = matrix
        self.is_dagger = is_dagger

    def to_zw(self):
        return self

    def truncation(
        self, input_dims: List[int], output_dims: List[int]
    ) -> tensor.Box:

        if self.is_dagger:
            input_dims, output_dims = output_dims, input_dims

        def f(x):
            if not isinstance(x, np.ndarray):
                x = np.array(x, dtype=np.uint8)
            if len(x.shape) == 1:
                x = x.reshape(-1, 1)
            A = np.array(self.matrix, dtype=np.uint8)

            return list(((A @ x) % 2).reshape(1, -1)[0])

        classical_function = _ClassicalFunctionKraus(f, self.dom, self.cod)

        if self.is_dagger:
            return classical_function.truncation(
                input_dims, output_dims
            ).dagger()
        return classical_function.truncation(input_dims, output_dims)

    def determine_output_dimensions(self,
                                    input_dims: List[int]) -> List[int]:
        return _ClassicalFunctionKraus(
            None, self.dom, self.cod
        ).determine_output_dimensions(input_dims)

    def dagger(self):
        return _BinaryMatrixBoxKraus(self.matrix, not self.is_dagger)


class _BitControlledBoxKraus(optyx.Box):
    """
    A box controlled by a bit that switches between two boxes:
    - action_box: the box that is applied when the control bit is 1
    - default_box: the box that is applied when
    the control bit is 0 (default is Id)

    Example
    -------
    >>> from optyx.lo import Phase
    >>> from optyx.optyx import PhotonThresholdDetector, Mode
    >>> from optyx.zw import Create
    >>> action = Phase(0.1)
    >>> default = ZBox(1, 1, lambda x: 1)
    >>> action_result = action.to_zw().to_tensor().eval().array
    >>> default_result = default.to_zw().to_tensor().eval().array
    >>> action_test = ((Create(1) >> PhotonThresholdDetector()) @
    ...         Mode(len(action.cod)) >>
    ...         BitControlledBox(action)).to_zw().to_tensor().eval().array
    >>> default_test = ((Create(0) >> PhotonThresholdDetector()) @
    ...         Mode(len(default.cod)) >>
    ...         BitControlledBox(default)).to_zw().to_tensor().eval().array
    >>> assert np.allclose(action_result, action_test)
    >>> assert np.allclose(default_result, default_test)
    """

    def __init__(
        self,
        action_box: optyx.Box,
        default_box: optyx.Box = None,
        is_dagger: bool = False,
    ):

        if default_box is None:
            default_box = Id(action_box.dom)

        assert (
            action_box.dom == default_box.dom
            and action_box.cod == default_box.cod
        ), "action_box and default_box must have the same domain and codomain"
        assert len(action_box.dom) == len(
            action_box.cod
        ), "action_box must have the same number of inputs and outputs"

        dom = action_box.cod if is_dagger else optyx.Bit(1) @ action_box.dom
        cod = optyx.Bit(1) @ action_box.cod if is_dagger else action_box.cod

        if hasattr(action_box, "name"):
            box_name = action_box.name + "_controlled"
        else:
            box_name = "controlled_box"

        action_box = action_box.to_zw()
        default_box = default_box.to_zw()

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
                >> optyx.truncation_tensor(
                    default_box_tensor.cod.inside, output_dims
                )
            )
            .eval()
            .array.reshape(array[0, :, :].shape)
        )

        array[1, :, :] = (
            (
                action_box_tensor
                >> optyx.truncation_tensor(
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

    def to_zw(self):
        return self

    def dagger(self):
        return _BitControlledBoxKraus(
            self.action_box, self.default_box, not self.is_dagger
        )

    def conjugate(self):
        return _BitControlledBoxKraus(
            self.action_box.conjugate(),
            self.default_box.conjugate(),
            self.is_dagger,
        )


class _ControlledPhaseShiftKraus(optyx.Box):
    """
    A controlled phase shift on modes, where the control
    is a natural number and
    the phase applied is determined by a user-defined function.

    The function maps each control value to a list
    of real values (interpreted as 2π multiples of phase shifts).

    Example
    -------
    >>> from optyx.optyx import Id
    >>> from optyx.zw import Create
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

        dom = optyx.Mode(n_modes) if is_dagger else optyx.Mode(n_modes + n_control_modes)
        cod = optyx.Mode(n_modes + n_control_modes) if is_dagger else optyx.Mode(n_modes)

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
            np.meshgrid(*[range(i) for i in input_dims[:self.n_control_modes]]),
        ).T.reshape(-1, len(input_dims[:self.n_control_modes]))

        for i in input_combinations:
            fx = self.function(i)
            zbox = zw.Id(0)
            for y in fx:
                zbox @= zw.ZBox(
                    1, 1, lambda x, y=y: np.exp(2 * np.pi * 1j * y) ** x
                )

            zbox = zbox.to_tensor(input_dims[self.n_control_modes:])
            array[i, :] = (
                (zbox >> optyx.truncation_tensor(zbox.cod.inside, output_dims))
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
            return [optyx.MAX_DIM]*self.n_control_modes + input_dims
        return input_dims[self.n_control_modes:]

    def to_zw(self):
        return self

    def dagger(self):
        return _ControlledPhaseShiftKraus(
            self.function, self.n_modes,
            self.n_control_modes, not self.is_dagger
        )

    def conjugate(self):
        return _ControlledPhaseShiftKraus(
            self.function, self.n_modes, self.n_control_modes, self.is_dagger
        )