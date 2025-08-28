"""
Overview
--------

Classical operators acting on **bits** and
**natural-number modes** that can be
freely composed with quantum channels in *optyx*.
The module covers

* **logic gates** on bits,
* **arithmetic** on modes (:math:`\\mathbb{N}`),
* **control boxes** that condition quantum sub-circuits on classical data,
* **classical functions** defined by a Python function, or a binary matrix,
* generators for **copying, swapping, post-selection** and **discarding**
* bits, digits, & postselection such as :func:`Bit` and :func:`Digit`.

Logic gates
-----------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    NotBit
    XorBit
    AndBit
    OrBit
    CopyBit
    SwapBit
    Z
    X
    H

Arithmetic on modes
-------------------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    AddN
    SubN
    MultiplyN
    DivideN
    Mod2
    CopyN
    SwapN

Control & routing
-----------------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    BitControlledGate
    BitControlledPhaseShift
    ClassicalFunction
    BinaryMatrix

Bits, digits, & selection
------------------------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    PostselectBit
    PostselectDigit
    DiscardBit
    DiscardMode
    Select
    Digit
    Bit
    Id
    Scalar

Examples of usage
-----------------

**1. Classical functions implemented in three ways**

We can implement classical functions:

1. Using a :code:`ClassicalFunction` with a Python function.
2. Using a :code:`BinaryMatrix` to define the transformation.
3. Using primitives such as :code:`XorBit`, :code:`AddN`, etc.

>>> xor_gate = XorBit(2)
>>>
>>> f = ClassicalFunction(lambda b: [b[0] ^ b[1]],
...                       bit**2, bit)
>>> m = BinaryMatrix([[1, 1]])   # (a,b) => a ⊕ b
>>>
>>> import numpy as np
>>> target = xor_gate.double().to_tensor().eval().array
>>> assert np.allclose(f.double().to_tensor().eval().array, target)
>>> assert np.allclose(m.double().to_tensor().eval().array, target)

**2. A bit-controlled Pauli-Z on a photonic dual-rail qubit**

The classical functions defined above can be
used to control quantum operations.
In particular, given classical measurement outcomes,
we can perform postprocessing
and feed the result into a controlled quantum gate.

>>> from optyx.photonic import DualRail, PhaseShiftDR
>>> ctrl  = Bit(1)           # 1 post-selection (classical control wire)
>>> zgate = PhaseShiftDR(0.5)
>>> cZ    = BitControlledGate(zgate)   # applies Z only when control bit = 1
>>> hybrid = ctrl @ DualRail(1) >> cZ
>>> hybrid.draw(path="docs/_static/controlled_Z.svg")

.. image:: /_static/controlled_Z.svg
   :align: center

**3. Arithmetic on natural-number modes**

>>> num = Digit(3, 2)
>>> add = AddN(2)                     # (x,y,z) => x+y+z
>>> parity = add >> Mod2()            # outputs (x+y+z) mod 2 as a bit
>>> post = PostselectBit(1)
>>> assert np.allclose(
...     (num >> parity >> post).double().to_tensor().eval().array, 1
... )

"""

from typing import Callable, List
import numpy as np
from optyx.core import (
    channel,
    control,
    zw,
    zx,
    diagram,
    path
)
from optyx import (
    bit,
    mode,
    qmode,
    qubit,
    Discard,
    CQMap,
    Channel,
    Diagram
)


class BitControlledGate(Channel):
    """
    Represents a gate that is
    controlled by a classical bit.
    It uses a `BitControlledBox` to define
    the Kraus operators for the gate.
    """
    class _BitControlledSingleBox(Channel):
        def __init__(
            self,
            control_gate,
            default_gate=None,
            classical=False
        ):
            if isinstance(control_gate, (Diagram, Channel)):
                assert control_gate.is_pure, \
                    "The input gates must be pure quantum channels"
                control_gate_single = control_gate.get_kraus()
            else:
                control_gate_single = control_gate

            if isinstance(default_gate, (Diagram, Channel)):
                assert default_gate.is_pure, \
                    "The input gates must be pure quantum channels"
                default_gate_single = default_gate.get_kraus()
            else:
                default_gate_single = default_gate

            if control_gate_single.dom[0] == diagram.bit:
                tp = qubit if not classical else bit
            else:
                tp = qmode if not classical else mode

            kraus = control.BitControlledBox(
                control_gate_single,
                default_gate_single
            )

            super().__init__(
                f"BitControlledGate({control_gate}, {default_gate})",
                kraus,
                bit @ tp**len(control_gate_single.dom),
                tp**len(control_gate_single.cod)
            )

    def __new__(cls, diag, default_box=None, is_dagger=False, classical=False):
        if default_box is not None:
            return cls._BitControlledSingleBox(
                diag, default_box, classical=classical
            ).dagger() if is_dagger else \
                 cls._BitControlledSingleBox(
                    diag,
                    default_box,
                    classical=classical
                )

        boxes = []
        for i in range(len(diag)):
            layer = diag[i]
            box = layer.inside[0][1]
            if box.cod == box.dom:
                left = layer.inside[0][0]
                right = layer.inside[0][2]
                copy = Z(1, 2)

                layers = [
                    copy @ left @ box.dom @ right,
                    channel.Diagram.permutation(
                        [
                            0, *range(2, 2+len(left)), 1,
                            *range(2+len(left), 2+len(left)+len(box.dom))
                        ], bit**2 @ left @ box.dom
                    ) @ right,
                    bit @ left @ cls._BitControlledSingleBox(box, ) @ right
                ]

                boxes.append(channel.Diagram.then(*layers))
            else:
                boxes.append(bit @ layer)

        boxes.append(Z(1, 0) @ layer.cod)
        if is_dagger:
            return channel.Diagram.then(*boxes).dagger()
        return channel.Diagram.then(*boxes)


class BitControlledPhaseShift(Channel):
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
        kraus = control.ControlledPhaseShift(function,
                                             n_modes,
                                             n_control_modes)
        super().__init__(
            "BitControlledPhaseShift",
            kraus,
            qmode**n_modes @ mode**n_control_modes,
            mode**n_modes,
        )


DiscardBit = lambda n: Discard(bit**n)  # noqa: E731
DiscardMode = lambda n: Discard(mode**n)  # noqa: E731


class ClassicalBox(CQMap):
    """
    Base class for classical boxes.
    """


class Scalar(ClassicalBox):
    """
    Scalar box in the classical circuit.
    """

    def __init__(self, value):
        super().__init__(
            f"{value}",
            diagram.Scalar(value),
            bit**0,
            bit**0
        )


class AddN(ClassicalBox):
    """
    Classical addition of n natural numbers.
    The domain of the map is n modes.
    The map will perform addition on the basis states.
    """
    def __init__(self, n):
        super().__init__(
            f"AddInt({n})",
            zw.Add(n),
            mode**n,
            mode
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
            (
                zw.Add(2).dagger() @ diagram.Id(diagram.Mode(1)) >>
                diagram.Id(diagram.Mode(1)) @ diagram.Spider(2, 0,
                                                             diagram.Mode(1))
            ),
            mode**2,
            mode
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
            zw.Multiply(),
            mode**2,
            mode
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
            zw.Divide(),
            mode**2,
            mode
        )


class Mod2(ClassicalBox):
    """
    Classical modulo 2.
    The domain of the map is a mode.
    The codomain of the map is a bit.
    The map will perform modulo 2 on the basis states.
    """
    def __init__(self):
        super().__init__(
            "ModInt",
            zw.Mod2(),
            mode,
            bit
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
            diagram.Spider(1, n, diagram.Mode(1)),
            mode,
            mode**n
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
            diagram.Swap(diagram.Mode(1), diagram.Mode(1)),
            mode**2,
            mode**2
        )


class PostselectBit(ClassicalBox):
    """
    Postselect on a bit result.
    The domain of the map is a bit.
    """
    def __init__(self, *bits):

        if not all(bit in (0, 1) for bit in bits):
            raise ValueError("Bits must be a list of 0s and 1s.")
        kraus = zx.X(1, 0, 0.5**bits[0])
        # pylint: disable=invalid-name
        for b in bits[1:]:
            kraus = kraus @ zx.X(1, 0, 0.5**b)
        kraus = kraus @ diagram.Scalar(1 / np.sqrt(2**len(bits)))
        super().__init__(
            f"PostselectBit({bits})",
            kraus,
            bit**len(bits),
            bit**0
        )


class PostselectDigit(ClassicalBox):
    """
    Postselect on a digit result.
    The domain of the map is a digit.
    """
    def __init__(self, *digits):
        if not all(isinstance(digit, int) for digit in digits):
            raise ValueError("Digits must be a list of integers.")
        super().__init__(
            f"PostselectDigit({digits})",
            zw.Select(*digits),
            mode**len(digits),
            mode**0
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
            zx.X(1, 1, 0.5),
            bit,
            bit
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
            zx.X(n, 1) @ diagram.Scalar(np.sqrt(n)),
            bit**n,
            bit
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
            zx.And(n),
            bit**2,
            bit
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
            zx.Z(1, n),
            bit,
            bit**n
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
            diagram.Swap(diagram.Bit(1), diagram.Bit(1)),
            bit**2,
            bit**2
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
            zx.Or(n),
            bit**n,
            bit
        )


# pylint: disable=invalid-name
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
            bit**n_legs_in,
            bit**n_legs_out,
        )


# pylint: disable=invalid-name
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
            bit**n_legs_in,
            bit**n_legs_out,
        )


# pylint: disable=invalid-name
class H(ClassicalBox):
    """Hadamard spider."""
    tikzstyle_name = "H"
    color = "blue"
    draw_as_spider = True

    def __init__(self):
        kraus = zx.H
        super().__init__(
            "H",
            kraus,
            bit,
            bit,
        )


class ControlChannel(ClassicalBox):
    """
    Syntactic sugar.
    Converts a classical circuit (Diagram or Box)
    into a CQMap, allowing
    it to be used as a control channel in hybrid quantum-classical systems.
    """


class ClassicalFunction(ControlChannel):
    """
    A classical function box between modes or bits,
    mapping an input list of natural numbers or
    a list of bits to a list of
    natural numbers or a list of bits.

    Example
    -------
    >>> from optyx.classical import X, Scalar
    >>> xor = X(2, 1) @ Scalar(2**0.5)
    >>> f_res = (ClassicalFunction(lambda x: [x[0] ^ x[1]],
    ...         bit**2,
    ...         bit)).double().to_tensor().eval().array
    >>> xor_res = xor.double().to_tensor().eval().array
    >>> assert np.allclose(f_res, xor_res)
    """

    def __init__(self, function, dom, cod):
        box = control.ClassicalFunctionBox(
            function,
            dom.single(),
            cod.single()
        )
        super().__init__(
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
    >>> from optyx.classical import X, Scalar
    >>> xor = X(2, 1) @ Scalar(2**0.5)
    >>> matrix = [[1, 1]]
    >>> m_res = BinaryMatrix(matrix).double().to_tensor().eval().array
    >>> xor_res = xor.double().to_tensor().eval().array
    >>> assert np.allclose(m_res, xor_res)
    """

    def __init__(self, matrix):
        box = control.BinaryMatrixBox(matrix)
        super().__init__(
            box.name,
            box,
            channel.Ty(
                *[channel.Ob._classical[ob.name] for ob in box.dom.inside]
            ),
            channel.Ty(
                *[channel.Ob._classical[ob.name] for ob in box.cod.inside]
            ),
        )


class Select(Channel):
    """
    Post-select on an occupation number.
    """
    def __init__(self, *photons: int):
        self.photons = photons
        super().__init__(
            f"Select({photons})",
            zw.Select(*photons)
        )

    def to_path(self, dtype=complex) -> path.Matrix:
        array = np.eye(len(self.photons))
        return path.Matrix[dtype](
            array, len(self.photons), 0, selections=self.photons
        )


class Digit(ClassicalBox):
    """
    Create a classical state with
    a natural number.
    """
    def __init__(self, *photons: int):
        self.photons = photons
        super().__init__(
            f"Digit({photons})",
            zw.Create(*photons),
            mode**0,
            mode**len(photons)
        )


Bit = lambda *bits: PostselectBit(*bits).dagger()  # noqa: E731


def Id(n):
    """
    Classical identity wire.
    """
    if isinstance(n, channel.Ty):
        return Diagram.id(n)
    raise TypeError(f"Expected a channel.Ty, got {type(n)}")