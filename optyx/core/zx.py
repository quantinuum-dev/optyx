"""
Overview
--------

ZX diagrams, to and from conversions with :code:`pyzx`,
evaluation with to_tensor via :code:`quimb`,
mapping to post-selected linear
optical circuits :code:`zx_to_path`.


Generators
-------------
.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Box
    Spider
    Z
    X

Functions
----------
.. autosummary::
    :template: function.rst
    :nosignatures:
    :toctree:

    zx_to_path
    decomp
    zx2path
    circuit2zx


Examples of usage
------------------

We can map ZX diagrams to :class:`path` diagrams using
dual-rail encoding. For example, we can create a GHZ state:

>>> from discopy.drawing import Equation
>>> from optyx.core.diagram import dual_rail, embedding_tensor
>>> ghz = Z(0, 3)
>>> ghz_decom = decomp(ghz)
>>> ghz_path = zx_to_path(ghz_decom)
>>> Equation(ghz >> dual_rail(3), ghz_path, \\
... symbol="$\\mapsto$").draw(figsize=(10, 10), \\
... path="docs/_static/ghz_dr.svg")

.. image:: /_static/ghz_dr.svg
    :align: center

We can also create a graph state as follows
(where we omit the labels):

>>> graph = (Z(0, 2) >> Id(1) @ H >> Id(1) @ Z(1, 2) >> \\
... Id(2) @ H >> Id(2) @ Z(1, 2))
>>> graph_decom = decomp(graph)
>>> graph_path = zx_to_path(graph_decom)
>>> Equation(graph >> dual_rail(4), graph_path, \\
... symbol="$\\mapsto$").draw(figsize=(10, 14), \\
... path="docs/_static/graph_dr.svg", draw_type_labels=False, \\
... draw_box_labels=False)

.. image:: /_static/graph_dr.svg
    :align: center

We can check that both diagrams produce the same tensors (we need to
ensure the tensor dimensions match):

>>> assert np.allclose(graph_path.to_tensor().eval().array, \\
... ((graph >> dual_rail(4)).to_tensor() >> \\
... (tensor.Id(Dim(*[2]*7)) @ embedding_tensor(1, 5))).eval().array)

As shown in the example above, we need to decompose a ZX diagram
into more elementary spiders before mapping it to a path diagram.
More explicitely:

>>> dgr = Z(2, 1, 0.25) >> X(1, 1, 0.35)
>>> print(decomp(dgr))
Z(2, 1) >> Z(1, 1, 0.25) >> H >> Z(1, 1, 0.35) >> H
>>> print(zx2path(decomp(dgr))[:2])
mode @ W[::-1] @ mode >> mode @ Select(1) @ mode
>>> assert zx2path(decomp(dgr)) == zx_to_path(dgr)

Evaluating ZX diagrams using PyZX or via the dual rail
encoding is equivalent.

>>> ket = lambda *xs: Id(diagram.Bit(0)).tensor(\\
...         *[X(0, 1, 0.5 if x == 1 else 0) for x in xs])
>>> cnot = Z(1, 2) @ Id(1) >> Id(1) @ X(2, 1)
>>> control = lambda x: ket(x) @ Id(1) >> cnot >> ket(x).dagger() @ Id(1)
>>> assert np.allclose(zx_to_path(control(0)).to_path().eval(1).array, \\
...                    control(0).to_pyzx().to_tensor())
>>> assert np.allclose(zx_to_path(control(1)).to_path().eval(1).array, \\
...                    control(1).to_pyzx().to_tensor())
>>> cz = lambda phi: cnot >> Z(1, 1, phi) @ H
>>> amplitude = ket(1, 1) >> cz(0.7) >> ket(1, 1).dagger()
>>> assert np.allclose(zx_to_path(amplitude).to_path().eval().array, \\
...                    amplitude.to_pyzx().to_tensor())

Corner case where :code:`to_pyzx` and
:code:`zx_to_path` agree only up to global
phase.

>>> diagram = X(0, 2) @ Z(0, 1, 0.25) @ scalar(1/2)\\
...     >> Id(1) @ Z(2, 1) >> X(2, 0, 0.35)
>>> print(decomp(diagram)[:3])
X(0, 1) >> H >> Z(1, 2)
>>> print(zx_to_path(diagram)[:2])
Create(1) >> mode @ Create((0,))
>>> pyzx_ampl = diagram.to_pyzx().to_tensor()
>>> assert np.allclose(pyzx_ampl, zx_to_path(diagram).to_path().eval().array)

The array properties of Z and X spiders agree with PyZX.

>>> z = Z(n_legs_in = 2, n_legs_out = 2, phase = 0.5)
>>> assert np.allclose(z.array.flatten(), z.to_pyzx().to_tensor().flatten())

>>> x = X(n_legs_in = 2, n_legs_out = 2, phase = 0.5)
>>> assert np.allclose(x.array.flatten(), x.to_pyzx().to_tensor().flatten())
"""

from math import pi
from typing import List

import numpy as np
from discopy import quantum
from discopy import symmetric
from discopy import cat
from discopy.quantum.gates import Bra, Ket, Rz, Rx, CX, CZ, Controlled
from discopy.quantum.circuit import qubit, Circuit
from discopy.quantum.gates import Scalar as GatesScalar
from discopy.cat import Category
from discopy.utils import factory_name
from discopy.frobenius import Dim
from discopy import tensor
from optyx.core import diagram, zw


class Box(diagram.Box):
    """A box in a ZX diagram."""

    def __init__(self, name, dom, cod, **params):
        if isinstance(dom, int):
            dom = diagram.Bit(dom)
        if isinstance(cod, int):
            cod = diagram.Bit(cod)
        super().__init__(name=name, dom=dom, cod=cod, **params)

    @property
    def array(self):
        if self.data is not None:
            return self.data
        raise NotImplementedError(f"Array not implemented for {self}.")

    @array.setter
    def array(self, value):
        self._array = value

    def conjugate(self):
        raise NotImplementedError

    def determine_output_dimensions(self, input_dims: list[int]) -> list[int]:
        """Determine the output dimensions"""
        return [2 for _ in range(len(self.cod))]

    def truncation(self, input_dims=None, output_dims=None) -> tensor.Box:
        "Return a :class:`tensor.Box` with the underlying array"
        out_dims = Dim(*[2 for i in range(len(self.cod))])
        in_dims = Dim(*[2 for i in range(len(self.dom))])

        return tensor.Box(self.name, in_dims, out_dims, self.array)

    def __eq__(self, other):
        return (
            isinstance(other, type(self))
            and self.name == other.name
            and self.dom == other.dom
            and np.all(self.data == other.data)
        )


class Spider(diagram.Spider, Box):
    """Abstract spider box."""

    def __init__(self, n_legs_in, n_legs_out, phase=0):
        super().__init__(n_legs_in, n_legs_out, diagram.Bit(1), phase)
        factory_str = type(self).__name__
        phase_str = f", {self.phase}" if self.phase else ""
        self.name = f"{factory_str}({n_legs_in}, {n_legs_out}{phase_str})"
        self.n_legs_in, self.n_legs_out = n_legs_in, n_legs_out

    def conjugate(self):
        return Spider(self.n_legs_in, self.n_legs_out, -self.phase)

    def __repr__(self):
        return str(self).replace(type(self).__name__, factory_name(type(self)))

    def subs(self, *args):
        phase = cat.rsubs(self.phase, *args)
        return type(self)(len(self.dom), len(self.cod), phase=phase)

    def grad(self, var, **params):
        """Gradient with respect to a variable."""

        if var not in self.free_symbols:
            return diagram.Sum((), self.dom, self.cod)
        gradient = self.phase.diff(var)
        gradient = complex(gradient) if not gradient.free_symbols else gradient
        return diagram.Scalar(pi * gradient) @ type(self)(
            len(self.dom), len(self.cod), self.phase + 0.5
        )

    def dagger(self):
        return type(self)(len(self.cod), len(self.dom), -self.phase)

    def rotate(self, left=False):
        del left
        return type(self)(len(self.cod), len(self.dom), self.phase)

    @property
    def array(self):
        return self.truncation().eval().array

    def truncation(self, input_dims = None, output_dims = None):
        return super().truncation([2]*self.n_legs_in, [2]*self.n_legs_out)


class ZBox(Spider):
    """Z box."""

    tikzstyle_name = "ZBox"

    def conjugate(self):
        return ZBox(self.n_legs_in, self.n_legs_out, np.conjugate(self.phase))

    def truncation(self, input_dims=None, output_dims=None) -> tensor.Box:
        return zw.ZBox(
            self.n_legs_in, self.n_legs_out, [1, self.phase]
        ).truncation([2] * self.n_legs_in)


class Z(Spider):
    """Z spider."""

    tikzstyle_name = "Z"

    def conjugate(self):
        return Z(self.n_legs_in, self.n_legs_out, -self.phase)

    def truncation(self, input_dims=None, output_dims=None) -> tensor.Box:
        return zw.ZBox(
            self.n_legs_in, self.n_legs_out,
            [1, np.exp(1j * self.phase * 2 * np.pi)]
        ).truncation([2] * self.n_legs_in)


class X(Spider):
    """X spider."""

    tikzstyle_name = "X"
    color = "red"

    def conjugate(self):
        return X(self.n_legs_in, self.n_legs_out, -self.phase)

    def truncation(self, input_dims=None, output_dims=None) -> tensor.Box:
        in_hadamards = tensor.Id(1)
        for i in range(self.n_legs_in):
            in_hadamards @= H.truncation()

        out_hadamards = tensor.Id(1)
        for _ in range(self.n_legs_out):
            out_hadamards @= H.truncation()
        return (
            in_hadamards
            >> Z(self.n_legs_in, self.n_legs_out, self.phase).truncation()
            >> out_hadamards
        )


def scalar(data):
    """Returns a scalar."""
    return diagram.Scalar(data)


def make_spiders(n):
    """Constructs the Z spider 1 -> n from spiders 1 -> 2.

    >>> assert len(make_spiders(6)) == 5
    """
    spider = Id(1)
    for k in range(n - 1):
        spider = spider >> Z(1, 2) @ Id(k)
    return spider


def decomp_ar(box):
    """
    Decomposes a ZX diagram into Z spiders
    with at most two inputs/outputs and hadamards.

    >>> assert len(decomp(X(2, 2, 0.25))) == 7
    """
    n, m = len(box.dom), len(box.cod)
    if isinstance(box, X):
        phase = box.phase
        if (n, m) in ((1, 0), (0, 1)):
            return box
        box = (
            Id(0).tensor(*[H] * n) >> Z(n, m, phase) >> Id(0).tensor(*[H] * m)
        )
        return decomp(box)
    if isinstance(box, Z):
        phase = box.phase
        rot = Id(1) if phase == 0 else Z(1, 1, phase)
        if n == 0:
            return X(0, 1) >> H >> rot >> make_spiders(m)
        if m == 0:
            return X(1, 0) << H << rot << make_spiders(n).dagger()
        return make_spiders(n).dagger() >> rot >> make_spiders(m)
    return box


decomp = symmetric.Functor(
    ob=lambda x: diagram.Bit(len(x)),
    ar=decomp_ar,
    cod=symmetric.Category(diagram.Bit, diagram.Diagram),
)


def Id(n):
    return diagram.Diagram.id(n) if isinstance(n, diagram.Ty) \
          else diagram.Diagram.id(diagram.Bit(n))


def ar_zx2path(box):
    """Mapping from ZX generators to QPath diagrams

    >>> zx2path(decomp(X(0, 1) @ X(0, 1) >> Z(2, 1))).to_path().eval()
    Amplitudes([2.+0.j, 0.+0.j], dom=1, cod=2)
    """
    from optyx.photonic import HadamardBS, Phase, BS

    unit = zw.Create(0)
    counit = zw.Select(0)
    create = zw.Create(1)
    annil = zw.Select(1)
    comonoid = zw.Split(2)
    monoid = zw.Merge(2)
    BS = BS.get_kraus()

    n, m = len(box.dom), len(box.cod)
    if isinstance(box, diagram.Scalar):
        return diagram.Scalar(box.data)
    if isinstance(box, X):
        phase = 1 + box.phase if box.phase < 0 else box.phase
        if (n, m, phase) == (0, 1, 0):
            return create @ unit @ root2
        if (n, m, phase) == (0, 1, 0.5):
            return unit @ create @ root2
        if (n, m, phase) == (1, 0, 0):
            return annil @ counit @ root2
        if (n, m, phase) == (1, 0, 0.5):
            return counit @ annil @ root2
        if (n, m, phase) == (1, 1, 0.25):
            return BS.dagger()
        if (n, m, phase) == (1, 1, -0.25):
            return BS
    if isinstance(box, Z):
        phase = box.phase
        if (n, m) == (0, 1):
            return create >> comonoid
        if (n, m) == (1, 1):
            return Id(diagram.Mode(1)) @ Phase(phase).get_kraus()
        if (n, m, phase) == (2, 1, 0):
            return Id(diagram.Mode(1)) @ (monoid >> annil) @ Id(diagram.Mode(1))
        if (n, m, phase) == (1, 2, 0):
            plus = create >> comonoid
            bot = (plus >> Id(diagram.Mode(1)) @ plus @ Id(diagram.Mode(1))) @ (
                Id(diagram.Mode(1)) @ plus @ Id(diagram.Mode(1))
            )
            mid = Id(diagram.Mode(2)) @ BS.dagger() @ BS @ Id(diagram.Mode(2))
            fusion = Id(diagram.Mode(1)) @ plus.dagger() @ Id(diagram.Mode(1)) >> plus.dagger()
            return bot >> mid >> (Id(diagram.Mode(2)) @ fusion @ Id(diagram.Mode(2)))
    if box == H:
        return HadamardBS().get_kraus()
    raise NotImplementedError(f"No translation of {box} in QPath.")


zx2path = symmetric.Functor(
    ob=lambda x: diagram.Mode(2 * len(x)),
    ar=ar_zx2path,
    cod=symmetric.Category(diagram.Mode, diagram.Diagram),
)


def zx_to_path(diagram: diagram.Diagram) -> diagram.Diagram:
    """
    Dual-rail encoding of any ZX diagram as a QPath diagram.
    """
    return zx2path(decomp(diagram))


root2 = scalar(2**0.5)


H = Box("H", 1, 1)
H.dagger = lambda: H
H.conjugate = lambda: H
H.draw_as_spider = True
(H.drawing_name, H.tikzstyle_name,) = (
    "",
    "H",
)
H.data = np.array([[1, 1], [1, -1]]) / 2**0.5
H.color, H.shape = "yellow", "rectangle"

SWAP = diagram.Swap(diagram.bit, diagram.bit)
SWAP.array = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])


def swap_truncation(diagram, _, __):
    return tensor.Box(diagram.name, Dim(2, 2), Dim(2, 2), diagram.array)


SWAP.truncation = swap_truncation


class And(diagram.Box):
    """
    Reversible classical AND gate on n bits.

    This gate acts as a classical operation that maps:
        - (1, 1) ↦ 1
        - Otherwise ↦ 0

    Example
    -------
    >>> and_box = And().to_tensor(input_dims=[2, 2]).eval().array
    >>> import numpy as np
    >>> assert np.isclose(and_box[1, 1, 1], 1.0)
    >>> for a in [0, 1]:
    ...     for b in [0, 1]:
    ...         expected = int(a & b)
    ...         assert np.isclose(and_box[a, b, expected], 1.0)
    """

    def __init__(self, n=2, is_dagger: bool = False):
        super().__init__("And", diagram.Bit(n), diagram.Bit(1))
        self.is_dagger = is_dagger

    def truncation(
        self, input_dims: List[int], output_dims: List[int]
    ) -> tensor.Box:

        if self.is_dagger:
            input_dims, output_dims = output_dims, input_dims

        array = np.zeros((*input_dims, *output_dims), dtype=complex)
        array[..., 0] = 1
        all_ones = (1,) * len(input_dims)
        array[all_ones + (0,)] = 0
        array[all_ones + (1,)] = 1

        if self.is_dagger:
            return tensor.Box(
                self.name, Dim(*input_dims), Dim(*output_dims), array
            ).dagger()
        return tensor.Box(
            self.name, Dim(*input_dims), Dim(*output_dims), array
        )

    def determine_output_dimensions(self, input_dims):
        if self.is_dagger:
            return [2]*len(input_dims)
        return [2]

    def dagger(self):
        return And(not self.is_dagger)


class Or(diagram.Box):
    """
    Reversible classical OR gate on *n* bits.

    This gate acts as a classical operation that maps
        - (0, 0, …, 0) ↦ 0
        - Otherwise    ↦ 1

    Example
    -------
    >>> or_box = Or().to_tensor(input_dims=[2, 2]).eval().array
    >>> import numpy as np
    >>> assert np.isclose(or_box[0, 0, 0], 1.0)   # only all-zeros→0
    >>> for a in [0, 1]:
    ...     for b in [0, 1]:
    ...         expected = int(a | b)
    ...         assert np.isclose(or_box[a, b, expected], 1.0)
    """

    def __init__(self, n: int = 2, is_dagger: bool = False):
        super().__init__("Or", diagram.Bit(n), diagram.Bit(1))
        self.is_dagger = is_dagger

    def truncation(
        self,
        input_dims: List[int],
        output_dims: List[int],
    ) -> tensor.Box:
        """
        Build the (broadcast-sized) truth-table tensor for the OR gate.
        """
        if self.is_dagger:
            input_dims, output_dims = output_dims, input_dims

        array = np.zeros((*input_dims, *output_dims), dtype=complex)

        array[..., 1] = 1

        all_zeros = (0,) * len(input_dims)
        array[all_zeros + (1,)] = 0
        array[all_zeros + (0,)] = 1

        box = tensor.Box(self.name, Dim(*input_dims), Dim(*output_dims), array)
        return box.dagger() if self.is_dagger else box

    def determine_output_dimensions(self, input_dims: List[int]) -> List[int]:
        return [2] * len(input_dims) if self.is_dagger else [2]

    def dagger(self):
        return Or(n=self.dom.n, is_dagger=not self.is_dagger)
