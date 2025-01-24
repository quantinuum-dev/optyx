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
>>> from optyx.optyx import dual_rail, embedding_tensor
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

>>> assert np.allclose(graph_path.to_zw().to_tensor().eval().array, \\
... ((graph >> dual_rail(4)).to_tensor() >> \\
... (tensor.Id(Dim(*[2]*7)) @ embedding_tensor(1, 5))).eval().array)

As shown in the example above, we need to decompose a ZX diagram
into more elementary spiders before mapping it to a path diagram.
More explicitely:

>>> diagram = Z(2, 1, 0.25) >> X(1, 1, 0.35)
>>> print(decomp(diagram))
Z(2, 1) >> Z(1, 1, 0.25) >> H >> Z(1, 1, 0.35) >> H
>>> print(zx2path(decomp(diagram))[:2])
mode @ W[::-1] @ mode >> mode @ Select(1) @ mode
>>> assert zx2path(decomp(diagram)) == zx_to_path(diagram)

Evaluating ZX diagrams using PyZX or via the dual rail
encoding is equivalent.

>>> ket = lambda *xs: Id(Bit(0)).tensor(\\
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

>>> diagram = X(0, 2) @ Z(0, 1, 0.25) @ Scalar(1/2)\\
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
from optyx import optyx
from optyx import zw
from optyx import lo
from optyx.optyx import Diagram, Bit, Sum, Swap, bit, Mode, Scalar


class Box(optyx.Box):
    """A box in a ZX diagram."""

    def __init__(self, name, dom, cod, **params):
        if isinstance(dom, int):
            dom = Bit(dom)
        if isinstance(cod, int):
            cod = Bit(cod)
        super().__init__(name=name, dom=dom, cod=cod, **params)

    @property
    def array(self):
        if self.data is not None:
            return self.data
        raise NotImplementedError(f"Array not implemented for {self}.")

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


class Spider(optyx.Spider, Box):
    """Abstract spider box."""

    def __init__(self, n_legs_in, n_legs_out, phase=0):
        super().__init__(n_legs_in, n_legs_out, Bit(1), phase)
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
            return Sum((), self.dom, self.cod)
        gradient = self.phase.diff(var)
        gradient = complex(gradient) if not gradient.free_symbols else gradient
        return Scalar(pi * gradient) @ type(self)(
            len(self.dom), len(self.cod), self.phase + 0.5
        )

    def dagger(self):
        return type(self)(len(self.cod), len(self.dom), -self.phase)

    def rotate(self, left=False):
        del left
        return type(self)(len(self.cod), len(self.dom), self.phase)


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

    @property
    def array(self):

        return_array = np.zeros(
            (2**self.n_legs_out, 2**self.n_legs_in), dtype=complex
        )

        return_array[0, 0] = 1
        return_array[
            2**self.n_legs_out - 1, 2**self.n_legs_in - 1
        ] = np.exp(1j * self.phase * 2 * np.pi)

        return return_array


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

    @property
    def array(self):

        dim_out = 2**self.n_legs_out
        dim_in = 2**self.n_legs_in

        matrix = np.ones((dim_out, dim_in), dtype=complex) / (
            2 ** ((self.n_legs_out + self.n_legs_in) / 2)
        )

        signs_out = (-1) ** np.array(
            [bin(i).count("1") for i in range(dim_out)]
        )
        signs_in = (-1) ** np.array([bin(i).count("1") for i in range(dim_in)])

        phase_term = (
            np.exp(1j * self.phase * 2 * np.pi)
            * np.outer(signs_out, signs_in)
            / (2 ** ((self.n_legs_out + self.n_legs_in) / 2))
        )

        matrix += phase_term

        return matrix


def scalar(data):
    """Returns a scalar."""
    return Scalar(data)


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
    ob=lambda x: Bit(len(x)),
    ar=decomp_ar,
    cod=symmetric.Category(Bit, Diagram),
)

unit = zw.Create(0)
counit = zw.Select(0)
create = zw.Create(1)
annil = zw.Select(1)
comonoid = zw.Split(2)
monoid = zw.Merge(2)
BS = lo.BS


def Id(n):
    return Diagram.id(n) if isinstance(n, optyx.Ty) else Diagram.id(Bit(n))


def ar_zx2path(box):
    """Mapping from ZX generators to QPath diagrams

    >>> zx2path(decomp(X(0, 1) @ X(0, 1) >> Z(2, 1))).to_path().eval()
    Amplitudes([2.+0.j, 0.+0.j], dom=1, cod=2)
    """
    n, m = len(box.dom), len(box.cod)
    if isinstance(box, Scalar):
        return zw.Scalar(box.data)
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
            return Id(Mode(1)) @ lo.Phase(phase)
        if (n, m, phase) == (2, 1, 0):
            return Id(Mode(1)) @ (monoid >> annil) @ Id(Mode(1))
        if (n, m, phase) == (1, 2, 0):
            plus = create >> comonoid
            bot = (plus >> Id(Mode(1)) @ plus @ Id(Mode(1))) @ (
                Id(Mode(1)) @ plus @ Id(Mode(1))
            )
            mid = Id(Mode(2)) @ BS.dagger() @ BS @ Id(Mode(2))
            fusion = Id(Mode(1)) @ plus.dagger() @ Id(Mode(1)) >> plus.dagger()
            return bot >> mid >> (Id(Mode(2)) @ fusion @ Id(Mode(2)))
    if box == H:
        hadamard_bs = (
            comonoid @ comonoid
            >> zw.Endo(np.sqrt(1 / 2))
            @ zw.Endo(np.sqrt(1 / 2))
            @ zw.Endo(np.sqrt(1 / 2))
            @ zw.Endo(-np.sqrt(1 / 2))
            >> zw.Id(Mode(1)) @ zw.SWAP @ zw.Id(Mode(1))
            >> monoid @ monoid
        )
        return hadamard_bs
    raise NotImplementedError(f"No translation of {box} in QPath.")


zx2path = symmetric.Functor(
    ob=lambda x: Mode(2 * len(x)),
    ar=ar_zx2path,
    cod=symmetric.Category(Mode, Diagram),
)


def zx_to_path(diagram: Diagram) -> optyx.Diagram:
    """
    Dual-rail encoding of any ZX diagram as a QPath diagram.
    """
    return zx2path(decomp(diagram))


root2 = scalar(2**0.5)


def gate2zx(box):
    """Turns gates into ZX diagrams."""
    if isinstance(box, (Bra, Ket)):
        dom, cod = (1, 0) if isinstance(box, Bra) else (0, 1)
        spiders = [X(dom, cod, phase=0.5 * bit) for bit in box.bitstring]
        return Id(Bit(0)).tensor(*spiders) @ scalar(
            pow(2, -len(box.bitstring) / 2)
        )
    if isinstance(box, (Rz, Rx)):
        return (Z if isinstance(box, Rz) else X)(1, 1, box.phase)
    if isinstance(box, Controlled) and box.name.startswith("CRz"):
        return (
            Z(1, 2) @ Z(1, 2, box.phase / 2)
            >> Id(1) @ (X(2, 1) >> Z(1, 0, -box.phase / 2)) @ Id(1) @ root2
        )
    if isinstance(box, Controlled) and box.name.startswith("CRx"):
        return (
            X(1, 2) @ X(1, 2, box.phase / 2)
            >> Id(1) @ (Z(2, 1) >> X(1, 0, -box.phase / 2)) @ Id(1) @ root2
        )
    if isinstance(box, quantum.CU1):
        return Z(1, 2, box.phase) @ Z(1, 2, box.phase) >> Id(1) @ (
            X(2, 1) >> Z(1, 0, -box.phase)
        ) @ Id(1)
    if isinstance(box, GatesScalar):
        if box.is_mixed:
            raise NotImplementedError
        return scalar(box.data)
    if isinstance(box, Controlled) and box.distance != 1:
        return circuit2zx(box._decompose())
    standard_gates = {
        quantum.H: H,
        quantum.Z: Z(1, 1, 0.5),
        quantum.X: X(1, 1, 0.5),
        quantum.Y: Z(1, 1, 0.5) >> X(1, 1, 0.5) @ scalar(1j),
        quantum.S: Z(1, 1, 0.25),
        quantum.T: Z(1, 1, 0.125),
        CZ: Z(1, 2) @ Id(1) >> Id(1) @ H @ Id(1) >> Id(1) @ Z(2, 1) @ root2,
        CX: Z(1, 2) @ Id(1) >> Id(1) @ X(2, 1) @ root2,
    }
    return standard_gates[box]


circuit2zx = quantum.circuit.Functor(
    ob={qubit: Bit(1)},
    ar=gate2zx,
    dom=Category(quantum.circuit.Ty, Circuit),
    cod=Category(Bit, Diagram),
)

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

SWAP = Swap(bit, bit)
SWAP.array = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])


def swap_truncation(diagram, _, __):
    return tensor.Box(diagram.name, Dim(2, 2), Dim(2, 2), diagram.array)


SWAP.truncation = swap_truncation
