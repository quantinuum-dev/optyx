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
from optyx.diagram.zx import (
    X as XSingle,
    Z as ZSingle,
    H as HSingle,
    decomp,
    zx2path
)
from optyx.diagram.channel import (
    Channel,
    Circuit,
    qubit,
    qmode,
    Ty
)
from optyx.utils import explode_channel
from discopy import symmetric
from discopy.cat import factory
from pytket.extensions.pyzx import tk_to_pyzx, pyzx_to_tk
from optyx.diagram.optyx import Bit, Diagram
from pyzx import extract_circuit

@factory
class QubitCircuit(Circuit):
    """Qubit circuit."""

    ty_factory = Ty

    def decomp(self):

        assert self.is_pure, "Circuit must be pure to convert to tket."

        return symmetric.Functor(
            ob=lambda x: qubit**len(x),
            ar=lambda arr: arr.decomp(),
            cod=symmetric.Category(Ty, QubitCircuit),
        )(self)

    def to_dual_rail(self):
        """Convert to dual-rail encoding."""

        assert self.is_pure, "Circuit must be pure to convert to tket."

        return symmetric.Functor(
            ob=lambda x: qmode**(2*len(x)),
            ar=lambda arr: arr.to_dual_rail(),
            cod=symmetric.Category(Ty, Circuit),
        )(self.decomp())

    @classmethod
    def from_tket(self, tket_circuit):
        """Convert from tket circuit."""
        pyzx_circuit = tk_to_pyzx(tket_circuit).to_graph()
        zx_diagram = Diagram.from_pyzx(pyzx_circuit)
        return explode_channel(
            zx_diagram,
            QubitChannel,
            QubitCircuit
        )


    def to_tket(self):
        """
        Convert to tket circuit. The circuit must be a pure circuit.
        """

        assert self.is_pure, "Circuit must be pure to convert to tket."

        kraus_maps = []
        for layer in self:
            left = layer.inside[0][0]
            right = layer.inside[0][2]
            generator = layer.inside[0][1]

            kraus_maps.append(
                Bit(len(left)) @ generator.kraus @ Bit(len(right))
            )

        return pyzx_to_tk(
            extract_circuit(
                Diagram.then(
                    *kraus_maps
                ).to_pyzx()
            ).to_basic_gates()
        )


class QubitChannel(QubitCircuit, Channel):
    """Qubit channel."""

    def decomp(self):
        decomposed = decomp(self.kraus)
        return explode_channel(
            decomposed,
            QubitChannel,
            QubitCircuit
        )

    def to_dual_rail(self):
        """Convert to dual-rail encoding."""
        kraus_path = zx2path(self.kraus)
        return explode_channel(
            kraus_path,
            Channel,
            Circuit
        )


class Z(QubitChannel):
    """Z spider."""

    tikzstyle_name = "Z"
    color = "green"

    def __init__(self, n_legs_in, n_legs_out, phase=0):
        kraus = ZSingle(n_legs_in, n_legs_out, phase)
        super().__init__(
            f"Z({n_legs_in}, {n_legs_out}, {phase})",
            kraus,
            qubit**n_legs_in,
            qubit**n_legs_out,
        )


class X(QubitChannel):
    """X spider."""

    tikzstyle_name = "X"
    color = "red"

    def __init__(self, n_legs_in, n_legs_out, phase=0):
        kraus = XSingle(n_legs_in, n_legs_out, phase)
        super().__init__(
            f"X({n_legs_in}, {n_legs_out}, {phase})",
            kraus,
            qubit**n_legs_in,
            qubit**n_legs_out,
        )


class H(QubitChannel):
    """Hadamard gate."""

    tikzstyle_name = "H"
    color = "yellow"

    def __init__(self):
        super().__init__(
            "H",
            HSingle(),
            qubit,
            qubit,
        )
