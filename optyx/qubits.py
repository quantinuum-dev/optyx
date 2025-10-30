"""
Overview
--------

Operators on qubits. Intented to be defined
via ZX-calculus or using tket or discopy circuits.


Circuits (from tket, discopy, or PyZX)
---------------------------------------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:


    Circuit
    QubitChannel


Classical-quantum
------------------------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:


    Encode
    Measure
    Discard

ZX
------------------------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:


    Z
    X
    H
    Scalar
    Ket
    Bra

Errors
------------------------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:


    BitFlipError
    DephasingError


Examples of usage
------------------

**Pure ZX diagrams**

The main way of defining qubit maps in :code:`optyx` is using ZX diagrams.
In particular, in the setting of photonic quantum computing, we are interested
in using them to represent graph states for MBQC or fusion-based computing.
The table below pairs common **single- and two-qubit gates** with a ZX
diagrams. For example we have the following translation from common
quantum gates to ZX diagrams:

======================  ===========================================
Gate                    ZX diagram
======================  ===========================================
Identity ``I``          ``Id(1)``
Phase gate ``Rz(a)``    ``Z(1, 1, a)``
Phase gate ``Rx(b)``    ``X(1, 1, b)``
Hadamard ``H``          ``H()``
CNOT (CX)               ``(Id(1) @ Z(1, 2)) >> (X(2, 1) @ Id(1))``
Controlled-Z (CZ)       ``(Id(1) @ Z(1, 2)) >> Id(1) @ H() @ Id(1) >> (Z(2, 1) @ Id(1))``
SWAP                    ``(X(1, 2) @ Id(1)) >> (Id(1) @ Z(2, 1)) >> (X(1, 2) @ Id(1))``
======================  ===========================================


A single-qubit phase gate:

>>> rz = Z(1, 1, 0.5)      # Rz(pi)
>>> rz.draw(path='docs/_static/rz.svg')

.. image:: /_static/rz.svg
   :align: center


CNOT expressed in ZX:

>>> cnot_zx = (Id(1) @ Z(1, 2)) >> (X(2, 1) @ Id(1))
>>> cnot_zx.draw(path='docs/_static/cnot_zx.svg', figsize=(4, 2))

.. image:: /_static/cnot_zx.svg
   :align: center

CZ gate expressed in ZX:

>>> cz_zx = (Id(1) @ Z(1, 2)) >> Id(1) @ H() @ Id(1) >> (Z(2, 1) @ Id(1))
>>> cz_zx.draw(path='docs/_static/cz_zx.svg', figsize=(4, 2))

.. image:: /_static/cz_zx.svg
    :align: center

**Classical-quantum ZX diagrams**

We now add new generators to the ZX-calculus to represent
mixed channels with classical data:

* :code:`Encode` (:code:`bit` -> :code:`qubit`) for encoding classical bits into qubits
* :code:`Measure` (:code:`qubit` -> :code:`bit`) for measuring qubits and obtaining classical bits
* :code:`Discard` (:code:`qubit` -> :code:`0`) for discarding qubits

We can therefore measure a quantum register and perform classical
operations on the results, such as copying or discarding bits.

>>> from optyx.classical import (
...    X as XClassical
... )
>>> quantum_state = Ket(0) >> H() >> Z(1, 1, 0.5)
>>> classical_state = quantum_state >> Measure(1)
>>> classical_state_operation = classical_state >> XClassical(1, 1, 0.5)

:code:`XClassical(1, 1, 0.5)` acts as a NOT gate on the classical bit result.

>>> classical_state_operation.draw(path='docs/_static/classical_state.svg',
... figsize=(3, 5))

.. image:: /_static/classical_state.svg
    :align: center

**Quantum teleportation**

Quantum teleportation is the canonical example of
a ZX diagram that uses classical data. In :code:`optyx`,
we can make use of controlled boxes from :code:`optyx.classical`
to correct the state of the qubit depending on the measurement result.

>>> from optyx.classical import (
...    BitControlledGate, Id as IdClassical
... )
>>> from optyx import bit
>>> teleportation = (
...     Id(1) @ Z(0, 2) @ Scalar(2**0.5)>>
...     Z(1, 2) @ Id(2) >>
...     Id(1) @ Z(2, 1) @ Id(1) >>
...     Id(1) @ H() @ Id(1) >>
...     Measure(1)**2 @ Id(1) >>
...     IdClassical(bit) @ BitControlledGate(X(1, 1, 0.5)) >>
...     BitControlledGate(Z(1, 1, 0.5))
... )

>>> teleportation.draw(path='docs/_static/teleportation_qubit.svg',
... figsize=(4, 6))

.. image:: /_static/teleportation_qubit.svg
    :align: center

This produces the same protocol as an identity operation:

>>> assert np.allclose(
...     (teleportation.double().to_tensor().to_quimb()^...).data,
...     (Id(1).double().to_tensor().to_quimb()^...).data
... )

**Interfacing with external tools**

In :code:`optyx`, we can convert circuits from
:code:`tket`, :code:`discopy`, or :code:`pyzx` to ZX diagrams.

Let us consider a simple :code:`tket` circuit that creates a
GHZ state:

>>> #import pytket
>>> #import matplotlib.pyplot as plt
>>> #from pytket.extensions.qiskit import tk_to_qiskit
>>> #from qiskit.visualization import circuit_drawer
>>> #ghz_circ = pytket.Circuit(3).H(0).CX(0, 1).CX(1, 2).measure_all()
>>> #fig = circuit_drawer(tk_to_qiskit(ghz_circ), output="mpl",
>>>   #interactive=False)
>>> #fig.savefig("docs/_static/ghz_circuit_qiskit.png")
>>> #plt.close(fig)

.. image:: /_static/ghz_circuit_qiskit.png
    :align: center

We can explicitly convert it to optyx. The resulting circuit involves
explicit manipulation of classical data.

>>> #Circuit(ghz_circ).draw(path="docs/_static/ghz_circuit_exp.svg",
>>> #figsize=(6, 9))

.. image:: /_static/ghz_circuit_exp.svg
    :align: center

We can evaluate these two curcuits.
First, let's evaluate with tket:

>>> #from pytket.extensions.qiskit import AerBackend
>>> #from pytket.utils import probs_from_counts
>>> #backend = AerBackend()
>>> #compiled_circ = backend.get_compiled_circuit(ghz_circ)
>>> #handle = backend.process_circuit(compiled_circ, n_shots=200000)
>>> #counts = backend.get_result(handle).get_counts()
>>> #tket_probs = probs_from_counts({key: np.round(v, 2) \\
>>> #for key, v in probs_from_counts(counts).items()})

Then, let us evaluate with Optyx:

>>> #from optyx import classical
>>> #circ = Circuit(ghz_circ)
>>> #circ = Ket(0)**3 @ classical.Bit(0)**3 >> circ >> Discard(3) @ bit**3
>>> #res = (circ.double().to_tensor().to_quimb()^...).data
>>> #rounded_result = np.round(res, 6)
>>> #non_zero_dict = {idx: val for idx, val
>>>   #in np.ndenumerate(rounded_result) if val != 0}

They agree:

>>> #assert tket_probs == non_zero_dict

**Interaction with photonic components**

We can use a circuit from an external package to define a
graph state, which can then be used for photonic quantum computing.
Let us use the GHZ state from the example above.

>>> #from optyx.classical import DiscardBit, Z as ZClassical
>>> #circ = Circuit(ghz_circ)
>>> #circ = Ket(0)**3 @ classical.Bit(0)**3 >> circ >> qubit**3 @ ZClassical(1, 0)**3

>>> #from optyx.photonic import DualRail, HadamardBS, XMeasurementDR
>>> #circ_photonic = circ >> DualRail(3) >> HadamardBS() @ XMeasurementDR(0.5)**2
>>> #circ_photonic.draw(path="docs/_static/ghz_circ_photonic.svg",
>>> #figsize=(6, 12))

.. image:: /_static/ghz_circ_photonic.svg
    :align: center

**Direct convertion to dual-rail encoding**

We can create a graph state as follows
(where we omit the labels):

>>> from discopy.drawing import Equation
>>> from optyx.photonic import DualRail
>>> graph = (Z(0, 2) >> Id(1) @ H() >> Id(1) @ Z(1, 2) >> \\
... Id(2) @ H() >> Id(2) @ Z(1, 2))
>>> Equation(graph >> DualRail(4), graph.to_dual_rail(), \\
... symbol="$\\mapsto$").draw(figsize=(15, 20), \\
... path="docs/_static/graph_dr_qubit.svg", draw_type_labels=False, \\
... draw_box_labels=False)

.. image:: /_static/graph_dr_qubit.svg
    :align: center

We can map ZX diagrams to dual-rail encoding.
For example, we can create a GHZ state:

>>> from discopy.drawing import Equation
>>> from optyx.photonic import DualRail
>>> from optyx.core.diagram import embedding_tensor
>>> ghz = Z(0, 3)
>>> ghz_decom = ghz.decomp()
>>> ghz_path = ghz_decom.to_dual_rail()
>>> Equation(ghz >> DualRail(3), ghz_path, \\
... symbol="$\\mapsto$").draw(figsize=(10, 10), \\
... path="docs/_static/ghz_dr.svg")

.. image:: /_static/ghz_dr.svg
    :align: center

We can also create a graph state as follows
(where we omit the labels):

>>> graph = (Z(0, 2) >> Id(1) @ H() >> Id(1) @ X(1, 2, 0.5) >> \\
... Id(2) @ H() >> Id(2) @ Z(1, 2))
>>> graph_decom = graph.decomp()
>>> graph_path = graph_decom.to_dual_rail()
>>> Equation(graph >> DualRail(4), graph_path, \\
... symbol="$\\mapsto$").draw(figsize=(10, 14), \\
... path="docs/_static/graph_dr.svg", draw_type_labels=False, \\
... draw_box_labels=False)

.. image:: /_static/graph_dr.svg
    :align: center

"""  # noqa E501

from typing import Literal
from enum import Enum
import numpy as np
from pyzx.graph.base import BaseGraph
import graphix
from discopy import quantum as quantum_discopy
from discopy import symmetric
from sympy import lambdify
# from pytket import circuit as tket_circuit
from optyx.utils.misc import explode_channel
from optyx.core import (
    channel,
    diagram,
    zx
)
from optyx.core.channel import (
    bit,
    qubit,
    Measure as MeasureChannel,
    Discard as DiscardChannel,
    Encode as EncodeChannel,
    Channel,
    Diagram
)


class ImportObjectType(Enum):
    """
    Type of object that can be imported.
    """
    ZX = "zx"
    PYZX = "pyzx"
    TKET = "tket"
    DISCOPY = "discopy"
    GRAPHIX = "graphix"


class Circuit(Diagram):
    """
    A circuit that operates on qubits.
    It can be initialised from a ZX diagram, PyZX diagram,
    a tket circuit, or a discopy circuit. This is black box circuit
    until evaluated.
    """

    def __new__(cls, circuit):
        return cls._to_optyx(circuit)

    @classmethod
    def _detect_type(cls, underlying_circuit):
        """
        Detect the type of the underlying circuit.
        """
        if isinstance(underlying_circuit,
                      quantum_discopy.circuit.Circuit):
            return ImportObjectType.DISCOPY
        if isinstance(underlying_circuit, BaseGraph):
            return ImportObjectType.PYZX
        # if isinstance(underlying_circuit, tket_circuit.Circuit):
        #     return ImportObjectType.TKET
        if isinstance(underlying_circuit, Diagram):
            return ImportObjectType.ZX
        if isinstance(underlying_circuit, graphix.pattern.Pattern):
            return ImportObjectType.GRAPHIX
        raise TypeError("Unsupported circuit type")  # pragma: no cover

    @classmethod
    def _to_optyx(cls, underlying_circuit):
        """
        Convert the circuit to an optyx channel diagram.
        """
        type_ = cls._detect_type(underlying_circuit)
        if type_ == ImportObjectType.DISCOPY:
            return cls._to_optyx_from_discopy(underlying_circuit)
        if type_ == ImportObjectType.PYZX:
            return cls._to_optyx_from_pyzx(underlying_circuit)
        # if type_ == ImportObjectType.TKET:
        #     return cls._to_optyx_from_tket(underlying_circuit)
        if type_ == ImportObjectType.ZX:
            return cls._to_optyx_from_zx(underlying_circuit)
        if type_ == ImportObjectType.GRAPHIX:
            return cls._to_optyx_from_graphix(underlying_circuit)
        raise TypeError("Unsupported circuit type")  # pragma: no cover

    # @classmethod
    # def _to_optyx_from_tket(cls, underlying_circuit):
    #     """
    #     Convert a tket circuit to an optyx channel diagram.
    #     """
    #     underlying_circuit = quantum_discopy.circuit.Circuit.from_tk(
    #         underlying_circuit, init_and_discard=False
    #     )
    #     return cls._to_optyx_from_discopy(underlying_circuit)

    @classmethod
    def _to_optyx_from_pyzx(cls, underlying_circuit):
        """
        Convert a PyZX circuit to an optyx channel diagram.
        """
        zx_diagram = zx.ZXDiagram.from_pyzx(underlying_circuit)
        return explode_channel(
            zx_diagram,
            Channel,
            Diagram
        )

    @classmethod
    def _to_optyx_from_discopy(cls, underlying_circuit):
        """
        Convert a discopy circuit to an optyx channel diagram.
        """

        # pylint: disable=invalid-name
        def ob(o):
            if o.name == "qubit":
                return qubit**len(o)
            if o.name == "bit":
                return bit**len(o)
            raise TypeError(f"Unsupported object type: {o.name}")

        return symmetric.Functor(
            ob=ob,
            ar=QubitChannel.from_discopy,
            dom=symmetric.Category(
                quantum_discopy.circuit.Ty,
                quantum_discopy.circuit.Circuit
            ),
            cod=symmetric.Category(
                channel.Ty,
                Diagram
            ),
        )(underlying_circuit)

    @classmethod
    def _to_optyx_from_zx(cls, underlying_circuit):
        return underlying_circuit

    @classmethod
    def _to_optyx_from_graphix(cls, underlying_circuit):
        """
        Convert a Graphix measurement pattern to an optyx ZX diagram.
        """
        # pylint: disable=import-outside-toplevel
        from graphix import opengraph
        from graphix import pyzx

        og = opengraph.OpenGraph.from_pattern(underlying_circuit)
        pyzx_diagram = pyzx.to_pyzx_graph(og)
        # pylint: disable=protected-access
        pyzx_diagram._inputs = tuple(pyzx_diagram._inputs)
        # pylint: disable=protected-access
        pyzx_diagram._outputs = tuple(pyzx_diagram._outputs)
        return cls._to_optyx_from_pyzx(pyzx_diagram)


class QubitChannel(Channel):
    """Qubit channel."""

    # def decomp(self):
    #     """Decompose into elementary gates."""
    #     from optyx.utils.misc import decomp_ar
    #     from discopy import symmetric
    #     return symmetric.Functor(
    #         ob=lambda x: qubit**len(x),
    #         ar=decomp_ar,
    #         cod=symmetric.Category(channel.Ty, channel.Diagram),
    #     )(self)

    # def to_dual_rail(self):
    #     """Convert to dual-rail encoding."""
    #     from optyx.utils.misc import ar_zx2path
    #     from optyx import qmode
    #     from discopy import symmetric

    #     return symmetric.Functor(
    #         ob=lambda x: qmode**(2 * len(x)),
    #         ar=lambda ar : ar_zx2path(ar.decomp()),
    #         cod=symmetric.Category(channel.Ty, channel.Diagram),
    #     )(self)

    # pylint: disable=too-many-locals
    # pylint: disable=too-many-return-statements
    # pylint: disable=too-many-branches
    @classmethod
    def from_discopy(cls, discopy_circuit):
        """Turns gates into ZX diagrams."""
        # pylint: disable=import-outside-toplevel
        from discopy.quantum.gates import (
            Rz, Rx,
            CX, CZ, Controlled  # , Digits
        )
        from discopy.quantum.gates import (
            Bra as Bra_,
            Ket as Ket_
        )
        from discopy.quantum.gates import Scalar as GatesScalar
        # from optyx import classical

        # pylint: disable=invalid-name
        def get_perm(n):
            return sorted(sorted(list(range(n))), key=lambda i: i % 2)

        root2 = Scalar(2**0.5)
        if isinstance(discopy_circuit, (Bra_, Ket_)):
            dom, cod = (1, 0) if isinstance(discopy_circuit, Bra_) else (0, 1)
            spiders = [X(dom, cod, phase=0.5 * bit)
                       for bit in discopy_circuit.bitstring]
            return Id(0).tensor(*spiders) @ Scalar(
                pow(2, -len(discopy_circuit.bitstring) / 2)
            )
        if isinstance(discopy_circuit, (Rz, Rx)):
            return (Z if isinstance(discopy_circuit, Rz)
                    else X)(1, 1, discopy_circuit.phase)
        if isinstance(discopy_circuit,
                      Controlled) and discopy_circuit.name.startswith("CRz"):
            return (
                Z(1, 2) @ Z(1, 2, discopy_circuit.phase / 2)
                >> Id(1) @
                (X(2, 1) >> Z(1, 0, -discopy_circuit.phase / 2)) @
                Id(1) @ root2
            )
        if isinstance(discopy_circuit,
                      Controlled) and discopy_circuit.name.startswith("CRx"):
            return (
                X(1, 2) @ X(1, 2, discopy_circuit.phase / 2)
                >> Id(1) @
                (Z(2, 1) >> X(1, 0, -discopy_circuit.phase / 2)) @
                Id(1) @ root2
            )
        # if isinstance(discopy_circuit, Digits):
        #     dgrm = Diagram.id(bit**0)
        #     # pylint: disable=invalid-name
        #     for d in discopy_circuit.digits:
        #         if d > 1:
        #             raise ValueError(
        #                 "Only qubits supported. Digits must be 0 or 1."
        #             )
        #      dgrm @= classical.X(0, 1, 0.5**d) @ classical.Scalar(0.5**0.5)
        #     return dgrm
        if isinstance(discopy_circuit, quantum_discopy.CU1):
            return (
                Z(1, 2, discopy_circuit.phase) @
                Z(1, 2, discopy_circuit.phase) >>
                Id(1) @
                (X(2, 1) >> Z(1, 0, -discopy_circuit.phase)) @
                Id(1)
            )
        if isinstance(discopy_circuit, GatesScalar):
            return Scalar(discopy_circuit.data)
        if isinstance(discopy_circuit,
                      Controlled) and discopy_circuit.distance != 1:
            # pylint: disable=protected-access
            return Circuit(discopy_circuit._decompose())
        # if isinstance(discopy_circuit, quantum_discopy.Discard):
        #     return Discard(len(discopy_circuit.dom))
        # if isinstance(discopy_circuit, quantum_discopy.Measure):
        #     no_qubits = sum([1 if i.name == "qubit" else
        #                      0 for i in discopy_circuit.dom])
        #     dgrm = Measure(no_qubits)
        #     if discopy_circuit.override_bits:
        #         dgrm @= DiscardChannel(bit**no_qubits)
        #     if discopy_circuit.destructive:
        #         return dgrm
        #     dgrm >>= classical.CopyBit(2)**no_qubits
        #     dgrm >>= Diagram.permutation(
        #         get_perm(2 * no_qubits), bit**(2 * no_qubits)
        #     )
        #     dgrm >>= (
        #         Encode(no_qubits) @
        #         Diagram.id(bit**no_qubits)
        #     )
        #     return dgrm
        if isinstance(discopy_circuit, quantum_discopy.Encode):
            raise NotImplementedError(
                "Converting Encode to QubitChannel is not implemented."
            )
        standard_gates = {
            quantum_discopy.H: H(),
            quantum_discopy.Z: Z(1, 1, 0.5),
            quantum_discopy.X: X(1, 1, 0.5),
            quantum_discopy.Y: Z(1, 1, 0.5) >> X(1, 1, 0.5) @ Scalar(1j),
            quantum_discopy.S: Z(1, 1, 0.25),
            quantum_discopy.T: Z(1, 1, 0.125),
            CZ: (
                Z(1, 2) @ Id(1) >>
                Id(1) @ H() @ Id(1) >>
                Id(1) @ Z(2, 1) @ root2
                ),
            CX: (
                Z(1, 2) @ Id(1) >>
                Id(1) @ X(2, 1) @ root2
                ),
        }
        return standard_gates[discopy_circuit]


class Measure(MeasureChannel):
    """
    Ideal qubit measurement (in computational basis)
    from qubit to bit.
    """

    def __init__(self, n):
        super().__init__(
            qubit**n
        )


class Discard(DiscardChannel):
    """
    Discard :math:`n` qubits.
    """

    def __init__(self, n):
        super().__init__(
            qubit**n
        )


class Encode(EncodeChannel):
    """
    Encode :math:`n` bits into :math:`n` qubits.
    """

    def __init__(self, n):
        super().__init__(
            bit**n
        )


# pylint: disable=invalid-name
class Z(Channel):
    """Z spider."""

    tikzstyle_name = "Z"
    color = "green"
    draw_as_spider = True

    def __init__(self, n_legs_in, n_legs_out, phase=0):
        kraus = zx.Z(n_legs_in, n_legs_out, phase)
        super().__init__(
            f"Z({phase})",
            kraus,
            qubit**n_legs_in,
            qubit**n_legs_out,
        )
        self.data = phase
        self.phase = phase

    def lambdify(self, *symbols, **kwargs):
        return lambda *xs: type(self)(
            len(self.dom),
            len(self.cod),
            lambdify(symbols, self.phase, **kwargs)(*xs)
        )

    def _decomp(self):
        n, m = len(self.dom), len(self.cod)
        phase = self.phase
        rot = Id(1) if phase == 0 else Z(1, 1, phase)
        if n == 0:
            return X(0, 1) >> H() >> rot >> self._make_spiders(m)
        if m == 0:
            return X(1, 0) << H() << rot << self._make_spiders(n).dagger()
        return self._make_spiders(n).dagger() >> rot >> self._make_spiders(m)

    @staticmethod
    def _make_spiders(n):
        """Constructs the Z spider 1 -> n from spiders 1 -> 2.

        >>> assert len(Z._make_spiders(6)) == 5
        """
        from optyx import qubits

        spider = qubits.Id(1)
        for k in range(n - 1):
            spider = spider >> qubits.Z(1, 2) @ qubits.Id(k)
        return spider

    def _to_dual_rail(self):
        """Convert to dual-rail encoding."""

        from optyx import (
            photonic,
            qmode,
            classical
        )
        from optyx.core import zw
        create = photonic.Create(1)
        annil = photonic.Select(1)
        comonoid = Channel("Split", zw.Split(2))
        monoid = Channel("Merge", zw.Merge(2))
        BS = photonic.BS

        n, m = len(self.dom), len(self.cod)
        phase = self.phase
        if (n, m) == (0, 1):
            return create >> comonoid
        if (n, m) == (1, 1):
            return qmode @ photonic.Phase(phase)
        if (n, m, phase) == (2, 1, 0):
            return (
                qmode @
                (monoid >> annil) @
                qmode
                )
        if (n, m, phase) == (1, 2, 0):
            plus = create >> comonoid
            bot = (
                (plus >> qmode @ plus @ qmode) @
                (qmode @ plus @ qmode)
            )
            mid = qmode**2 @ BS.dagger() @ BS @ qmode**2
            fusion = (
                qmode @ plus.dagger() @
                qmode >> plus.dagger()
            )
            return (
                bot >> mid >> (qmode**2 @
                               fusion @ qmode**2)
                )
        raise NotImplementedError(f"No translation of {self} in QPath.")

    def dagger(self):
        return type(self)(
            len(self.cod),
            len(self.dom),
            -self.phase
        )


# pylint: disable=invalid-name
class X(Channel):
    """X spider."""

    tikzstyle_name = "X"
    color = "red"
    draw_as_spider = True

    def __init__(self, n_legs_in, n_legs_out, phase=0):
        kraus = zx.X(n_legs_in, n_legs_out, phase)
        super().__init__(
            f"X({phase})",
            kraus,
            qubit**n_legs_in,
            qubit**n_legs_out,
        )
        self.data = phase
        self.phase = phase

    def lambdify(self, *symbols, **kwargs):
        return lambda *xs: type(self)(
            len(self.dom),
            len(self.cod),
            lambdify(symbols, self.phase, **kwargs)(*xs),
        )

    def _decomp(self):
        n, m = len(self.dom), len(self.cod)
        phase = self.phase
        if (n, m) in ((1, 0), (0, 1)):
            return self
        box = (
            Id(0).tensor(*[H()] * n) >>
            Z(n, m, phase) >>
            Id(0).tensor(*[H()] * m)
        )
        return box.decomp()

    def _to_dual_rail(self):  # pragma: no cover
        """Convert to dual-rail encoding."""
        from optyx import (
            photonic,
            classical
        )

        root2 = photonic.Scalar(2**0.5)
        unit = photonic.Create(0)
        counit = photonic.Select(0)
        create = photonic.Create(1)
        annil = photonic.Select(1)
        BS = photonic.BS
        n, m = len(self.dom), len(self.cod)
        phase = 1 + self.phase if self.phase < 0 else self.phase
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
        raise NotImplementedError(f"No translation of {self} in QPath.")

    def dagger(self):
        return type(self)(
            len(self.cod),
            len(self.dom),
            -self.phase
        )


# pylint: disable=invalid-name
class H(Channel):
    """Hadamard gate."""

    tikzstyle_name = "H"
    color = "yellow"

    def __init__(self):
        super().__init__(
            "H",
            zx.H,
            qubit,
            qubit,
        )

    def _decomp(self):
        return H()

    def _to_dual_rail(self):
        """Convert to dual-rail encoding."""
        from optyx import photonic
        return photonic.HadamardBS()

    def dagger(self):
        return self


class Scalar(Channel):
    """
    Scalar.
    """

    def __init__(self, value: float):
        super().__init__(
            f"Scalar({value})",
            zx.scalar(value),
            qubit**0,
            qubit**0,
        )
        self.data = value

    def _decomp(self):
        return Scalar(self.data)

    def _to_dual_rail(self):
        """Convert to dual-rail encoding."""
        from optyx import photonic
        return photonic.Scalar(self.data)

    def dagger(self):
        return type(self)(np.conj(self.data))


class BitFlipError(Channel):
    """
    Represents a bit-flip error channel.
    """

    def __init__(self, prob):
        x_error = zx.X(1, 2) >> zx.Id(1) @ zx.ZBox(
            1, 1, np.sqrt((1 - prob) / prob)
        ) @ zx.scalar(np.sqrt(prob * 2))
        super().__init__(
            name=f"BitFlipError({prob})",
            kraus=x_error,
            dom=qubit,
            cod=qubit,
            env=diagram.Bit(1),
        )

    def dagger(self):
        return self


class DephasingError(Channel):
    """
    Represents a quantum dephasing error channel.
    """
    def __init__(self, prob):
        z_error = (
            zx.H
            >> zx.X(1, 2)
            >> zx.H
            @ zx.ZBox(1, 1, np.sqrt((1 - prob) / prob))
            @ zx.scalar(np.sqrt(prob * 2))
        )
        super().__init__(
            name=f"DephasingError({prob})",
            kraus=z_error,
            dom=qubit,
            cod=qubit,
            env=diagram.Bit(1),
        )

    def dagger(self):
        return self


class Ket(Channel):
    """Computational basis state for qubits"""

    def __init__(
        self, value: Literal[0, 1, "+", "-"], cod: channel.Ty = qubit
    ) -> None:
        spider = zx.X if value in (0, 1) else zx.Z
        phase = 0 if value in (0, "+") else 0.5
        kraus = spider(0, 1, phase) @ diagram.Scalar(1 / np.sqrt(2))
        super().__init__(f"|{value}>", kraus, cod=cod)


class Bra(Channel):
    """Post-selected measurement for qubits"""

    def __init__(
        self, value: Literal[0, 1, "+", "-"], dom: channel.Ty = qubit
    ) -> None:
        spider = zx.X if value in (0, 1) else zx.Z
        phase = 0 if value in (0, "+") else 0.5
        kraus = spider(1, 0, phase) @ diagram.Scalar(1 / np.sqrt(2))
        super().__init__(f"<{value}|", kraus, dom=dom)


def Id(n):
    """
    Qubit identity wire.
    """
    return Diagram.id(n) if \
        isinstance(n, channel.Ty) else Diagram.id(qubit**n)
