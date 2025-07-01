"""
Overview
--------

Operators on qubits. Intented to be defined
via ZX-calculus or using tket or discopy circuits.


Circuits (from tket, discopy, or PyZX)
------------------------

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

**ZX diagrams**

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


**Converting from tket**

>>> import pytket
>>> import matplotlib.pyplot as plt
>>> from pytket.extensions.qiskit import tk_to_qiskit
>>> from qiskit.visualization import circuit_drawer
>>> ghz_circ = pytket.Circuit(3).H(0).CX(0, 1).CX(1, 2).measure_all()
>>> fig = circuit_drawer(tk_to_qiskit(ghz_circ), output="mpl",
...   interactive=False)
>>> fig.savefig("docs/_static/ghz_circuit_qiskit.png")
>>> plt.close(fig)

.. image:: /_static/ghz_circuit_qiskit.png
    :align: center

The circuit appears as a black box operator.
It is only converted to optyx when we evaluate it.

>>> Circuit(ghz_circ).draw(path="docs/_static/ghz_circuit.svg")

.. image:: /_static/ghz_circuit.svg
    :align: center

We can explicitly convert it to optyx though. The resulting circuit involves
explicit manipulation of classical data.

>>> Circuit(ghz_circ)._to_optyx().draw(path="docs/_static/ghz_circuit_exp.svg")

.. image:: /_static/ghz_circuit_exp.svg
    :align: center

First, evaluate with tket:

>>> from pytket.extensions.qiskit import AerBackend
>>> from pytket.utils import probs_from_counts
>>> backend = AerBackend()
>>> compiled_circ = backend.get_compiled_circuit(ghz_circ)
>>> handle = backend.process_circuit(compiled_circ, n_shots=200000)
>>> counts = backend.get_result(handle).get_counts()
>>> tket_probs = probs_from_counts({key: np.round(v, 2) \\
... for key, v in probs_from_counts(counts).items()})

Then, evaluate with Optyx:

>>> res = (Circuit(ghz_circ).double().to_tensor().to_quimb()^...).data
>>> rounded_result = np.round(res, 6)
>>> non_zero_dict = {idx: val for idx, val
...   in np.ndenumerate(rounded_result) if val != 0}

They agree:

>>> assert tket_probs == non_zero_dict

"""

import numpy as np
from typing import Literal
from pyzx.graph.base import BaseGraph
from discopy import quantum as quantum_discopy
from discopy import symmetric
from pytket import circuit as tket_circuit
from optyx._utils import explode_channel
from optyx.core import (
    channel,
    diagram,
    zx
)
from optyx import (
    bit,
    qubit,
    Measure as MeasureChannel,
    Discard as DiscardChannel,
    Encode as EncodeChannel,
    Channel,
    Diagram
)

class Circuit(Diagram):
    """
    A circuit that operates on qubits.
    It can be initialised from a ZX diagram, PyZX diagram,
    a tket circuit, or a discopy circuit. This is black box circuit
    until evaluated.
    """

    def __init__(self, circuit):
        self._underlying_circuit = circuit
        self.name = "Circuit"
        self.type = self._detect_type()
        dom, cod = self._get_dom_cod()
        inside = Channel(
            "Circuit",
            diagram.Box(
                name="Circuit",
                dom=dom.single(),
                cod=cod.single()
            ),
            dom=dom,
            cod=cod
        )
        super().__init__(
            dom=dom,
            cod=cod,
            inside=inside.inside
        )

    def double(self):
        """
        Convert the circuit to a double circuit.
        """
        return self._to_optyx().double()

    @property
    def is_pure(self):
        """
        Check if the circuit is pure.
        """
        return self._to_optyx().is_pure

    def get_kraus(self):
        """
        Get the kraus operators of the circuit.
        """
        return self._to_optyx().get_kraus()

    def _get_dom_cod(self):
        if self.type == "tket":
            self._underlying_circuit = quantum_discopy.circuit.Circuit.from_tk(
                self._underlying_circuit
            )
            self.type = "discopy"
        if self.type == "discopy":
            assert all(o.name in ("qubit", "bit") for o in
                       self._underlying_circuit.dom), \
                        "Only bit and qubit allowed"
            assert all(o.name in ("qubit", "bit") for o in
                       self._underlying_circuit.cod), \
                        "Only bit and qubit allowed"
            return (
                channel.Ty().tensor(
                    *[qubit if o.name == "qubit" else
                      bit for o in self._underlying_circuit.dom]
                ),
                channel.Ty().tensor(
                    *[qubit if o.name == "qubit" else
                      bit for o in self._underlying_circuit.cod]
                )
            )
        if self.type == "pyzx":
            return (
                qubit**len(self._underlying_circuit.inputs()),
                qubit**len(self._underlying_circuit.outputs())
            )
        if self.type == "zx":
            return (
                self._underlying_circuit.dom,
                self._underlying_circuit.cod
            )
        raise TypeError("Unsupported circuit type")  # pragma: no cover

    def _detect_type(self):
        """
        Detect the type of the underlying circuit.
        """
        if isinstance(self._underlying_circuit,
                      quantum_discopy.circuit.Circuit):
            return "discopy"
        if isinstance(self._underlying_circuit, BaseGraph):
            return "pyzx"
        if isinstance(self._underlying_circuit, tket_circuit.Circuit):
            return "tket"
        if isinstance(self._underlying_circuit, Diagram):
            return "zx"
        raise TypeError("Unsupported circuit type")  # pragma: no cover

    def _to_optyx(self):
        """
        Convert the circuit to an optyx channel diagram.
        """
        if self.type == "discopy":
            return self._to_optyx_from_discopy()
        if self.type == "pyzx":
            return self._to_optyx_from_pyzx()
        if self.type == "tket":
            return self._to_optyx_from_tket()
        if self.type == "zx":
            return self._to_optyx_from_zx()
        raise TypeError("Unsupported circuit type")  # pragma: no cover

    def _to_optyx_from_tket(self):
        """
        Convert a tket circuit to an optyx channel diagram.
        """
        self._underlying_circuit = quantum_discopy.circuit.Circuit.from_tk(
            self._underlying_circuit
        )
        return self._to_optyx_from_discopy()

    def _to_optyx_from_pyzx(self):
        """
        Convert a PyZX circuit to an optyx channel diagram.
        """
        zx_diagram = zx.ZXDiagram.from_pyzx(self._underlying_circuit)
        return explode_channel(
            zx_diagram,
            Channel,
            Diagram
        )

    def to_dual_rail(self):
        """Convert to dual-rail encoding."""
        return self._to_optyx().to_dual_rail()

    def _to_optyx_from_discopy(self):
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
        )(self._underlying_circuit)

    def _to_optyx_from_zx(self):
        return self._underlying_circuit


class QubitChannel(Channel):
    """Qubit channel."""

    def _decomp(self):
        decomposed = zx.decomp(self.kraus)
        return explode_channel(
            decomposed,
            QubitChannel,
            Diagram
        )

    def to_dual_rail(self):
        """Convert to dual-rail encoding."""
        kraus_path = zx.zx2path(self.kraus)
        return explode_channel(
            kraus_path,
            Channel,
            Diagram
        )

    # pylint: disable=too-many-locals
    # pylint: disable=too-many-return-statements
    # pylint: disable=too-many-branches
    @classmethod
    def from_discopy(cls, discopy_circuit):
        """Turns gates into ZX diagrams."""
        # pylint: disable=import-outside-toplevel
        from discopy.quantum.gates import (
            Rz, Rx,
            CX, CZ, Controlled, Digits
        )
        from discopy.quantum.gates import (
            Bra as Bra_,
            Ket as Ket_
        )
        from discopy.quantum.gates import Scalar as GatesScalar
        from optyx import classical

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
        if isinstance(discopy_circuit, Digits):
            dgrm = Diagram.id(bit**0)
            # pylint: disable=invalid-name
            for d in discopy_circuit.digits:
                if d > 1:
                    raise ValueError(
                        "Only qubits supported. Digits must be 0 or 1."
                    )
                dgrm @= classical.X(0, 1, 0.5**d) @ classical.Scalar(0.5**0.5)
            return dgrm
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
            return Circuit(discopy_circuit._decompose())._to_optyx()
        if isinstance(discopy_circuit, quantum_discopy.Discard):
            return Discard(len(discopy_circuit.dom))
        if isinstance(discopy_circuit, quantum_discopy.Measure):
            no_qubits = sum([1 if i.name == "qubit" else
                             0 for i in discopy_circuit.dom])
            dgrm = Measure(no_qubits)
            if discopy_circuit.override_bits:
                dgrm @= DiscardChannel(bit**no_qubits)
            if discopy_circuit.destructive:
                return dgrm
            dgrm >>= classical.CopyBit(2)**no_qubits
            dgrm >>= Diagram.permutation(
                get_perm(2 * no_qubits), bit**(2 * no_qubits)
            )
            dgrm >>= (
                Encode(no_qubits) @
                Diagram.id(bit**no_qubits)
            )
            return dgrm
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


class Scalar(Channel):
    def __init__(self, value: float):
        super().__init__(
            f"Scalar({value})",
            zx.scalar(value),
            qubit**0,
            qubit**0,
        )


class BitFlipError(Channel):
    """
    Represents a bit-flip error channel.
    """

    def __init__(self, prob):
        # pylint: disable=import-outside-toplevel
        from optyx.core import zx
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
        # pylint: disable=import-outside-toplevel
        from optyx.core import zx
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
    return Diagram.id(n) if \
          isinstance(n, channel.Ty) else Diagram.id(qubit**n)
