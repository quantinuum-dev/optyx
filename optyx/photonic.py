"""
Overview
--------

A collection of operators acting on photonic modes.
This includes: measurements, states, linear optical gates,
dual rail encoded gates, and fusion measurements.

Measurements
------------------------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:


    Discard
    PhotonThresholdMeasurement
    NumberResolvingMeasurement

Linear optical gates
------------------------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:


    Gate
    Phase
    BBS
    TBS
    MZI
    ansatz

Dual rail encoded operators
----------------------------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:


    DualRail
    PhaseShiftDR
    ZMeasurementDR
    XMeasurementDR
    HadamardBS
    FusionTypeI
    FusionTypeII

States
------------------------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:


    Encode
    Create

Other
------------------------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:


    NumOp
    Scalar
    PhotonLoss


Examples of usage
------------------

**Hang-Ou-Mandel effect**

We can use the linear optical generators to build photonic "chips"
to simulate quantum photonics experiments.

The Hong-Ou-Mandel (HOM) effect is a two-photon interference phenomenon:
when two perfectly indistinguishable photons enter
the two input ports of a 50:50 beam-splitter at the same time,
quantum interference forces them to exit together through
the same output port, eliminating coincident detections.
The depth of the resulting “HOM dip” is the usual test of photon
indistinguishability — crucial for reliable entangling operations
and scalable photonic circuits - so any deviation from a perfect
dip directly exposes timing, spectral or polarization mismatches
that would otherwise degrade gate fidelity.

Let's use the beam-splitter to experiment with the Hong-Ou-Mandel effect.
Beam-splitter is just a box in the diagram, which we can draw:

>>> BS = BBS(0)
>>> BS.draw(path='docs/_static/BS.png')

.. image:: /_static/BS.png
    :align: center

Some of the diagrams of the module can be converted to :class:`zw` diagrams.
This includes the beam splitter. The :code:`zw` diagram is used to
construct the tensor network for evaluation.

>>> from discopy.drawing import Equation
>>> BS = BBS(0)
>>> double_BS = BS.get_kraus()
>>> Equation(BS, double_BS, symbol="$\\mapsto$").draw(\\
... path="docs/_static/double_BS.png")

.. image:: /_static/double_BS.png
    :align: center

**Evaluating circuits with tensor networks**

If we want to evaluate the effect of
inputting two photons using :code:`quimb`,
we need to feed the circuit with two photons.
Finally, let's check the effect of having both
photons on two output modes using postselection.

>>> from optyx.classical import Select
>>> diagram_qpath = Create(1, 1) >> BS >> Select(1, 1)
>>> diagram_qpath.draw(path='docs/_static/BS_hom_2.png', figsize=(3, 3))

.. image:: /_static/BS_hom_2.png
    :align: center

>>> float(np.round(diagram_qpath.double().to_tensor().to_quimb()^..., 1))
0.0

It is an impossible event for an ideal beam splitter.

**Evaluating circuits with using permanent-based methods**

We can also evaluate the same experiment using :code:`Perceval`:

>>> diagram_qpath.to_path().prob_with_perceval().array[0, 0]
0j

**Photon loss**

Photon loss is the disappearance of a qubit-carrying photon—through absorption,
scattering, imperfect coupling, or detector inefficiency—before it
can participate in its intended operation. Because each photonic
qubit exists in just one photon (in dual rail encoding),
losing that photon erases the quantum
state entirely, so circuit success probabilities plummet
as systems grow and errors accumulate. As photon-loss is the
leading source of error in current photonic systems, it is
important to be able model it.

With photon loss, HOM effect does not hold anymore.
We can actually observe one photon in one output mode with a non-zero
probability.

>>> diagram_qpath = (
...     Create(1, 1) >>
...     PhotonLoss(0.2) @ Id(1) >>
...     BS >>
...     Select(1, 0)
... )
>>> diagram_qpath.draw(path='docs/_static/BS_loss.png', figsize=(3, 3))

.. image:: /_static/BS_loss.png
    :align: center

>>> float(np.round(diagram_qpath.double().to_tensor().to_quimb()^..., 1))
0.4

**Photon distinguishability**

If two photons carry subtle features which allow for telling them apart —
arriving a few nanoseconds apart, having different energies,
or differing in polarisation — their paths can be told apart.
In photonic quantum computing that is a serious
drawback, because most logic operations rely on two
or more photons behaving as
perfectly identical “bosons” that merge, overlap
and interfere. When the photons
are distinguishable that interference is weakened,
so the gates misfire more often
and the computation accumulates errors long before
it can finish. Keeping photons truly
indistinguishable therefore sits alongside low loss
and high detector efficiency as one
of the core engineering targets for scalable photonic processors.

Again, we can model photon distinguishability in :code:`optyx`.
Let us try the Hong-Ou-Mandel effect with distinguishable photons.

The two internal states are random:

>>> internal_state_1 = np.random.rand(2) + 1j*np.random.rand(2)
>>> internal_state_1 = internal_state_1 / np.linalg.norm(internal_state_1)
>>> internal_state_2 = np.random.rand(2) + 1j*np.random.rand(2)
>>> internal_state_2 = internal_state_2 / np.linalg.norm(internal_state_2)

:code:`Create` accepts internal states of photons as an argument:

>>> channel_HOM = (
...     Create(1, 1, internal_states=(internal_state_1,
...                                 internal_state_2)) >>
...     BS >> NumberResolvingMeasurement(2)
... )

>>> channel_HOM.draw(path='docs/_static/BS_hom_distinguishable.png',
... figsize=(3, 3))

.. image:: /_static/BS_hom_distinguishable.png
    :align: center

Let's evaluate the circuit with distinguishable photons.
:code:`Channel.inflate` is used to indicate
evaluation of the channel taking into account the internal states:

>>> channel_HOM = channel_HOM.inflate(len(internal_state_1))
>>> result = (
...     channel_HOM.double().to_tensor().to_quimb()^...
... ).data

Let's get the probabilities of the outcomes:

>>> rounded_result = np.round(result, 6)
>>> non_zero_dict = {idx: val for idx, val in
...     np.ndenumerate(rounded_result) if val != 0}

The probability of detecting one photon in each output mode is
:math:`\\frac{1}{2} - \\frac{1}{2} |\\langle s_1 | s_2 \\rangle|^2`.

>>> assert np.isclose(
...     non_zero_dict[(1, 1)],
...     0.5 - 0.5*np.abs(np.array(internal_state_1) \\
...         .dot(np.array(internal_state_2).conjugate()))**2, 3
... )

"""


from functools import cached_property
from abc import abstractmethod, ABC
from collections.abc import Iterable
import numpy as np
import sympy as sp
from sympy import Expr, lambdify, Symbol, Mul
from discopy.cat import rsubs

from optyx.core import (
    channel,
    diagram,
    zw
)

from optyx.classical import ClassicalFunction, DiscardMode
from optyx.utils.utils import matrix_to_zw

from optyx import (
    bit,
    mode,
    qmode,
    Measure,
    Discard as DiscardChannel,
    Encode as EncodeChannel,
    Channel,
    Diagram
)


class Scalar(Channel):
    """
    Scalar with a complex value.
    """
    def __init__(self, value):
        if not isinstance(value, (Symbol, Mul)):
            self.scalar = complex(value)
        else:
            self.scalar = value
        super().__init__(
            f"{value}",
            diagram.Scalar(value)
        )
        self.data = value

    def subs(self, *args):
        data = rsubs(self.scalar, *args)
        return Scalar(data)

    # pylint: disable=unused-argument
    def grad(self, var, **params):
        """Gradient with respect to :code:`var`."""
        if var not in self.free_symbols:
            return self.sum_factory((), self.dom, self.cod)
        return Scalar(self.scalar.diff(var))

    def lambdify(self, *symbols, **kwargs):
        return lambda *xs: type(self)(
            lambdify(symbols, self.scalar, **kwargs)(*xs)
        )


class Encode(EncodeChannel):
    """
    Encode :math:`n` modes into :math:`n` qmodes.
    """
    def __init__(self, n):
        super().__init__(mode**n)  # pragma: no cover


class Discard(DiscardChannel):
    """
    Discard :math:`n` qmodes.
    """

    def __init__(self, n):
        super().__init__(qmode**n)  # pragma: no cover


class PhotonThresholdMeasurement(Channel):
    """
    Ideal photon-number non-resolving detector
    from mode to bit from qmode to bit.
    Detects whether one or more photons are present.
    """

    def __init__(self, n=1):
        super().__init__(
            "PhotonThresholdMeasurement",
            diagram.PhotonThresholdDetector()**n,
            cod=bit**n
        )


class NumberResolvingMeasurement(Measure):
    """
    Number-resolving measurement of :math:`n` photons.
    """

    def __init__(self, n):
        super().__init__(qmode**n)  # pragma: no cover


class Create(Channel):
    """
    Fock basis states (occupation numbers).
    """
    def __init__(self, *photons: int,
                 internal_states: tuple[list[int]] = None):
        self.photons = photons
        super().__init__(
            f"Create({photons})",
            zw.Create(*photons, internal_states=internal_states)
        )


class AbstractGate(Channel, ABC):
    """
    Abstract class for linear optical gates.
    """
    def __init__(
        self,
        dom: int,
        cod: int,
        name: str,
        data=None
    ):

        self.dtype = Expr if self._contains_expr(data) else complex
        super().__init__(
            name,
            self._normal_form(dom, cod)
        )
        self.data = data

    def _normal_form(self, dom, cod):
        return matrix_to_zw(self.array.reshape(dom, cod))

    @cached_property
    def array(self):
        """
        Array to be used for building zw diagrams.
        """
        return np.asarray(self._compute_array())

    @abstractmethod
    def _compute_array(self):
        pass  # pragma: no cover

    def _contains_expr(self, obj):
        if isinstance(obj, Expr):
            return True
        if isinstance(obj, Iterable) and not isinstance(obj, (str, bytes)):
            return any(self._contains_expr(item) for item in obj)
        return False


class Gate(AbstractGate):
    """
    Unitary LO gate in a diagram.

    Parameters:
        array : Unitary matrix (not checked on initialisation)
        dom : int
        cod : int
        name : str

    Example
    -------
    >>> hbs_array = (1 / 2) ** (1 / 2) * np.array([[1, 1], [1, -1]])
    >>> HBS = Gate(hbs_array, 2, 2, "HBS")
    >>> assert np.allclose(
    ...     (HBS.dagger() >> HBS).to_path().eval(2).array,
    ...                 diagram.Id(diagram.Mode(2)).to_path().eval(2).array)
    """

    # pylint: disable=too-many-positional-arguments
    def __init__(
        self,
        matrix,
        dom: int,
        cod: int,
        name: str,
        data=None
    ):
        self._matrix = np.asanyarray(matrix)
        super().__init__(dom, cod, name, data=data)

    def _compute_array(self):
        return self._matrix

    def dagger(self):
        return Gate(
            np.conjugate(self.array.T),
            len(self.cod),
            len(self.dom),
            self.name
        )

    def conjugate(self):
        """
        Conjugate defined on the underlying matrix.
        """
        return Gate(
            np.conjugate(self.array),
            len(self.dom),
            len(self.cod),
            self.name
        )  # pragma: no cover


class Phase(AbstractGate):
    """
    Phase shift with angle parameter between 0 and 1

    Parameters:
        angle : Phase parameter between 0 and 1

    Example
    -------
    >>> Phase(1/2).to_path().eval(1).array.round(3)
    array([[-1.+0.j]])
    >>> from sympy.abc import psi
    >>> derivative = Phase(psi).grad(psi).subs((psi,
    ...                     0.5)).to_path().eval(2).array
    >>> assert np.allclose(derivative, 4 * np.pi * 1j)
    """

    def __init__(self, angle: float):
        self.angle = angle
        super().__init__(
            1, 1,
            f"Phase({angle})",
            data=angle
        )

    def _compute_array(self):
        backend = sp if self.dtype is Expr else np
        return [backend.exp(2 * np.pi * 1j * self.angle)]

    def grad(self, var):
        """Gradient with respect to :code:`var`."""
        if var not in self.free_symbols:
            return self.sum_factory((), self.dom, self.cod)
        s = 2j * np.pi * self.angle.diff(var)
        d = Scalar(s) @ (self >> NumOp())
        return d

    def lambdify(self, *symbols, **kwargs):
        return lambda *xs: type(self)(
            lambdify(symbols, self.angle, **kwargs)(*xs)
        )

    def dagger(self):
        return Phase(-self.angle)

    def conjugate(self):
        """
        Conjugate defined on the underlying matrix.
        """
        return Phase(-self.angle)


class NumOp(Channel):
    """
    Number operator.
    """
    def __init__(self):
        super().__init__(
            "NumOp",
            (
                zw.Split(2) >>
                zw.Id(1) @ (zw.Select() >> zw.Create()) >>
                zw.Merge(2)
            )
        )


class BBS(AbstractGate):
    """
    Beam splitter with a bias.

    Corresponds to :py:class:`Matrix`
    :math:`\\begin{pmatrix}
    \\tt{sin}((0.25 + bias)\\pi)
    & i \\tt{cos}((0.25 + bias)\\pi) \\\\
    i \\tt{cos}((0.25 + bias)\\pi)
    & \\tt{sin}((0.25 + bias)\\pi) \\end{pmatrix}`.

    Parameters
    ----------
    bias : float
        Bias from standard 50/50 beam splitter, parameter between 0 and 1.

    Example
    -------
    The standard beam splitter is:

    >>> BS = BBS(0)

    We can check the Hong-Ou-Mandel effect:

    >>> from optyx.classical import Select
    >>> d = Create(1, 1) >> BS
    >>> assert np.isclose((d >> Select(0, 2)).to_path().prob().array,
    ...                                                                0.5)
    >>> assert np.isclose((d >> Select(2, 0)).to_path().prob().array,
    ...                                                                0.5)
    >>> assert np.isclose((d >> Select(1, 1)).to_path().prob().array,
    ...                                                                  0)

    Check the dagger:

    >>> y = BBS(0.4)
    >>> x = y.get_kraus()
    >>> assert np.allclose((
    ...     y >> y.dagger()).to_path().eval(2).array,
    ...             diagram.Id(diagram.Mode(2)).to_path().eval(2).array)
    >>> comp = (x @ x >> diagram.Id(diagram.Mode(1)) @ x @ \\
    ...             diagram.Id(diagram.Mode(1))) >> \\
    ...             (x @ x >> diagram.Id(diagram.Mode(1)) @ x @ \\
    ...             diagram.Id(diagram.Mode(1))).dagger()
    >>> assert np.allclose(comp.to_path().eval(2).array,
    ...           diagram.Id(diagram.Mode(4)).to_path().eval(2).array)

    """

    def __init__(self, bias, is_conj=False):
        self.bias = bias
        self.is_conj = is_conj
        super().__init__(
            2, 2,
            f"BBS({bias})",
            data=bias
        )

    def _compute_array(self):
        backend = sp if self.dtype is Expr else np
        sin = backend.sin((0.25 + self.bias) * np.pi)
        cos = backend.cos((0.25 + self.bias) * np.pi)
        if self.is_conj:
            array = [-1j * cos, sin, sin, -1j * cos]
        else:
            array = [1j * cos, sin, sin, 1j * cos]
        return np.array(array).reshape(2, 2)

    def lambdify(self, *symbols, **kwargs):
        return lambda *xs: type(self)(
            lambdify(symbols, self.bias, **kwargs)(*xs)
        )

    def dagger(self):
        return BBS(0.5 - self.bias)

    def conjugate(self):
        """
        Conjugate defined on the underlying matrix.
        """
        return BBS(self.bias, not self.is_conj)


class TBS(AbstractGate):
    """
    Tunable Beam Splitter.

    Corresponds to :py:class:`Matrix`
    :math:`\\begin{pmatrix}
    \\tt{sin}(\\theta \\, \\pi)
    & \\tt{cos}(\\theta \\, \\pi) \\\\
    \\tt{cos}(\\theta \\, \\pi) & - \\tt{sin}(\\theta \\, \\pi)
    \\end{pmatrix}`.

    Parameters
    ----------
    theta : float
        TBS parameter ranging from 0 to 1.

    Example
    -------
    >>> BS = BBS(0)
    >>> tbs = lambda x: (
    ...       BS >>
    ...       Diagram.id(qmode) @ Phase(x) >>
    ...       BS
    ... )
    >>> assert np.allclose(
    ...     TBS(0.15).to_path().array, tbs(0.15).to_path().array)
    >>> assert np.allclose(
    ...     (TBS(0.25) >> TBS(0.25).dagger()).to_path().array,
    ...     Diagram.id(qmode**2).to_path().array)
    >>> assert (TBS(0.25).dagger().global_phase ==\\
    ...         np.conjugate(TBS(0.25).global_phase))

    """

    def __init__(self,
                 theta,
                 is_gate_dagger=False,
                 is_conj=False):
        self.theta = theta
        self.is_gate_dagger = is_gate_dagger
        self.is_conj = is_conj
        super().__init__(
            2, 2,
            f"TBS({theta})",
            data=theta
        )

    @cached_property
    def global_phase(self):
        """
        Global phase of the TBS.
        """
        backend = sp if self.dtype is Expr else np
        return (
            -1j * backend.exp(-1j * self.theta * backend.pi)
            if self.is_gate_dagger or self.is_conj
            else 1j * backend.exp(1j * self.theta * backend.pi)
        )

    def _compute_array(self):
        backend = sp if self.dtype is Expr else np
        sin = backend.sin(self.theta * backend.pi)
        cos = backend.cos(self.theta * backend.pi)
        array = np.array([sin, cos, cos, -sin]).reshape(2, 2)
        if self.is_gate_dagger:
            array = np.conjugate(array.T)
        if self.is_conj:
            array = np.conjugate(array)
        return array * self.global_phase

    def lambdify(self, *symbols, **kwargs):
        return lambda *xs: type(self)(
            lambdify(symbols, self.theta, **kwargs)(*xs),
            is_gate_dagger=self.is_gate_dagger,
            is_conj=self.is_conj
        )

    def _decomp(self):
        d = BS >> qmode @ Phase(self.theta) >> BS
        return d.dagger() if self.is_gate_dagger else d

    def grad(self, var):
        """Gradient with respect to :code:`var`."""
        if var not in self.free_symbols:
            return self.sum_factory((), self.dom, self.cod)
        return self._decomp().grad(var)

    def conjugate(self):
        """
        Conjugate defined on the underlying matrix.
        """
        return TBS(self.theta, self.is_gate_dagger, not self.is_conj)

    def dagger(self):
        return TBS(self.theta, is_gate_dagger=not self.is_gate_dagger,
                   is_conj=self.is_conj)


class MZI(AbstractGate):
    """
    Mach-Zender interferometer.

    Corresponds to :py:class:`Matrix`
    :math:`\\begin{pmatrix}
    e^{2\\pi i \\phi} \\tt{sin}(\\theta \\, \\pi)
    & \\tt{cos}(\\theta \\, \\pi) \\\\
    e^{2\\pi i \\phi} \\tt{cos}(\\theta \\, \\pi)
    & - \\tt{sin}(\\theta \\, \\pi) \\end{pmatrix}`.

    Parameters
    ----------
    theta: float
        Internal phase parameter, ranging from 0 to 1.
    phi: float
        External phase parameter, ranging from 0 to 1.

    Example
    -------
    >>> assert np.allclose(
    ...     MZI(0.28, 0).to_path().array,
    ...     TBS(0.28).to_path().array)
    >>> assert np.isclose(
    ...    MZI(0.28, 0.3).global_phase,
    ...    TBS(0.28).global_phase)
    >>> assert np.isclose(
    ...     MZI(0.12, 0.3).global_phase.conjugate(),
    ...     MZI(0.12, 0.3).dagger().global_phase)
    >>> mach = lambda x, y: TBS(x) >> Phase(y) @ \\
    ...          Diagram.id(qmode)
    >>> assert np.allclose(
    ...     MZI(0.28, 0.9).to_path().array,
    ...     mach(0.28, 0.9).to_path().array)
    >>> assert np.allclose(
    ...     (MZI(0.28, 0.34) >> MZI(0.28, 0.34).dagger()).to_path().array,
    ...     Diagram.id(qmode**2).to_path().array)

    """

    def __init__(self,
                 theta,
                 phi,
                 is_gate_dagger=False,
                 is_conj=False):
        self.theta, self.phi = theta, phi
        self.is_gate_dagger = is_gate_dagger
        self.is_conj = is_conj
        super().__init__(
            2, 2,
            f"MZI({theta}, {phi})",
            data=(theta, phi)
        )

    @cached_property
    def global_phase(self):
        """
        Global phase of the MZI.
        """
        backend = sp if self.dtype is Expr else np
        return (
            -1j * backend.exp(-1j * self.theta * backend.pi)
            if self.is_gate_dagger or self.is_conj
            else 1j * backend.exp(1j * self.theta * backend.pi)
        )

    def _compute_array(self):
        backend = sp if self.dtype is Expr else np
        cos = backend.cos(backend.pi * self.theta)
        sin = backend.sin(backend.pi * self.theta)
        exp = backend.exp(1j * 2 * backend.pi * self.phi)
        array = np.array([exp * sin, cos, exp * cos, -sin]).reshape(2, 2)
        if self.is_gate_dagger:
            array = np.conjugate(array.T)
        if self.is_conj:
            array = np.conjugate(array)
        return array * self.global_phase

    def lambdify(self, *symbols, **kwargs):
        return lambda *xs: type(self)(
            *lambdify(symbols, [self.theta, self.phi], **kwargs)(*xs),
            is_gate_dagger=self.is_gate_dagger,
            is_conj=self.is_conj
        )

    def _decomp(self):
        x, y = self.theta, self.phi
        d = BS >> qmode @ Phase(x) >> BS >> Phase(y) @ qmode
        return d.dagger() if self.is_gate_dagger else d

    def grad(self, var):
        """Gradient with respect to :code:`var`."""
        if var not in self.free_symbols:
            return self.sum_factory((), self.dom, self.cod)
        return self._decomp().grad(var)

    def dagger(self):
        return MZI(self.theta, self.phi,
                   is_gate_dagger=not self.is_gate_dagger,
                   is_conj=self.is_conj)

    def conjugate(self):
        """
        Conjugate defined on the underlying matrix.
        """
        return MZI(self.theta, self.phi, self.is_gate_dagger, not self.is_conj)


def ansatz(width, depth):
    """
    Returns a universal interferometer given width, depth and parameters x,
    based on https://arxiv.org/abs/1603.08788.

    Parameters
    ----------
    width: int
        Number of modes in the ansatz.
    depth: int
        Number of layers in the ansatz.

    Example
    -------
    >>> ansatz(6, 4).draw(path='docs/_static/ansatz6_4.png')
    >>> ansatz(5, 4).draw(path='docs/_static/ansatz5_4.png')

    .. image:: /_static/ansatz6_4.png
        :align: center

    .. image:: /_static/ansatz5_4.png
        :align: center
    """

    def p(i, j):
        return sp.Symbol(f"a_{i}_{j}"), sp.Symbol(f"b_{i}_{j}")

    d = Diagram.id(qmode**width)
    for i in range(depth):
        n_mzi = (width - 1) // 2 if i % 2 else width // 2
        left = qmode**(i % 2)
        right = qmode**(width - (i % 2) - 2 * n_mzi)
        wire = Diagram.id(qmode**0)
        for j in range(n_mzi):
            wire @= MZI(*p(i, j))
        d >>= left @ wire @ right

    return d


class HadamardBS(Gate):
    """
    An alternative version of the beam splitter
    which implements a Hadamard gate in dual rail
    encoding.
    """
    def __init__(self):
        matrix = np.sqrt(1 / 2) * np.array([[1, 1], [1, -1]])
        super().__init__(
            matrix, 2, 2, "HadamardBS"
        )


class DualRail(Channel):
    """
    Represents a dual-rail quantum channel
    encoding a specified number of qubit registers.
    """
    def __init__(self, n_qubits, internal_states=None):
        super().__init__(
            f"DualRail({n_qubits})",
            diagram.dual_rail(n_qubits, internal_states=internal_states)
        )


class PhaseShiftDR(Channel):
    """
    Represents a phase shift operation in dual-rail encoding.
    """

    def __init__(self, phase):
        super().__init__(
            f"PhaseShift({phase})",
            diagram.Mode(1) @ Phase(phase).get_kraus()
        )


class ZMeasurementDR(Diagram):
    """
    ZMeasurement circuit that performs a measurement in the Z basis
    after applying a phase shift of alpha.
    """
    def __new__(cls, alpha):
        return (
            qmode @ Phase(alpha) >>
            HadamardBS() >>
            NumberResolvingMeasurement(2) >>
            DiscardMode(1) @ mode
        )


class XMeasurementDR(Diagram):
    """
    XMeasurement circuit that performs a measurement in the X basis
    after applying a Hadamard beam splitter.
    """
    def __new__(cls, alpha):
        return (
            HadamardBS() >>
            ZMeasurementDR(alpha)
        )


class FusionTypeI(Diagram):
    r"""
    Type-I fusion measurement on two dual-rail photonic qubits.

    This probabilistic operation interferes one rail of
    each qubit on a 50/50 beam-splitter, performs
    number-resolving detection on the ancillary modes, and—conditional
    on the outcome—fuses the qubits into a *single* dual-rail qubit.

    **Domain**
        ``qmode ** 4``
        (four photonic modes encoding two qubits).

    **Codomain**
        ``qmode ** 2 @ bit ** 2``
        - the surviving dual-rail qubit followed by two classical bits
        ``[s, k]`` where

        * ``s`` is the parity (success) bit
        * ``k`` is the Pauli-correction bit for feed-forward.

    Notes
    -----
    * Succeeds with probability 0.5.
    * When ``s = 1`` the fusion succeeds; the required Pauli-Z
      correction on the output qubit is ``Z^k``.

    Examples
    --------
    >>> from optyx.photonic import Create, FusionTypeI
    >>> circuit = Create(1, 0, 1, 0) >> FusionTypeI()
    >>> circuit.draw(path="docs/_static/fusioni.svg")

    .. image:: /_static/fusioni.svg
        :align: center
    """
    def __new__(cls):
        # pylint: disable=invalid-name
        kraus_map_fusion_I = (
            diagram.Mode(1) @ diagram.Swap(
                diagram.Mode(1),
                diagram.Mode(1)
                ) @ diagram.Mode(1) >>
            diagram.Mode(1) @ HadamardBS().get_kraus() @ diagram.Mode(1) >>
            diagram.Mode(2) @ diagram.Swap(
                diagram.Mode(1),
                diagram.Mode(1)
                ) >>
            diagram.Mode(1) @ diagram.Swap(
                diagram.Mode(1),
                diagram.Mode(1)
                ) @ diagram.Mode(1)
        )

        fusion_I = Channel(
            "Fusion I", kraus_map_fusion_I
        )

        def fusion_I_function(x):
            """
            A classical function that returns two bits based on an input x,
            based on the classical logical for the Fusion type I circuit.
            """
            a = x[0]
            b = x[1]
            s = (a % 2) ^ (b % 2)
            k = int(s*b + (1-s)*(1 - (a + b)/2)) % 2
            return [s, k]

        classical_function_I = ClassicalFunction(
            fusion_I_function,
            mode**2,
            bit**2
        )

        return (
            fusion_I >>
            qmode**2 @ NumberResolvingMeasurement(2) >>
            qmode**2 @ classical_function_I
        )


class Swap(channel.Swap):
    """
    Swap channel for qmodes.
    """
    def __init__(self, left, right):
        super().__init__(qmode**left, qmode**right)


class PhotonLoss(Channel):
    """
    Photon loss channel that models the loss of a photon
    with a given survival probability.

    Examples
    -------
    >>> loss_single = PhotonLoss(0.25)
    >>> loss_double = PhotonLoss(0.5) >> PhotonLoss(0.5)
    >>> assert np.allclose(
    ...     (loss_single.double().to_tensor().to_quimb()^...).data,
    ...     (loss_double.double().to_tensor().to_quimb()^...).data
    ... )

    Survival probability of 0.0 means the photon is lost with certainty:

    >>> loss = Create(1) >> PhotonLoss(0.0)
    >>> zero_state = Create(0)
    >>> assert np.allclose(
    ...     (loss.double().to_tensor().to_quimb()^...).data,
    ...     (zero_state.double().to_tensor().to_quimb()^...).data
    ... )

    Survival probability of 1.0 means the photon is never lost:

    >>> loss = Create(1) >> PhotonLoss(1.0)
    >>> one_state = Create(1)
    >>> assert np.allclose(
    ...     (loss.double().to_tensor().to_quimb()^...).data,
    ...     (one_state.double().to_tensor().to_quimb()^...).data
    ... )
    """

    def __init__(self, p_survive):
        self.p_survive = p_survive
        kraus = (
            zw.W(2) >>
            zw.Endo(p_survive**0.5) @ zw.Endo((1-p_survive)**0.5)
        )
        super().__init__(
            f"Loss({p_survive})",
            kraus,
            qmode,
            qmode @ channel.Ty(),
            env=qmode.single()
        )


class FusionTypeII(Diagram):
    r"""
    Type-II fusion measurement for dual-rail photonic qubits.

    A scheme that **consumes both
    qubits**.  After a network of four 50/50 beam-splitters and mode
    swaps, all four output modes are measured with
    number-resolving detectors.  No photonic modes remain; the
    classical outcome determines whether an entanglement link has been
    created between the neighbouring cluster-state nodes.

    **Domain**
        ``qmode ** 4``
        (two dual-rail qubits).

    **Codomain**
        ``bit ** 2``
        containing

        * ``s`` - success / parity bit
        * ``k`` - Pauli-correction bit (applied to neighbouring nodes).

    Notes
    -----
    * Success probability is 0.5.
    * On success (``s = 1``) the measurement produces a Bell-type
      entanglement; on failure the qubits are lost.

    Examples
    --------
    >>> from optyx.photonic import Create, FusionTypeII
    >>> circuit = Create(1, 0, 1, 0) >> FusionTypeII()
    >>> circuit.draw(path="docs/_static/fusionii.svg")

    .. image:: /_static/fusionii.svg
        :align: center
    """
    def __new__(cls):
        # pylint: disable=invalid-name
        fusion_II = Channel(
            "Fusion II",
            (
                HadamardBS().get_kraus() @ HadamardBS().get_kraus() >>
                diagram.Mode(1) @ diagram.Swap(
                    diagram.Mode(1),
                    diagram.Mode(1)
                    ) @ diagram.Mode(1) >>
                diagram.Mode(1) @ HadamardBS().get_kraus() @ diagram.Mode(1) >>
                diagram.Mode(2) @ diagram.Swap(
                    diagram.Mode(1),
                    diagram.Mode(1)
                    ) >>
                diagram.Mode(1) @ diagram.Swap(
                    diagram.Mode(1),
                    diagram.Mode(1)
                    ) @ diagram.Mode(1) >>
                HadamardBS().get_kraus() @ diagram.Mode(2)
            )
        )

        def fusion_II_function(x):
            """
            A classical function that returns two bits based on an input x,
            based on the classical logical for the Fusion type II circuit.
            """
            a = x[0]
            b = x[1]
            d = x[3]
            s = (a % 2) ^ (b % 2)
            k = int(s*(b + d) + (1-s)*(1 - (a + b)/2)) % 2
            return [s, k]

        classical_function_II = ClassicalFunction(
            fusion_II_function,
            mode**4,
            bit**2
        )

        return (
            fusion_II >>
            NumberResolvingMeasurement(4) >>
            classical_function_II
        )


BS = BBS(0)


def Id(n):
    return Diagram.id(n) if \
          isinstance(n, channel.Ty) else Diagram.id(qmode**n)
