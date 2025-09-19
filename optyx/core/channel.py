"""

Overview
--------

Implements classical-quantum channels.

Quantum channels are completely positive maps acting on
the doubled space :code:`H @ H` for a Hilbert space :code:`H`.
These can be initialised from the Kraus decomposition,
given as an :code:`diagram.Diagram` with domain :code:`H` and
codomain :code:`H @ E` for an auxiliary space :code:`E`,
called the environment, which is not observed.

Channels can moreover have a classical interface,
in the form of input :code:`bit` or :code:`mode` types.
The Kraus map is then given by an :class:`diagram.Diagram`
with domain :code:`H @ C` and codomain :code:`H @ C @ E`,
where the classical type :code:`C` represents
the classical inputs or outputs of the computation.
In the doubled picture, encoding or measuring a classical type
is implemented through instances of :class:`diagram.Spider`.

This module allows to build an arbitrary syntactic :class:`Diagram`
from instances of :class:`Channel`.
The :code:`Diagram.double` method returns an :class:`diagram.Diagram`,
whose tensor evaluation gives all the relevant statistics of the circuit.

Types
-----

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Ob
    Ty

Generators and diagrams
------------------------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Diagram
    Channel
    Measure
    Encode
    Discard


Examples
--------

A Channel is initialised by its Kraus map from `dom` to `cod @ env`.

>>> from optyx.core import zx, zw, diagram
>>> from optyx import photonic
>>> circ = (
...     photonic.Phase(0.25) @
...     photonic.BS @
...     photonic.Phase(0.56) >>
...     photonic.BS @ photonic.BS
... ).get_kraus()
>>> channel = Channel(name='circuit', kraus=circ,\\
...                   dom=qmode ** 4, cod=qmode ** 4, env=diagram.Ty())

We can check that this channel is causal:

>>> import numpy as np
>>> discards = Discard(qmode ** 4)
>>> rhs = (channel >> discards).double().to_tensor().eval().array
>>> lhs = (discards).double().to_tensor().eval().array
>>> assert np.allclose(lhs, rhs)

We can calculate the probability of an input-output pair:

>>> state = Channel('state', zw.Create(1, 0, 1, 0))
>>> effect = Channel('effect', zw.Select(1, 0, 1, 0))
>>> prob = (state >> channel >> effect).double(\\
...     ).to_tensor().eval().array
>>> amp = (zw.Create(1, 0, 1, 0) >> circ >> zw.Select(1, 0, 1, 0)\\
...     ).to_tensor().eval().array
>>> assert np.allclose(prob, np.absolute(amp) ** 2)

We can check that the probabilities of a normalised state sum to 1:

>>> bell_state = Channel('Bell', diagram.Scalar(1/np.sqrt(2)) @ zx.Z(0, 2))
>>> dual_rail = Channel('2R', diagram.dual_rail(2))
>>> measure = Discard(qmode ** 3) @ Measure(qmode)
>>> setup = bell_state >> dual_rail >> channel >> measure
>>> assert np.isclose(sum(setup.double().to_tensor().eval().array), 1)

We can construct a lossy optical channel and compute its probabilities:

>>> eff = 0.95
>>> kraus = zw.W(2) >> zw.Endo(np.sqrt(eff)) @ zw.Endo(np.sqrt(1 - eff))
>>> loss = Channel(str(eff), kraus, dom=qmode, cod=qmode, env=diagram.mode)
>>> uniform_loss = loss.tensor(*[loss for _ in range(3)])
>>> lossy_channel = channel >> uniform_loss
>>> lossy_prob = (state >> lossy_channel >> effect).double(\\
...     ).to_tensor().eval().array
>>> assert np.allclose(lossy_prob, prob * (eff ** 2))

**Diagrams from Bosonic Operators**

The :code:`from_bosonic_operator` method
supports creating :class:`path` diagrams:

>>> from optyx.core.zw import Split, Select, Id
>>> from optyx.core.diagram import Mode
>>> from optyx.photonic import Scalar
>>> d1 = Diagram.from_bosonic_operator(
...     n_modes=2,
...     operators=((0, False), (1, False), (0, True)),
...     scalar=2.1
... )

>>> annil = Channel(
...     "annil", Split(2) >> Select(1) @ Id(Mode(1))
... )
>>> create = annil.dagger()

>>> d2 = Scalar(2.1) @ annil @ qmode >> \\
... qmode @ annil >> create @ qmode

>>> assert d1 == d2

We can map ZX diagrams to :class:`path` diagrams using
dual-rail encoding. For example, we can create a GHZ state:

>>> from discopy.drawing import Equation
>>> from optyx.qubits import Z
>>> from optyx.photonic import DualRail
>>> ghz = Z(0, 3)
>>> ghz_path = ghz.to_dual_rail()
>>> Equation(ghz >> DualRail(3), ghz_path, \\
... symbol="$\\mapsto$").draw(figsize=(10, 10), \\
... path="docs/_static/ghz_dr.svg")

.. image:: /_static/ghz_dr.svg
    :align: center

"""

from __future__ import annotations

from discopy import tensor
from discopy import symmetric, frobenius, hypergraph
from discopy.cat import factory
from pytket.extensions.pyzx import pyzx_to_tk
from pyzx import extract_circuit
from optyx.core import zx, diagram
from optyx.utils.utils import explode_channel
from optyx.core.path import Matrix


class Ob(frobenius.Ob):
    """Basic object: bit, mode, qubit or qmode"""

    _classical = {
        "bit": "bit",
        "mode": "mode",
        "qubit": "bit",
        "qmode": "mode",
    }
    _quantum = {
        "bit": "qubit",
        "mode": "qmode",
        "qubit": "qubit",
        "qmode": "qmode",
    }

    @property
    def is_classical(self):
        """Classical objects are :code:`bit` and :code:`mode`."""
        return self.name not in ["qubit", "qmode"]

    @property
    def single(self):
        """Maps :code:`qubit` to :code:`diagram.bit`
        and :code:`qmode` to :code:`diagram.mode`."""
        return diagram.Ty(self._classical[self.name])

    @property
    def double(self):
        """Maps :code:`qubit` to :code:`diagram.bit @ diagram.bit`
        and :code:`qmode` to :code:`diagram.mode @ diagram.mode`."""
        if self.is_classical:
            return diagram.Ty(self.name)
        name = self._classical[self.name]
        return diagram.Ty(name, name)


@factory
class Ty(frobenius.Ty):
    """Classical and quantum types."""

    ob_factory = Ob

    def single(self):
        """Returns the diagram.Ty obtained by mapping
        :code:`qubit` to :code:`bit` and :code:`qmode` to :code:`mode`"""
        return diagram.Ty().tensor(*[ob.single for ob in self.inside])

    def double(self):
        """Returns the diagram.Ty obtained by mapping
        :code:`qubit` to :code:`bit @ bit`
        and :code:`qmode` to :code:`mode @ mode`"""
        return diagram.Ty().tensor(*[ob.double for ob in self.inside])

    @staticmethod
    # pylint: disable=invalid-name
    def from_optyx(ty):
        """
        Get quantum types from core/diagram.Ty.
        """
        assert isinstance(ty, diagram.Ty)
        # pylint: disable=protected-access
        return Ty(*[Ob._quantum[ob.name] for ob in ty.inside])

    def needs_inflation(self) -> bool:
        """
        Diagrams with at least one :code:`qmode` need inflation.
        """
        return "qmode" in self.name

    # pylint: disable=invalid-name
    def inflate(self, d) -> Ty:
        """
        Inflate the type.
        """
        return (mode**0).tensor(
                *(o**d if o.needs_inflation() else o for o in self)
        )


bit = Ty("bit")
mode = Ty("mode")
qubit = Ty("qubit")
qmode = Ty("qmode")


@factory
class Diagram(frobenius.Diagram):
    """Classical-quantum circuits over qubits and optical modes"""

    ty_factory = Ty
    grad = tensor.Diagram.grad

    def needs_inflation(self) -> bool:
        """
        If the domain or codomain need inflation,
        the diagram needs inflation.
        """
        return self.dom.needs_inflation() or self.cod.needs_inflation()

    # pylint: disable=invalid-name
    def inflate(self, d):
        r"""Translates from an indistinguishable setting
        to a distinguishable one. For a map on :math:`F(\mathbb{C})`,
        obtain a map on :math:`F(\mathbb{C})^{\widetilde{\otimes} d}`."""
        assert isinstance(d, int), "Dimension must be an integer"
        assert d > 0, "Dimension must be positive"

        dom = frobenius.Category(Ty, Diagram)
        cod = frobenius.Category(Ty, Diagram)

        return frobenius.Functor(
            lambda x: x.inflate(d),
            lambda f: f.inflate(d),
            dom,
            cod
        )(self)

    def double(self):
        """Returns the diagram.Diagram obtained by
        doubling every quantum dimension
        and building the completely positive map."""
        dom = frobenius.Category(Ty, Diagram)
        cod = frobenius.Category(diagram.Ty, diagram.Diagram)
        return frobenius.Functor(
            lambda x: x.double(), lambda f: f.double(), dom, cod
        )(self)

    @property
    def is_pure(self):
        """
        Check if the diagram is pure, i.e. it does not
        contain any discards or measures acting on quantum types,
        and does not prepare quantum types from classical types.
        """
        are_layers_pure = []
        are_layers_classical = []
        for layer in self:
            generator = layer.inside[0][1]

            # if we have a discard/measure
            # acting on quantum types, it's not pure
            if (
                isinstance(generator, (Discard, Measure)) and
                any(not ty.is_classical for ty in generator.dom.inside)
            ):
                return False
            if hasattr(generator, 'env') and generator.env != diagram.Ty():
                return False

            # if we prepare quantum from classical types, it's not pure
            if (
                isinstance(generator, Encode) and
                any(ty.is_classical for ty in generator.cod.inside)
            ):
                return False

            # if we're mixing classical and quantum types, it's not pure
            are_layers_pure.append(
                any(ty.is_classical for ty in generator.cod.inside) or
                any(ty.is_classical for ty in generator.dom.inside) or
                isinstance(generator, Discard)
            )

            # assume all classical maps are pure
            are_layers_classical.append(
                all(ty.is_classical for ty in generator.cod.inside) and
                all(ty.is_classical for ty in generator.dom.inside)
            )

        return not any(are_layers_pure) or all(are_layers_classical)

    def get_kraus(self):
        """
        Obtain the Kraus map of a pure circuit.
        """
        assert self.is_pure, "Cannot get a Kraus map of non-pure circuit"
        kraus_maps = [diagram.Id(self.dom.single())]
        for layer in self:
            left = diagram.Ty().tensor(*[ty.single()
                                       for ty in layer.inside[0][0]])
            right = diagram.Ty().tensor(*[ty.single()
                                        for ty in layer.inside[0][2]])
            generator = layer.inside[0][1]

            if isinstance(generator, Swap):
                kraus_maps.append(
                    left @ diagram.Swap(generator.dom.single()[0],
                                        generator.cod.single()[1]) @ right
                )
            else:
                kraus_maps.append(
                    left @ generator.kraus @ right
                )

        if len(kraus_maps) == 1:
            return kraus_maps[0]
        return kraus_maps[0].then(
            *kraus_maps[1:]
        )

    def to_path(self, dtype: type = complex):
        """Returns the :class:`Matrix` normal form
        of a :class:`Diagram`.
        In other words, it is the underlying matrix
        representation of a :class:`path` and :class:`photonic` diagrams."""
        # pylint: disable=import-outside-toplevel
        from optyx.core import path

        assert self.is_pure, "Diagram must be pure to convert to path."

        return frobenius.Functor(
            ob=len,
            ar=lambda f: f.get_kraus().to_path(dtype),
            cod=frobenius.Category(int, path.Matrix[dtype]),
        )(self)

    def decomp(self):
        # pylint: disable=protected-access
        return frobenius.Functor(
            ob=lambda x: qubit**len(x),
            ar=lambda arr: arr._decomp(),
            cod=frobenius.Category(Ty, Diagram),
        )(self)

    def to_dual_rail(self):
        """Convert to dual-rail encoding."""

        assert self.is_pure, "Diagram must be pure to convert to dual rail."

        return frobenius.Functor(
            ob=lambda x: qmode**(2*len(x)),
            ar=lambda arr: arr._to_dual_rail(),
            cod=frobenius.Category(Ty, Diagram),
        )(self.decomp())

    def to_tket(self):
        """
        Convert to tket circuit. The circuit must be a pure circuit.
        """

        assert self.is_pure, "Diagram must be pure to convert to tket."

        kraus_maps = []
        for layer in self:
            left = layer.inside[0][0]
            right = layer.inside[0][2]
            generator = layer.inside[0][1]

            kraus_maps.append(
                diagram.Bit(len(left)) @
                generator.kraus @
                diagram.Bit(len(right))
            )

        # pylint: disable=no-value-for-parameter
        return pyzx_to_tk(
            extract_circuit(
                diagram.Diagram.then(
                    *kraus_maps
                ).to_pyzx()
            ).to_basic_gates()
        )

    def to_pyzx(self):
        """Convert to PyZX circuit. The circuit must be a pure circuit."""
        assert self.is_pure, "Diagram must be pure for conversion."

        return self.get_kraus().to_pyzx()

    @classmethod
    def from_tket(cls, tket_circuit):
        """Convert from tket circuit."""
        # pylint: disable=import-outside-toplevel
        from optyx.qubits import Circuit
        return Circuit(tket_circuit)

    @classmethod
    def from_pyzx(cls, pyzx_circuit):
        """Convert from PyZX circuit."""
        # pylint: disable=import-outside-toplevel
        from optyx.qubits import Circuit
        return Circuit(pyzx_circuit)

    @classmethod
    def from_discopy(cls, discopy_circuit):
        """Convert from discopy circuit."""
        # pylint: disable=import-outside-toplevel
        from optyx.qubits import Circuit
        return Circuit(discopy_circuit)

    # @classmethod
    # def from_bosonic_operator(cls, n_modes, operators, scalar=1):
    #     return Channel(
    #         "Bosonic operator",
    #         diagram.Diagram.from_bosonic_operator(
    #             n_modes, operators, scalar=scalar
    #         )
    #     )

    @classmethod
    def from_bosonic_operator(cls, n_modes, operators, scalar=1):
        """Create a :class:`zw` diagram from a bosonic operator."""
        # pylint: disable=import-outside-toplevel
        from optyx.core import zw
        from optyx.photonic import Scalar

        # pylint: disable=invalid-name
        d = Diagram.id(qmode**n_modes)
        annil = Channel("annil", zw.Split(2) >> zw.Select(1) @ zw.Id(1))
        create = annil.dagger()
        for idx, dagger in operators:
            if not 0 <= idx < n_modes:
                raise ValueError(f"Index {idx} out of bounds.")
            box = create if dagger else annil
            d = d >> qmode**idx @ box @ qmode**(n_modes - idx - 1)

        if scalar != 1:
            # pylint: disable=invalid-name
            d = Scalar(scalar) @ d
        return d

    @classmethod
    def from_graphix(cls, measurement_pattern):
        """Convert from Graphix measurement pattern."""
        # pylint: disable=import-outside-toplevel
        from optyx.qubits import Circuit
        return Circuit(measurement_pattern)

    @classmethod
    def from_perceval(cls, p):
        """
        Convert pcvl.Circuit or pcvl.Processor
        into optyx diagrams.

        Cannot convert objects involving components
        acting on polarisation modes, time delays,
        and with symbols.
        """
        # pylint: disable=import-outside-toplevel
        from optyx import photonic
        from optyx.utils import perceval_conversion
        import perceval as pcvl

        if isinstance(p, pcvl.Circuit):
            p_ = pcvl.Processor("SLOS", p.m)
            p_.add(0, p)

            p_new = pcvl.Processor("SLOS", p.m)
            for c in p_.flatten():
                p_new.add(c[0][0], c[1])
            p = p_new

        n_modes = p.circuit_size
        circuit = photonic.Id(n_modes)
        heralds = p.heralds

        circuit = perceval_conversion.heralds_diagram(
            heralds, n_modes, circuit, "in"
        ) >> circuit

        for wires, component in p.components:
            left = circuit.cod[:min(wires)]
            right = circuit.cod[max(wires) + 1:]

            if isinstance(component, pcvl.Detector):
                box = perceval_conversion.detector(component, wires)
            elif isinstance(
                component,
                pcvl.components.feed_forward_configurator.FFCircuitProvider
            ):
                box, left, right = perceval_conversion.ff_circuit_provider(
                    component, wires, circuit
                )
            elif isinstance(
                component,
                pcvl.components.feed_forward_configurator.FFConfigurator
            ):
                box, left, right = perceval_conversion.ff_configurator(
                    component, wires, circuit
                )
            elif isinstance(component, pcvl.components.Barrier):
                continue
            elif hasattr(component, "U"):
                box = perceval_conversion.unitary(component, wires)
            else:
                raise ValueError(
                    f"Unsupported perceval component type: {type(component)}"
                )

            circuit >>= (left @ box @ right)

        circuit >>= perceval_conversion.heralds_diagram(
            heralds, n_modes, circuit, "out"
        )
        if p.post_select_fn is not None:
            circuit >>= perceval_conversion.postselection(circuit, p)
        return circuit

    # pylint: disable=invalid-name
    def __pow__(self, n):
        if n == 1:
            return self
        return self @ self ** (n - 1)

    def eval(self, backend=None, **kwargs):
        """
        Evaluate the diagram using the specified backend.
        If no backend is specified, it uses the QuimbBackend.
        """
        # pylint: disable=import-outside-toplevel
        from optyx.core.backends import QuimbBackend
        if backend is None:
            backend = QuimbBackend()

        return backend.eval(self, **kwargs)


class Channel(Diagram, frobenius.Box):
    """
    Channel initialised by its Kraus map.
    """

    def __init__(
            self,
            name,
            kraus,
            dom=None,
            cod=None,
            env=diagram.Ty()):
        assert isinstance(kraus, diagram.Diagram)
        if dom is None:
            dom = Ty.from_optyx(kraus.dom)
        if cod is None:
            cod = Ty.from_optyx(kraus.cod)
        assert kraus.dom == dom.single()
        assert kraus.cod == cod.single() @ env
        self.kraus = kraus
        self.env = env
        super().__init__(name, dom, cod)

    def double(self):
        """
        Returns the :class:`diagram.Diagram` representing
        the action of the channel as a CP map on the doubled space.
        """

        def get_spiders(dom):
            spiders = diagram.Id()
            # pylint: disable=invalid-name
            for ob in dom.inside:
                if ob.is_classical:
                    box = diagram.Spider(1, 2, ob.single)
                else:
                    box = diagram.Id(ob.double)
                spiders @= box
            return spiders

        # pylint: disable=invalid-name
        def get_perm(n):
            return sorted(sorted(list(range(n))), key=lambda i: i % 2)

        cod = self.cod.single()
        top_spiders = get_spiders(self.dom)
        top_perm = diagram.Diagram.permutation(
            get_perm(len(top_spiders.cod)), top_spiders.cod
        )
        swap_env = diagram.Id(cod @ self.env) @ diagram.Diagram.swap(
            cod, self.env
        )
        discard = (
            diagram.Id(cod)
            @ diagram.Diagram.spiders(2, 0, self.env)
            @ diagram.Id(cod)
        )
        new_cod = diagram.Ty().tensor(*[ty @ ty for ty in cod])
        bot_perm = diagram.Diagram.permutation(
            get_perm(2 * len(cod)), new_cod
        ).dagger()
        bot_spiders = get_spiders(self.cod).dagger()
        top = top_spiders >> top_perm
        bot = swap_env >> discard >> bot_perm >> bot_spiders
        return top >> self.kraus @ self.kraus.conjugate() >> bot

    def dagger(self):
        return Channel(
            name=self.name + ".dagger()",
            kraus=self.kraus.dagger(),
            dom=self.cod,
            cod=self.dom,
        )

    def _decomp(self):
        # pylint: disable=import-outside-toplevel
        raise NotImplementedError(
            "Decomposition is only implemented for ZX channels."
        )

    def _to_dual_rail(self):
        raise TypeError(
            "Only ZX channels can be converted to dual rail."
            )

    def lambdify(self, *symbols, **kwargs):
        # Non-symbolic gates can be returned directly
        return lambda *xs: self

    def subs(self, *args) -> Diagram:
        syms, exprs = zip(*args)
        return self.lambdify(*syms)(*exprs)

    def inflate(self, d):
        r"""Translates from an indistinguishable setting
        to a distinguishable one. For a map on :math:`F(\mathbb{C})`,
        obtain a map on :math:`F(\mathbb{C})^{\widetilde{\otimes} d}`."""

        return Channel(
            name=self.name + f"^{d}",
            kraus=self.kraus.inflate(d) if
            self.needs_inflation() else self.kraus,
            dom=self.dom.inflate(d),
            cod=self.cod.inflate(d),
        )


class Spider(frobenius.Spider, Channel):  # pragma: no cover
    """
    Spider as a channel.
    """
    def __init__(self, n_legs_in: int, n_legs_out: int, typ: Ty, data=None,
                 **params):
        super().__init__(
            n_legs_in, n_legs_out, typ, data=data, **params
        )
        self.kraus = diagram.Spider(
            n_legs_in, n_legs_out, typ.single()
        )
        self.env = diagram.Ty()


class Sum(symmetric.Sum, Diagram):
    """
    Formal sum of optyx channel diagrams
    """

    __ambiguous_inheritance__ = (symmetric.Sum,)

    def double(self):
        return diagram.Diagram.sum_factory([t.double() for t in self])

    def grad(self, var, **params):
        """Gradient with respect to :code:`var`."""
        if var not in self.free_symbols:
            return self.sum_factory((), self.dom, self.cod)
        return sum(term.grad(var, **params) for term in self.terms)

    def get_kraus(self):
        if len(self.terms) == 0:
            return diagram.Scalar(0)

        return diagram.Diagram.sum_factory(
            [term.get_kraus() for term in self.terms]
        )


class CQMap(Diagram, frobenius.Box):
    """
    Channel initialised by its Density matrix.
    """

    def __init__(self, name, density_matrix, dom, cod):
        assert isinstance(density_matrix, diagram.Diagram)
        assert density_matrix.dom == dom.double()
        assert density_matrix.cod == cod.double()

        self.density_matrix = density_matrix
        super().__init__(name, dom, cod)

    def double(self):
        return self.density_matrix

    def dagger(self):
        return CQMap(
            name=self.name + ".dagger()",
            density_matrix=self.density_matrix.dagger(),
            dom=self.cod,
            cod=self.dom,
        )

    def inflate(self, d):
        r"""
        Translates from an indistinguishable setting
        to a distinguishable one. For a map on
        :math:`F(\mathbb{C}^d)`,
        obtain a map on :math:`F(\mathbb{C})^{\widetilde{\otimes} d}`.
        """

        return CQMap(
            name=self.name + f"^{d}",
            density_matrix=self.density_matrix.inflate(d) if
            self.needs_inflation() else self.density_matrix,
            dom=self.dom.inflate(d),
            cod=self.cod.inflate(d)
        )

    # pylint: disable=invalid-name
    def __pow__(self, n):
        if n == 1:
            return self
        return self @ self ** (n - 1)


class Swap(frobenius.Swap, Channel):
    def dagger(self):
        return self


class Measure(Channel):
    """Measuring a qubit or qmode corresponds to
    applying a 2 -> 1 spider in the doubled picture.

    >>> dom = qubit @ bit @ qmode @ mode
    >>> print(dom.single())
    bit @ bit @ mode @ mode
    >>> assert Measure(dom).double().cod == dom.single()
    """
    draw_as_measures = True

    def __init__(self, dom):
        cod = Ty(*[Ob._classical[ob.name] for ob in dom.inside])
        kraus = diagram.Id(dom.single())
        super().__init__(name="Measure", kraus=kraus, dom=dom, cod=cod)

    def inflate(self, d):
        r""" A specific choice of inflation for the Measure channel.
        The diagram discards the internal states and measures
        the number of photons in the modes. Only qmodes are inflated.
        The bit, qubit and mode are not inflated.
        """

        diagrams = [self._measure_wire(ob, d) for ob in self.dom]
        return diagram.Diagram.tensor(*diagrams)

    # pylint: disable=invalid-name
    def _measure_wire(self, ob, d):
        """Return the diagram that measures one `ob`."""
        # pylint: disable=import-outside-toplevel
        from optyx.core.zw import Add
        if ob.needs_inflation():
            return Measure(ob ** d) >> CQMap(
                "Gather photons", Add(d), mode ** d, mode
            )
        return Measure(ob)


class Encode(Channel):
    """Encoding a bit or mode corresponds to
    applying a 1 -> 2 spider in the doubled picture.

    >>> dom = qubit @ bit @ qmode @ mode
    >>> assert len(Encode(dom).double().cod) == 8
    """
    draw_as_measures = True

    def __init__(self,
                 dom,
                 internal_states: tuple[list[int]] = None):
        cod = Ty(*[Ob._quantum[ob.name] for ob in dom.inside])
        kraus = diagram.Id(dom.single())
        if internal_states is not None:
            if not isinstance(internal_states, tuple):
                internal_states = (internal_states,)
            assert len(internal_states) == sum(
                [1 if ob.name == "mode" else 0 for ob in dom.inside]
            ), "# of internal states must match the number of modes in dom"
            assert len(set(len(i) for i in internal_states)) == 1, \
                "All internal states must be of the same length"

        super().__init__(name="Encode", kraus=kraus, dom=dom, cod=cod)
        self.internal_states = internal_states

    def inflate(self, d):
        r"""
        The internal states are used to encode the modes only.
        Bit and qubit are not encoded, qmode is inflated and
        mode is encoded.
        The diagram is a dagger of the inflation of
        the Measure channel with the difference
        that instead of discarding becoming a maximally mixed state,
        we apply the encoding of the internal states.
        """

        if any(
            ob.name == "mode" for ob in self.dom.inside
        ):
            assert self.internal_states is not None, \
                "Internal states must be provided for encoding"
            assert all(
                len(internal_state) == d for
                internal_state in self.internal_states
            ), "All internal states must have length d"

        amps_iter = iter(self.internal_states or [])
        diagrams = [self._encode_wire(ob, d, amps_iter) for ob in self.dom]
        return diagram.Diagram.tensor(*diagrams)

    def _encode_wire(self, ob, d, amps_iter):
        """Return the diagram that encodes *one* object `ob`.

        `amps_iter` yields the internal‑state vectors for `mode` wires.
        """
        # pylint: disable=import-outside-toplevel
        from optyx.core.zw import Add, Endo

        if ob == mode:
            amps = next(amps_iter)
            amp_layer = diagram.Diagram.tensor(*[Endo(a) for a in amps])
            return (
                CQMap("Add†", Add(d).dagger(), mode, mode ** d)
                >> Encode(mode ** d)
                >> Channel("Amplitudes", amp_layer)
            )
        if ob == qmode:
            return Encode(qmode ** d)
        return Encode(ob)


class Discard(Channel):
    """Discarding a qubit or qmode corresponds to
    applying a 2 -> 0 spider in the doubled picture.

    >>> assert Discard(qmode).double() == diagram.Spider(2, 0, diagram.mode)
    """
    draw_as_discards = True

    def __init__(self, dom):
        env = dom.single()
        kraus = diagram.Id(dom.single())
        super().__init__("Discard", kraus, dom=dom, cod=Ty(), env=env)

    def inflate(self, d):
        """
        Distinguishable setting for the Discard channel.
        """
        return Discard(self.dom.inflate(d))


class Category(frobenius.Category):  # pragma: no cover
    """
    A hypergraph category is a compact category with a method :code:`spiders`.
    Parameters:
        ob : The objects of the category, default is :class:`Ty`.
        ar : The arrows of the category, default is :class:`Diagram`.
    """
    ob, ar = Ty, Diagram


class Functor(frobenius.Functor):  # pragma: no cover
    """
    A hypergraph functor is a compact functor that preserves spiders.
    Parameters:
        ob (Mapping[Ty, Ty]) : Map from atomic :class:`Ty` to :code:`cod.ob`.
        ar (Mapping[Box, Diagram]) : Map from :class:`Box` to :code:`cod.ar`.
        cod (Category) : The codomain of the functor.
    """
    dom = cod = Category()

    def __call__(self, other):
        return frobenius.Functor.__call__(self, other)


class Hypergraph(hypergraph.Hypergraph):  # pragma: no cover
    category, functor = Category, Functor


Hypergraph.ty_factory = Ty
Diagram.spider_factory = Spider
Diagram.hypergraph_factory = Hypergraph
Diagram.braid_factory = Swap
Diagram.sum_factory = Sum
