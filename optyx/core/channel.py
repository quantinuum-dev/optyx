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
"""

from __future__ import annotations

from discopy import tensor
from discopy import symmetric, frobenius
from discopy.cat import factory
from optyx.core import zx, diagram
from pytket.extensions.pyzx import pyzx_to_tk
from pyzx import extract_circuit

from optyx._utils import explode_channel


class Ob(symmetric.Ob):
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
class Ty(symmetric.Ty):
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
        assert isinstance(ty, diagram.Ty)
        # pylint: disable=protected-access
        return Ty(*[Ob._quantum[ob.name] for ob in ty.inside])

    def needs_inflation(self) -> bool:
        return "qmode" in self.name

    # pylint: disable=invalid-name
    def inflate(self, d) -> Ty:
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
        return self.dom.needs_inflation() or self.cod.needs_inflation()

    # pylint: disable=invalid-name
    def inflate(self, d):
        r"""Translates from an indistinguishable setting
        to a distinguishable one. For a map on :math:`F(\mathbb{C})`,
        obtain a map on :math:`F(\mathbb{C})^{\widetilde{\otimes} d}`."""
        assert isinstance(d, int), "Dimension must be an integer"
        assert d > 0, "Dimension must be positive"

        dom = symmetric.Category(Ty, Diagram)
        cod = symmetric.Category(Ty, Diagram)

        return symmetric.Functor(
            lambda x: x.inflate(d),
            lambda f: f.inflate(d),
            dom,
            cod
        )(self)

    def double(self):
        """Returns the diagram.Diagram obtained by
        doubling every quantum dimension
        and building the completely positive map."""
        dom = symmetric.Category(Ty, Diagram)
        cod = symmetric.Category(diagram.Ty, diagram.Diagram)
        return symmetric.Functor(
            lambda x: x.double(), lambda f: f.double(), dom, cod
        )(self)

    @property
    def is_pure(self):
        are_layers_pure = []
        for layer in self:
            generator = layer.inside[0][1]

            are_layers_pure.append(
                any(ty.is_classical for ty in generator.cod.inside) or
                any(ty.is_classical for ty in generator.dom.inside) or
                isinstance(generator, Discard)
            )

        return not any(are_layers_pure)

    def get_kraus(self):
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

        return symmetric.Functor(
            ob=len,
            ar=lambda f: f.get_kraus().to_path(dtype),
            cod=symmetric.Category(int, path.Matrix[dtype]),
        )(self)

    def _decomp(self):

        # pylint: disable=protected-access
        return symmetric.Functor(
            ob=lambda x: qubit**len(x),
            ar=lambda arr: arr._decomp(),
            cod=symmetric.Category(Ty, Diagram),
        )(self)

    def to_dual_rail(self):
        """Convert to dual-rail encoding."""

        assert self.is_pure, "Diagram must be pure to convert to dual rail."

        return symmetric.Functor(
            ob=lambda x: qmode**(2*len(x)),
            ar=lambda arr: arr.to_dual_rail(),
            cod=symmetric.Category(Ty, Diagram),
        )(self._decomp())

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

        assert self.is_pure, "Diagram must be pure for conversion."

        return self.get_kraus().to_pyzx()

    def to_discopy(self):
        raise NotImplementedError(
            "Conversion to discopy is not implemented for optyx diagrams."
        )

    @classmethod
    def from_tket(cls, tket_circuit):
        """Convert from tket circuit."""
        # pylint: disable=import-outside-toplevel
        from optyx.qubit import Circuit
        return Circuit(tket_circuit)

    @classmethod
    def from_pyzx(cls, pyzx_circuit):
        """Convert from PyZX circuit."""
        # pylint: disable=import-outside-toplevel
        from optyx.qubit import Circuit
        return Circuit(pyzx_circuit)

    @classmethod
    def from_discopy(cls, discopy_circuit):
        """Convert from discopy circuit."""
        # pylint: disable=import-outside-toplevel
        from optyx.qubit import Circuit
        return Circuit(discopy_circuit)

    @classmethod
    def from_bosonic_operator(cls, n_modes, operators, scalar=1):
        return Channel(
            "Bosonic operator",
            diagram.Diagram.from_bosonic_operator(
                n_modes, operators, scalar=scalar
            )
        )


class Channel(symmetric.Box, Diagram):
    """
    Channel initialised by its Kraus map.
    """

    def __init__(self, name, kraus, dom=None, cod=None, env=diagram.Ty()):
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

    # pylint: disable=invalid-name
    def __pow__(self, n):
        if n == 1:
            return self
        return self @ self ** (n - 1)

    def dagger(self):
        return Channel(
            name=self.name + ".dagger()",
            kraus=self.kraus.dagger(),
            dom=self.cod,
            cod=self.dom,
        )

    def _decomp(self):
        # pylint: disable=import-outside-toplevel
        from optyx.qubit import QubitChannel
        decomposed = zx.decomp(self.kraus)
        return explode_channel(
            decomposed,
            QubitChannel,
            Diagram
        )

    def to_dual_rail(self):
        raise NotImplementedError(
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

    def eval(self, n_photons=0, permanent=None, dtype=complex):
        """Evaluate the sum of diagrams."""
        # we need to implement the proper sums of qpath diagrams
        # this is only a temporary solution, so that the grad tests pass
        if permanent is None:
            # pylint: disable=import-outside-toplevel
            from optyx.core.path import npperm

            permanent = npperm
        return sum(
            term.to_path(dtype).eval(n_photons, permanent)
            for term in self.terms
        )


class CQMap(symmetric.Box, Diagram):
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


class Swap(symmetric.Swap, Channel):
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
    # pylint: disable=no-self-use
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


# class BitFlipError(Channel):

#     def __init__(self, prob):
#         x_error = zx.X(1, 2) >> zx.Id(1) @ zx.ZBox(
#             1, 1, np.sqrt((1 - prob) / prob)
#         ) @ zx.Scalar(np.sqrt(prob * 2))
#         super().__init__(
#             name=f"BitFlipError({prob})",
#             kraus=x_error,
#             dom=qubit,
#             cod=qubit,
#             env=diagram.bit,
#         )

#     def dagger(self):
#         return self


# class DephasingError(Channel):
#     def __init__(self, prob):
#         z_error = (
#             zx.H
#             >> zx.X(1, 2)
#             >> zx.H
#             @ zx.ZBox(1, 1, np.sqrt((1 - prob) / prob))
#             @ zx.Scalar(np.sqrt(prob * 2))
#         )
#         super().__init__(
#             name=f"DephasingError({prob})",
#             kraus=z_error,
#             dom=qubit,
#             cod=qubit,
#             env=diagram.bit,
#         )

#     def dagger(self):
#         return self


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


Diagram.braid_factory = Swap
Diagram.sum_factory = Sum
