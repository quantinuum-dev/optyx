"""

Overview
--------

Implements classical-quantum channels.

Quantum channels are completely positive maps acting on
the doubled space :code:`H @ H` for a Hilbert space :code:`H`.
These can be initialised from the Kraus decomposition,
given as an :code:`optyx.Diagram` with domain :code:`H` and
codomain :code:`H @ E` for an auxiliary space :code:`E`,
called the environment, which is not observed.

Channels can moreover have a classical interface,
in the form of input :code:`bit` or :code:`mode` types.
The Kraus map is then given by an :class:`optyx.Diagram`
with domain :code:`H @ C` and codomain :code:`H @ C @ E`,
where the classical type :code:`C` represents
the classical inputs or outputs of the computation.
In the doubled picture, encoding or measuring a classical type
is implemented through instances of :class:`optyx.Spider`.

This module allows to build an arbitrary syntactic :class:`Circuit`
from instances of :class:`Channel`.
The :code:`Circuit.double` method returns an :class:`optyx.Diagram`,
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

    Circuit
    Channel
    Measure
    Encode
    Discard


Examples
--------

A Channel is initialised by its Kraus map from `dom` to `cod @ env`.

>>> from optyx import lo, zx, zw
>>> circ = lo.Phase(0.25) @ lo.BS @ lo.Phase(0.56) >> lo.BS @ lo.BS
>>> channel = Channel(name='circuit', kraus=circ,\\
...                   dom=qmode ** 4, cod=qmode ** 4, env=optyx.Ty())

We can check that this channel is causal:

>>> import numpy as np
>>> discards = Discard(qmode ** 4)
>>> rhs = (channel >> discards).double().to_zw().to_tensor().eval().array
>>> lhs = (discards).double().to_zw().to_tensor().eval().array
>>> assert np.allclose(lhs, rhs)

We can calculate the probability of an input-output pair:

>>> state = Channel('state', zw.Create(1, 0, 1, 0))
>>> effect = Channel('effect', zw.Select(1, 0, 1, 0))
>>> prob = (state >> channel >> effect).double(\\
...     ).to_zw().to_tensor().eval().array
>>> amp = (zw.Create(1, 0, 1, 0) >> circ >> zw.Select(1, 0, 1, 0)\\
...     ).to_zw().to_tensor().eval().array
>>> assert np.allclose(prob, np.absolute(amp) ** 2)

We can check that the probabilities of a normalised state sum to 1:

>>> bell_state = Channel('Bell', optyx.Scalar(1/np.sqrt(2)) @ zx.Z(0, 2))
>>> dual_rail = Channel('2R', optyx.dual_rail(2))
>>> measure = Discard(qmode ** 3) @ Measure(qmode)
>>> setup = bell_state >> dual_rail >> channel >> measure
>>> assert np.isclose(sum(setup.double().to_zw().to_tensor().eval().array), 1)

We can construct a lossy optical channel and compute its probabilities:

>>> eff = 0.95
>>> kraus = zw.W(2) >> zw.Endo(np.sqrt(eff)) @ zw.Endo(np.sqrt(1 - eff))
>>> loss = Channel(str(eff), kraus, dom=qmode, cod=qmode, env=optyx.mode)
>>> uniform_loss = loss.tensor(*[loss for _ in range(3)])
>>> lossy_channel = channel >> uniform_loss
>>> lossy_prob = (state >> lossy_channel >> effect).double(\\
...     ).to_zw().to_tensor().eval().array
>>> assert np.allclose(lossy_prob, prob * (eff ** 2))
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from discopy import symmetric
from discopy.cat import factory

from optyx import optyx, zx


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
        """Maps :code:`qubit` to :code:`optyx.bit`
        and :code:`qmode` to :code:`optyx.mode`."""
        return optyx.Ty(self._classical[self.name])

    @property
    def double(self):
        """Maps :code:`qubit` to :code:`optyx.bit @ optyx.bit`
        and :code:`qmode` to :code:`optyx.mode @ optyx.mode`."""
        if self.is_classical:
            return optyx.Ty(self.name)
        name = self._classical[self.name]
        return optyx.Ty(name, name)


@factory
class Ty(symmetric.Ty):
    """Classical and quantum types."""

    ob_factory = Ob

    def single(self):
        """Returns the optyx.Ty obtained by mapping
        :code:`qubit` to :code:`bit` and :code:`qmode` to :code:`mode`"""
        return optyx.Ty().tensor(*[ob.single for ob in self.inside])

    def double(self):
        """Returns the optyx.Ty obtained by mapping
        :code:`qubit` to :code:`bit @ bit`
        and :code:`qmode` to :code:`mode @ mode`"""
        return optyx.Ty().tensor(*[ob.double for ob in self.inside])

    @staticmethod
    def from_optyx(ty):
        assert isinstance(ty, optyx.Ty)
        return Ty(*[Ob._quantum[ob.name] for ob in ty.inside])


bit = Ty("bit")
mode = Ty("mode")
qubit = Ty("qubit")
qmode = Ty("qmode")


@factory
class Circuit(symmetric.Diagram):
    """Classical-quantum circuits over qubits and optical modes"""

    ty_factory = Ty

    def inflate(self, d):
        r"""Translates from an indistinguishable setting
        to a distinguishable one. For a map on :math:`\mathbb{C}^d`,
        obtain a map on :math:`F(\mathbb{C})^{\widetilde{\otimes} d}`."""
        assert isinstance(d, int), "Dimension must be an integer"
        assert d > 0, "Dimension must be positive"

        dom = symmetric.Category(Ty, Circuit)
        cod = symmetric.Category(Ty, Circuit)

        def ob(x):
            return (mode**0).tensor(
                *(o**d if o.name in ["qmode"] else o for o in x)
            )

        return symmetric.Functor(lambda x: ob(x),
                                 lambda f: f.inflate(d),
                                 dom,
                                 cod)(self)

    def double(self):
        """Returns the optyx.Diagram obtained by
        doubling every quantum dimension
        and building the completely positive map."""
        dom = symmetric.Category(Ty, Circuit)
        cod = symmetric.Category(optyx.Ty, optyx.Diagram)
        return symmetric.Functor(
            lambda x: x.double(), lambda f: f.double(), dom, cod
        )(self)


class Channel(symmetric.Box, Circuit):
    """
    Channel initialised by its Kraus map.
    """

    def __init__(self, name, kraus, dom=None, cod=None, env=optyx.Ty()):
        assert isinstance(kraus, optyx.Diagram)
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
        Returns the :class:`optyx.Diagram` representing
        the action of the channel as a CP map on the doubled space.
        """

        def get_spiders(dom):
            spiders = optyx.Id()
            for ob in dom.inside:
                if ob.is_classical:
                    box = optyx.Spider(1, 2, ob.single)
                else:
                    box = optyx.Id(ob.double)
                spiders @= box
            return spiders

        def get_perm(n):
            return sorted(sorted(list(range(n))), key=lambda i: i % 2)

        cod = self.cod.single()
        top_spiders = get_spiders(self.dom)
        top_perm = optyx.Diagram.permutation(
            get_perm(len(top_spiders.cod)), top_spiders.cod
        )
        swap_env = optyx.Id(cod @ self.env) @ optyx.Diagram.swap(
            cod, self.env
        )
        discard = (
            optyx.Id(cod)
            @ optyx.Diagram.spiders(2, 0, self.env)
            @ optyx.Id(cod)
        )
        new_cod = optyx.Ty().tensor(*[ty @ ty for ty in cod])
        bot_perm = optyx.Diagram.permutation(
            get_perm(2 * len(cod)), new_cod
        ).dagger()
        bot_spiders = get_spiders(self.cod).dagger()
        top = top_spiders >> top_perm
        bot = swap_env >> discard >> bot_perm >> bot_spiders
        return top >> self.kraus @ self.kraus.conjugate() >> bot

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

    def inflate(self, d):
        r"""
        Translates from an indistinguishable setting
        to a distinguishable one. For a map on :math:`\mathbb{C}^d`,
        obtain a map on :math:`F(\mathbb{C})^{\widetilde{\otimes} d}`.
        """
        assert isinstance(d, int), "Dimension must be an integer"
        assert d > 0, "Dimension must be positive"

        def ob(x):
            return (mode**0).tensor(
                *(o**d if o.name in ["qmode"] else o for o in x)
            )

        def arr(f):
            if (
                any(o.name == "qmode" for o in self.dom.inside) or
                any(o.name == "qmode" for o in self.cod.inside)
            ):
                return f.inflate(d)
            else:
                return f

        return Channel(
            name=self.name + f"^{d}",
            kraus=arr(self.kraus),
            dom=ob(self.dom),
            cod=ob(self.cod),
        )


class CQMap(symmetric.Box, Circuit):
    """
    Channel initialised by its Density matrix.
    """

    def __init__(self, name, density_matrix, dom, cod):
        assert isinstance(density_matrix, optyx.Diagram)
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
        to a distinguishable one. For a map on :math:`\mathbb{C}^d`,
        obtain a map on :math:`F(\mathbb{C})^{\widetilde{\otimes} d}`.
        """
        assert isinstance(d, int), "Dimension must be an integer"
        assert d > 0, "Dimension must be positive"

        def ob(x):
            return (mode**0).tensor(
                *(o**d if o.name in ["qmode"] else o for o in x)
            )

        def arr(f):
            if (
                any(o.name == "qmode" for o in self.dom.inside) or
                any(o.name == "qmode" for o in self.cod.inside)
            ):
                return f.inflate(d)
            else:
                return f

        return CQMap(
            name=self.name + f"^{d}",
            density_matrix=arr(self.density_matrix),
            dom=ob(self.dom),
            cod=ob(self.cod),
        )

class Swap(symmetric.Swap, Channel):
    def dagger(self):
        return self


class Discard(Channel):
    """Discarding a qubit or qmode corresponds to
    applying a 2 -> 0 spider in the doubled picture.

    >>> assert Discard(qmode).double() == optyx.Spider(2, 0, optyx.mode)
    """

    def __init__(self, dom):
        env = dom.single()
        kraus = optyx.Id(dom.single())
        super().__init__('Discard', kraus, dom=dom, cod=Ty(), env=env)


class Measure(Channel):
    """Measuring a qubit or qmode corresponds to
    applying a 2 -> 1 spider in the doubled picture.

    >>> dom = qubit @ bit @ qmode @ mode
    >>> print(dom.single())
    bit @ bit @ mode @ mode
    >>> assert Measure(dom).double().cod == dom.single()
    """

    def __init__(self, dom):
        cod = Ty(*[Ob._classical[ob.name] for ob in dom.inside])
        kraus = optyx.Id(dom.single())
        super().__init__(name="Measure", kraus=kraus, dom=dom, cod=cod)

    def inflate(self, d):
        r"""Translates from an indistinguishable setting
        to a distinguishable one. For a map on :math:`\mathbb{C}^d`,
        obtain a map on :math:`F(\mathbb{C})^{\widetilde{\otimes} d}`.

        The diagram discards the internal states and measures
        the number of photons in the modes.
        """

        assert isinstance(d, int), "Dimension must be an integer"
        assert d > 0, "Dimension must be positive"

        from optyx.zw import Add

        channel = optyx.Diagram.tensor(
            *[(
                Measure(ty**d) >>
                CQMap(
                    "Gather photons",
                    Add(d),
                    mode**d,
                    mode,
                )
               )
            if ty == qmode else
            Measure(ty)
            for ty in self.dom]
        )

        return channel


class Encode(Channel):
    """Encoding a bit or mode corresponds to
    applying a 1 -> 2 spider in the doubled picture.

    >>> dom = qubit @ bit @ qmode @ mode
    >>> assert len(Encode(dom).double().cod) == 8
    """

    def __init__(self, dom, internal_states=None):
        cod = Ty(*[Ob._quantum[ob.name] for ob in dom.inside])
        kraus = optyx.Id(dom.single())
        if internal_states is not None:
            assert len(internal_states) == sum(
                [1 if ob.name == "mode" else 0 for ob in dom.inside]
            ), "Number of internal states must match the number of modes in dom"

        super().__init__(name="Encode", kraus=kraus, dom=dom, cod=cod)
        self.internal_states = internal_states

    def inflate(self, d):
        r"""Translates from an indistinguishable setting
        to a distinguishable one. For a map on :math:`\mathbb{C}^d`,
        obtain a map on :math:`F(\mathbb{C})^{\widetilde{\otimes} d}`.

        The internal states are used to encode the qmodes.
        The diagram is a dagger of the inflation of
        the Measure channel with the difference
        that instead of discarding becoming a maximally mixed state,
        we apply the encoding of the internal states.
        """

        assert isinstance(d, int), "Dimension must be an integer"
        assert d > 0, "Dimension must be positive"
        if any(
            ob.name == "mode" for ob in self.dom.inside
        ):
            assert self.internal_states is not None, \
                "Internal states must be provided for encoding"
            assert all(
                len(internal_state) == d for internal_state in self.internal_states
            ), "All internal states must have the same length as d"

        from optyx.zw import Add, Endo

        diagrams_to_tensor = []
        i = 0

        #only inflate the qmodes
        for ty in self.dom:
            if ty == mode:
                internal_amplitudes = lambda i: optyx.Diagram.tensor(
                    *[
                        Endo(s) for s in self.internal_states[i]
                    ]
                )

                diagrams_to_tensor.append(
                    (
                        CQMap(
                            "Add dagger",
                            Add(d).dagger(),
                            mode,
                            mode**d,
                        ) >>
                        Encode(mode**d) >>
                        Channel(
                            "Amplitudes",
                            internal_amplitudes(i)
                        )
                    )
                )

                i += 1
            elif ty == qmode:
                diagrams_to_tensor.append(Encode(qmode**d))
            else:
                diagrams_to_tensor.append(Encode(ty))

        channel = optyx.Diagram.tensor(*diagrams_to_tensor)

        return channel

class BitFlipError(Channel):

    def __init__(self, prob):
        x_error = zx.X(1, 2) >> zx.Id(1) @ zx.ZBox(
            1, 1, np.sqrt((1 - prob) / prob)
        ) @ zx.Scalar(np.sqrt(prob * 2))
        super().__init__(
            name=f"BitFlipError({prob})",
            kraus=x_error,
            dom=qubit,
            cod=qubit,
            env=optyx.bit,
        )

    def dagger(self):
        return self


class DephasingError(Channel):
    def __init__(self, prob):
        z_error = (
            zx.H
            >> zx.X(1, 2)
            >> zx.H
            @ zx.ZBox(1, 1, np.sqrt((1 - prob) / prob))
            @ zx.Scalar(np.sqrt(prob * 2))
        )
        super().__init__(
            name=f"DephasingError({prob})",
            kraus=z_error,
            dom=qubit,
            cod=qubit,
            env=optyx.bit,
        )

    def dagger(self):
        return self


class Discard(Channel):
    """Discarding a qubit or qmode corresponds to
    applying a 2 -> 0 spider in the doubled picture.

    >>> assert Discard(qmode).double() == optyx.Spider(2, 0, optyx.mode)
    """

    def __init__(self, dom):
        env = dom.single()
        kraus = optyx.Id(dom.single())
        super().__init__("Discard", kraus, dom=dom, cod=Ty(), env=env)


class Ket(Channel):
    """Computational basis state for qubits"""

    def __init__(
        self, value: Literal[0, 1, "+", "-"], cod: Ty = None
    ) -> None:
        spider = zx.X if value in (0, 1) else zx.Z
        phase = 0 if value in (0, "+") else 0.5
        kraus = spider(0, 1, phase) @ optyx.Scalar(1 / np.sqrt(2))
        super().__init__(f"|{value}>", kraus, cod=cod)


class Bra(Channel):
    """Post-selected measurement for qubits"""

    def __init__(
        self, value: Literal[0, 1, "+", "-"], dom: Ty = None
    ) -> None:
        spider = zx.X if value in (0, 1) else zx.Z
        phase = 0 if value in (0, "+") else 0.5
        kraus = spider(1, 0, phase) @ optyx.Scalar(1 / np.sqrt(2))
        super().__init__(f"<{value}|", kraus, dom=dom)


Circuit.braid_factory = Swap
