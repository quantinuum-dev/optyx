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

    _classical = {"bit": "bit", "mode": "mode",
                  "qubit": "bit", "qmode": "mode"}
    _quantum = {"bit": "qubit", "mode": "qmode",
                "qubit": "qubit", "qmode": "qmode"}

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

    def double(self):
        """ Returns the optyx.Diagram obtained by
        doubling every quantum dimension
        and building the completely positive map."""
        dom = symmetric.Category(Ty, Circuit)
        cod = symmetric.Category(optyx.Ty, optyx.Diagram)
        return symmetric.Functor(lambda x: x.double(),
                                 lambda f: f.double(), dom, cod)(self)


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

    def double(self, ):
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
            get_perm(len(top_spiders.cod)), top_spiders.cod)
        swap_env = optyx.Id(cod @ self.env) @ optyx.Diagram.swap(cod, self.env)
        discard = optyx.Id(cod) @ \
            optyx.Diagram.spiders(2, 0, self.env) @ optyx.Id(cod)
        new_cod = optyx.Ty().tensor(*[ty @ ty for ty in cod])
        bot_perm = optyx.Diagram.permutation(
            get_perm(2 * len(cod)), new_cod).dagger()
        bot_spiders = get_spiders(self.cod).dagger()
        top = top_spiders >> top_perm
        bot = swap_env >> discard >> bot_perm >> bot_spiders
        return top >> self.kraus @ self.kraus.conjugate() >> bot

    def __pow__(self, n):
        if n == 1:
            return self
        return self @ self ** (n - 1)


class DensityMatrix(symmetric.Box, Circuit):
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


class Swap(symmetric.Swap, Channel):
    pass


class Measure(Channel):
    """ Measuring a qubit or qmode corresponds to
    applying a 2 -> 1 spider in the doubled picture.

    >>> dom = qubit @ bit @ qmode @ mode
    >>> print(dom.single())
    bit @ bit @ mode @ mode
    >>> assert Measure(dom).double().cod == dom.single()
    """
    def __init__(self, dom):
        cod = Ty(*[Ob._classical[ob.name] for ob in dom.inside])
        kraus = optyx.Id(dom.single())
        super().__init__(name='Measure', kraus=kraus, dom=dom, cod=cod)


class Encode(Channel):
    """Encoding a bit or mode corresponds to
    applying a 1 -> 2 spider in the doubled picture.

    >>> dom = qubit @ bit @ qmode @ mode
    >>> assert len(Encode(dom).double().cod) == 8
    """
    def __init__(self, dom):
        cod = Ty(*[Ob._quantum[ob.name] for ob in dom.inside])
        kraus = optyx.Id(dom.single())
        super().__init__(name='Encode', kraus=kraus, dom=dom, cod=cod)


class BitFlipError(Channel):

    def __init__(self, prob):
        x_error = (
                zx.X(1, 2) >> zx.Id(1)
                @ zx.ZBox(1, 1, np.sqrt((1 - prob) / prob))
                @ zx.Scalar(np.sqrt(prob * 2))
        )
        super().__init__(name=f'BitFlipError({prob})',
                         kraus=x_error, dom=qubit, cod=qubit, env=optyx.bit)


class DephasingError(Channel):
    def __init__(self, prob):
        z_error = (
                zx.H >> zx.X(1, 2)
                >> zx.H @ zx.ZBox(1, 1, np.sqrt((1 - prob) / prob))
                @ zx.Scalar(np.sqrt(prob * 2))
        )
        super().__init__(name=f'DephasingError({prob})',
                         kraus=z_error, dom=qubit, cod=qubit, env=optyx.bit)


class Discard(Channel):
    """Discarding a qubit or qmode corresponds to
    applying a 2 -> 0 spider in the doubled picture.

    >>> assert Discard(qmode).double() == optyx.Spider(2, 0, optyx.mode)
    """
    def __init__(self, dom):
        env = dom.single()
        kraus = optyx.Id(dom.single())
        super().__init__('Discard', kraus, dom=dom, cod=Ty(), env=env)


class Ket(Channel):
    """Computational basis state for qubits"""

    def __init__(self, value: Literal[0, 1, "+", "-"], cod: Ty = None) -> None:
        spider = zx.X if value in (0, 1) else zx.Z
        phase = 0 if value in (0, "+") else 0.5
        kraus = spider(0, 1, phase) @ optyx.Scalar(1 / np.sqrt(2))
        super().__init__(f"|{value}>", kraus, cod=cod)


class Bra(Channel):
    """Post-selected measurement for qubits"""

    def __init__(self, value: Literal[0, 1, "+", "-"], dom: Ty = None) -> None:
        spider = zx.X if value in (0, 1) else zx.Z
        phase = 0 if value in (0, "+") else 0.5
        kraus = spider(1, 0, phase) @ optyx.Scalar(1 / np.sqrt(2))
        super().__init__(f"<{value}|", kraus, dom=dom)

Circuit.braid_factory = Swap
