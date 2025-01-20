"""
Implements Quantum Channels


Examples
--------

A Channel is initialised by its Kraus map from `dom` to `cod @ env`. 

>>> from optyx.lo import BS, Phase
>>> from optyx.zw import Create, Select
>>> circ = BS @ Phase(0.25) >> Phase(0.25) @ BS
>>> channel = Channel('circuit', circ)
>>> state = Channel('state', Create(1, 1, 1))
>>> effect = Channel('effect', Select(2, 0, 1))
>>> prob = (state >> channel >> effect).double().to_zw().to_tensor().eval().array
>>> amp = (Create(1, 1, 1) >> circ >> Select(2, 0, 1)).to_zw().to_tensor().eval().array
>>> import numpy as np
>>> assert np.allclose(prob, np.absolute(amp) ** 2)

Measuring, discarding and encoding classical information are modeled by spiders.

>>> discard = Channel("Discard", kraus=optyx.Id(optyx.mode), dom=qmode, cod=Ty(), env=optyx.mode)
>>> assert discard.double() == optyx.Spider(2, 0, optyx.mode)
>>> encode = Channel("Encode", kraus = optyx.Id(optyx.mode), dom=mode, cod=qmode)
>>> measure = Channel("Measure", kraus = optyx.Id(optyx.mode), dom=qmode, cod=mode)
>>> decoherence = optyx.Spider(2, 1, optyx.mode) >> optyx.Spider(1, 2, optyx.mode)
>>> assert (measure >> encode).double() == decoherence

We can model photon loss with discarding.

>>> import numpy as np
>>> kraus = lambda nu: zw.W(2) >> zw.Endo(np.sqrt(nu)) @ zw.Endo(np.sqrt(1 - nu)) 
>>> loss = lambda nu: Channel('Loss(' + str(nu) + ')', kraus(nu), dom=qmode, cod=qmode, env=optyx.mode)
>>> eff = 0.95
>>> lossy_channel = channel >> loss(eff) @ loss(eff) @ loss(eff)
>>> lossy_prob = (state >> lossy_channel >> effect).double().to_zw().to_tensor().eval().array
>>> assert np.allclose(lossy_prob, prob * (eff ** 3))
"""

from __future__ import annotations

from discopy import symmetric
from discopy.cat import factory, rsubs
from optyx import optyx, zw, zx

class Ob(symmetric.Ob):
    """Basic object: bit, mode, qubit or qmode"""
    _classical = {"bit": "bit", "mode":"mode", "qubit": "bit", "qmode": "mode"}
    _quantum = {"bit": "qubit", "mode":"qmode", "qubit": "qubit", "qmode": "qmode"}

    @property
    def is_classical(self):
        return False if self.name in ["qubit", "qmode"] else True

    @property
    def single(self): 
        return optyx.Ty(self._classical[self.name])

    @property
    def double(self):
        if self.is_classical:
            return optyx.Ty(self.name)
        else:
            name = self._classical[self.name]
            return optyx.Ty(name, name)
        raise NotImplementedError()

@factory
class Ty(symmetric.Ty):
    """Classical and quantum types."""
    ob_factory = Ob

    def single(self):
        """Returns the optyx.Ty obtained by mapping qubit to bit and qmode to mode"""
        return optyx.Ty().tensor(*[ob.single for ob in self.inside])

    def double(self):
        """Returns the optyx.Ty obtained by mapping qubit to bit @ bit and qmode to mode @ mode"""
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
    """Arbitrary classical-quantum circuits over bits and modes"""
    ty_factory = Ty

    def double(self):
        """ Returns the optyx.Diagram obtained by doubling every quantum dimension
        and building the completely positive map."""
        ob= lambda x: x.double()
        ar= lambda f: f.double()
        dom = symmetric.Category(Ty, Circuit)
        cod = symmetric.Category(optyx.Ty, optyx.Diagram)
        return symmetric.Functor(ob, ar, dom, cod)(self)

    def is_pure(self):
        return min([box.is_pure for box in self])


class Channel(symmetric.Box, Circuit):
    """A Channel is defined by its Kraus map, from dom.single() to cod.single() @ env and interpreted as a completely positive map by doubling.
    Every classical object in dom is encoded and decoded using optyx.Spider, see `Channel.double'."""
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

    @property
    def is_pure(self):
        if self.env != optyx.Ty():
            return False
        for ob in (self.dom @ self.cod).inside:
            if ob.is_classical:
                return False
        return True

    def double(self):
        def get_spiders(dom):
            spiders = optyx.Id()
            for ob in dom.inside:
                if ob.is_classical:
                    box = optyx.Spider(1, 2, ob.single)
                else:
                    box = optyx.Id(ob.double) 
                spiders @= box
            return spiders
        
        get_perm = lambda n: sorted(sorted(list(range(n))), key=lambda i: i % 2)
        cod = self.cod.single()
        top_spiders = get_spiders(self.dom)
        top_perm = optyx.Diagram.permutation(get_perm(len(top_spiders.cod)), top_spiders.cod)
        swap_env = optyx.Id(cod @ self.env) @ optyx.Diagram.swap(cod, self.env)
        discards = optyx.Id(cod) @ optyx.Diagram.spiders(2, 0, self.env) @ optyx.Id(cod)
        bot_perm = optyx.Diagram.permutation(get_perm(2 * len(cod)), cod @ cod).dagger()
        bot_spiders = get_spiders(self.cod).dagger()
        return top_spiders >> top_perm >> self.kraus @ self.kraus.conjugate() >> swap_env >> discards >> bot_perm >> bot_spiders

