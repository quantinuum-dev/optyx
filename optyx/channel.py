"""
Implements Quantum Channels

The discarding map corresponds to the cap.

>>> discard = Channel("Discard", qubit, Ty(), kraus=optyx.Id(optyx.bit), env=optyx.bit)
>>> assert discard.double() == optyx.Spider(2, 0, optyx.bit)

Encoding and measuring a qubit correspond to spiders .

>>> encode = Channel("Encode", bit, qubit, kraus = optyx.Id(optyx.bit))
>>> measure = Channel("Measure", qubit, bit, kraus = optyx.Id(optyx.bit))
>>> result = optyx.Spider(2, 1, optyx.bit) >> optyx.Spider(1, 2, optyx.bit)
>>> assert (measure >> encode).double() == result

We can model photon loss with discarding.

>>> import numpy as np
>>> kraus = lambda nu: zw.W(2) >> zw.Endo(np.sqrt(nu)) @ zw.Endo(np.sqrt(1 - nu)) 
>>> loss = lambda nu: Channel(str(nu), qmode, qmode, kraus(nu), env=optyx.mode)
"""

from __future__ import annotations

from discopy import symmetric
from discopy.cat import factory, rsubs
from optyx import optyx, zw, zx

class Ob(symmetric.Ob):
    """Basic object: bit, mode, qubit or qmode"""

@factory
class Ty(symmetric.Ty):
    """Classical and quantum types."""
    ob_factory = Ob

    def single(self):
        """Returns the optyx.Ty obtained by mapping qubit to bit and qmode to mode"""
        single = lambda x: "bit" if x == "qubit" else "mode" if x == "qmode" else x
        return optyx.Ty(*[single(x.name) for x in self])

    def double(self):
        """Returns the optyx.Ty obtained by mapping qubit to bit @ bit and qmode to mode @ mode"""
        double = optyx.Ty()
        for ty in self:
            if ty.name in ["bit", "mode"]:
                double @= optyx.Ty(ty.name)
            if ty.name == "qubit":
                double @= optyx.bit @ optyx.bit
            if ty.name == "qmode":
                double @= optyx.mode @ optyx.mode
        return double


bit = Ty("bit")
mode = Ty("mode")
qubit = Ty("qubit")
qmode = Ty("qmode")


@factory
class Circuit(symmetric.Diagram):
    """Arbitrary classical-quantum circuits over bits and modes"""
    ty_factory = Ty

    def double(self):
        """ Returns the optyx.Diagram obtained by doubling every quantum dimension and building the CP map."""
        ob, ar= lambda x: x.double(), lambda f: f.double()
        dom = symmetric.Category(Ty, Circuit)
        cod = symmetric.Category(optyx.Ty, optyx.Diagram)
        F = symmetric.Functor(ob, ar, dom, cod)
        return F(self)


class Channel(symmetric.Box, Circuit):
    """Channel defined by its Kraus map."""
    def __init__(self, name, dom, cod, kraus, env=optyx.Ty()):
        assert isinstance(kraus, optyx.Diagram)
        assert kraus.dom == dom.single()
        assert kraus.cod == cod.single() @ env
        self.kraus = kraus
        self.env = env
        super().__init__(name, dom, cod)

    def double(self):
        def get_spiders(dom):
            spiders = optyx.Id()
            for x in dom:
                if x.name in ["qubit", "qmode"]:
                    box = optyx.Id(Ty(x.name).double()) 
                else:
                    box = optyx.Spider(1, 2, optyx.Ty(x.name))
                spiders @= box
            return spiders
        
        get_perm = lambda n: sorted(sorted(list(range(n))), key=lambda i: i % 2)
        typ = self.cod.single()
        top_spiders = get_spiders(self.dom)
        top_perm = optyx.Diagram.permutation(get_perm(len(top_spiders.cod)), top_spiders.cod)
        swap = optyx.Id(typ @ self.env) @ optyx.Diagram.swap(typ, self.env)
        discards = optyx.Id(typ) @ optyx.Diagram.spiders(2, 0, self.env) @ optyx.Id(typ)
        bot_perm = optyx.Diagram.permutation(get_perm(2 * len(typ)), typ @ typ).dagger()
        bot_spiders = get_spiders(self.cod).dagger()
        return top_spiders >> top_perm >> self.kraus @ self.kraus >> swap >> discards >> bot_perm >> bot_spiders




        
        

