"""
Implements Quantum Channels

The discarding map corresponds to the cap.

>>> discard = Channel("Discard", qubit, Ty(), kraus=optyx.Id(optyx.bit), env=optyx.bit)
>>> assert discard.double() == optyx.Spider(2, 0, optyx.bit)

Encoding and measuring a qubit correspond to spiders.

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
    _single_dict = {"bit": "bit", "mode":"mode", "qubit": "bit", "qmode": "mode"}

    @property
    def is_classical(self):
        return False if self.name in ["qubit", "qmode"] else True

    @property
    def single(self): 
        return optyx.Ty(self._single_dict[self.name])

    @property
    def double(self):
        if self.is_classical:
            return optyx.Ty(self.name)
        else:
            name = self._single_dict[self.name]
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
        ob= lambda x: x.double()
        ar= lambda f: f.double()
        dom = symmetric.Category(Ty, Circuit)
        cod = symmetric.Category(optyx.Ty, optyx.Diagram)
        return symmetric.Functor(ob, ar, dom, cod)(self)

    def is_pure(self):
        return min([box.is_pure for box in self])


class Channel(symmetric.Box, Circuit):
    """Channel defined by its Kraus map."""
    def __init__(self, name, dom, cod, kraus, env=optyx.Ty()):
        assert isinstance(kraus, optyx.Diagram)
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
        swap = optyx.Id(cod @ self.env) @ optyx.Diagram.swap(cod, self.env)
        discards = optyx.Id(cod) @ optyx.Diagram.spiders(2, 0, self.env) @ optyx.Id(cod)
        bot_perm = optyx.Diagram.permutation(get_perm(2 * len(cod)), cod @ cod).dagger()
        bot_spiders = get_spiders(self.cod).dagger()
        return top_spiders >> top_perm >> self.kraus @ self.kraus >> swap >> discards >> bot_perm >> bot_spiders




        
        

