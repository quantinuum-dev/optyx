"""
ZX diagrams and their mapping to :class:`qpath.Diagram`.

.. autosummary::
    :template:
    :nosignatures:
    :toctree:

    zx_to_path

Example
-------

Evaluating ZX diagrams using PyZX or using `zx_to_path` are equivalent.

>>> ket = lambda *xs: zx.Id(0).tensor(*[zx.X(0, 1, 0.5 if x == 1 else 0) for x in xs])
>>> cnot = zx.Z(1, 2) @ zx.Id(1) >> zx.Id(1) @ zx.X(2, 1)
>>> control = lambda x: ket(x) @ zx.Id(1) >> cnot >> ket(x).dagger() @ zx.Id(1)
>>> assert np.allclose(zx_to_path(control(0)).eval(1).array, control(0).to_pyzx().to_tensor())
>>> assert np.allclose(zx_to_path(control(1)).eval(1).array, control(1).to_pyzx().to_tensor())
>>> cz = lambda phi: cnot >> zx.Z(1, 1, phi) @ zx.H
>>> amplitude = ket(1, 1) >> cz(0.7) >> ket(1, 1).dagger()
>>> assert np.allclose(zx_to_path(amplitude).eval().array, \\
...                    amplitude.to_pyzx().to_tensor())

Corner case where `to_pyzx` and `zx_to_path` agree only up to global phase.

>>> diagram = zx.X(0, 2) @ zx.Z(0, 1, 0.25) >> zx.Id(1) @ zx.Z(2, 1) >> zx.X(2, 0, 0.35)
>>> print(decomp(diagram)[:3])
X(0, 1) >> H >> Z(1, 2)
>>> print(zx_to_path(diagram)[:2])
Create() >> PRO(1) @ Create((0,))
>>> pyzx_prob = np.absolute(diagram.to_pyzx().to_tensor()) ** 2
>>> assert np.allclose(pyzx_prob, zx_to_path(diagram).prob().array)
"""

import numpy as np
from discopy.quantum import zx
from discopy import symmetric, rigid, frobenius

from optyx import qpath


def make_spiders(n):
    """ Constructs the Z spider 1 -> n from spiders 1 -> 2.

    >>> assert len(make_spiders(6)) == 5
    """
    spider = zx.Id(1)
    for k in range(n - 1):
        spider = spider >> zx.Z(1, 2) @ zx.Id(k)
    return spider

def decomp_ar(box):
    """ Decomposes a ZX diagram into Z spiders with at most two inputs/outputs and hadamards.
    
    >>> assert decomp(zx.X(2, 3, 0.25)) == zx.H @ zx.H >> zx.Z(2, 1)\\
    ...     >> zx.Z(1, 1, 0.25) >> zx.Z(1, 2) >> zx.Z(1, 2) @ zx.Id(1) >> zx.H @ zx.H @ zx.H
    """
    n, m = len(box.dom), len(box.cod)
    if isinstance(box, zx.X):
        phase = box.phase
        if (n, m) in ((1, 0), (0, 1)):
            return box
        box = zx.Id().tensor(*[zx.H] * n) >> zx.Z(n, m, phase) >> zx.Id().tensor(*[zx.H] * m)
        return decomp(box)
    if isinstance(box, zx.Z):
        phase = box.phase
        rot = zx.Id(1) if phase == 0 else zx.Z(1, 1, phase)
        if n == 0:
            return zx.X(0, 1) >> zx.H >> rot >> make_spiders(m)
        if m == 0:
            return zx.X(1, 0) << zx.H << rot << make_spiders(n).dagger()
        return make_spiders(n).dagger() >> rot >> make_spiders(m)
    return box


decomp = symmetric.Functor(ob=lambda x: x, ar=decomp_ar,
                           cod=frobenius.Category(rigid.PRO, zx.Diagram))

unit = qpath.Create(0)
counit = qpath.Select(0)
create = qpath.Create(1)
annil = qpath.Select(1)
comonoid = qpath.Split()
monoid = qpath.Merge()
BS = qpath.BS
Id = qpath.Id

def ar_zx2path(box):
    """ Mapping from ZX generators to QPath diagrams
    
    >>> zx2path(decomp(zx.X(0, 1) @ zx.X(0, 1) >> zx.Z(2, 1))).eval()
    Amplitudes([2.+0.j, 0.+0.j], dom=1, cod=2)
    >>> zx2path(zx.Scalar(0.35))
    optyx.qpath.Scalar('Scalar(0.35)', monoidal.PRO(0), monoidal.PRO(0))
    """
    n, m = len(box.dom), len(box.cod)
    if isinstance(box, zx.Scalar):
        return qpath.Scalar(box.data)
    if isinstance(box, zx.X):
        phase = 1 + box.phase if box.phase < 0 else box.phase
        root2 = qpath.Scalar(2 ** 0.5)
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
    if isinstance(box, zx.Z):
        phase = box.phase
        if (n, m) == (0, 1):
            return create >> comonoid
        if (n, m) == (1, 1):
            return qpath.Id(1) @ qpath.Phase(phase)
        if (n, m, phase) == (2, 1, 0):
            return Id(1) @ (monoid >> annil) @ Id(1)
        if (n, m, phase) == (1, 2, 0):
            plus = create >> comonoid
            bot = (plus >> Id(1) @ plus @ Id(1)) @ (Id(1) @ plus @ Id(1))
            mid = Id(2) @ BS.dagger() @ BS @ Id(2)
            fusion = Id(1) @ plus.dagger() @ Id(1) >> plus.dagger()
            return bot >> mid >> (Id(2) @ fusion @ Id(2))
    if box == zx.H:
        hbs_array = (1/2) ** (1/2) * np.array([[1, 1], [1, -1]])
        hadamard_bs = qpath.Gate('HBS', 2, 2, hbs_array)
        return hadamard_bs
    raise NotImplementedError(f'No translation of {box} in QPath.')


zx2path = symmetric.Functor(ob=lambda x: 2 * len(x), ar=ar_zx2path,
                            cod=symmetric.Category(int, qpath.Diagram))

def zx_to_path(diagram: zx.Diagram) -> qpath.Diagram:
    """ 
    Dual-rail encoding of any ZX diagram as a QPath diagram.
    
    >>> diagram = zx.Z(2, 1, 0.25) >> zx.X(1, 1, 0.35)
    >>> print(decomp(diagram))
    Z(2, 1) >> Z(1, 1, 0.25) >> H >> Z(1, 1, 0.35) >> H
    >>> print(zx2path(decomp(diagram))[:3])
    PRO(1) @ Merge() @ PRO(1) >> PRO(1) @ Select() @ PRO(1) >> PRO(1) @ Phase(0.25)
    >>> assert zx2path(decomp(diagram)) == zx_to_path(diagram)
    """
    return zx2path(decomp(diagram))
