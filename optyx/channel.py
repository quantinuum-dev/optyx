"""
Implements photonic classical-quantum channels.

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    W
    Z
    Id
    Swap
    Discard
    Prepare
    Measure
    Create
    Select

.. admonition:: Functions

    .. autosummary::
        :template: function.rst
        :nosignatures:
        :toctree:

Note
----
:class:`Channel` implements the classical-quantum photonic processes similar to
the doubling construction of Coecke and Kissinger :cite:`CoeckeKissinger17`
for the usual qubit QC.

Example
-------
"""
import numpy as np
from typing import Union

from discopy.monoidal import PRO
from discopy.monoidal import Layer


import optyx.zw as zw

class W(zw.Diagram):
    """
    Implements the W channel.

    Example
    -------

    """
    def __init__(self, n_legs: int, is_dagger: bool = False):

        dom = PRO(n_legs*2) if is_dagger else PRO(2)
        cod = PRO(2) if is_dagger else PRO(n_legs*2)

        self.n_legs = n_legs

        wire_permutation = [item for sublist in [[i, i+n_legs] for i in range(n_legs)]
                             for item in sublist]

        inside = (
                    Layer(PRO(0), zw.W(n_legs), PRO(1)),
                    Layer(PRO(n_legs), zw.W(n_legs), PRO(0)),
                    Layer(PRO(0), zw.Swap(n_legs*2, n_legs*2, wire_permutation), PRO(0)),
                )

        super().__init__(inside, dom, cod)


class Z(zw.Diagram):
    """
    Implements the Z channel.

    Example
    -------

    """
    def __init__(
        self,
        amplitudes: Union[np.ndarray, list, callable, zw.IndexableAmplitudes],
        legs_in: int,
        legs_out: int,
    ):
        """
        Args:
            amplitudes: The amplitudes of the Z channel.
            legs_in: The number of input legs.
            legs_out: The number of output legs.
        """
        dom = PRO(legs_in*2)
        cod = PRO(legs_out*2)
        self.amplitudes = amplitudes
        self.legs_in = legs_in
        self.legs_out = legs_out

        wire_permutation_in = [item for sublist in [[i, i+legs_in]
                                                    for i in range(legs_in)]
                             for item in sublist]
        wire_permutation_out = [item for sublist in [[i, i+legs_out]
                                                     for i in range(legs_out)]
                                for item in sublist]

        inside = (
                    Layer(PRO(0), zw.Swap(legs_in*2, 
                                          legs_in*2, 
                                          wire_permutation_in).dagger(), 
                          PRO(0)),
                    Layer(PRO(legs_in), zw.Z(amplitudes, 
                                             legs_in, 
                                             legs_out).conjugate(), 
                          PRO(0)),
                    Layer(PRO(0), zw.Z(amplitudes, 
                                       legs_in, 
                                       legs_out), 
                          PRO(legs_out)),
                    Layer(PRO(0), zw.Swap(legs_out*2, 
                                          legs_out*2, 
                                          wire_permutation_out), 
                          PRO(0)),
                )


        super().__init__(inside, dom, cod)

class Id(zw.Diagram):
    """
    Implements the identity channel.

    Example
    -------

    """
    def __init__(self, n_legs: int = 0):
        """
        Args:
            n_legs: The number of legs.
        """
        dom = PRO(n_legs*2)
        cod = PRO(n_legs*2)
        self.n_legs = n_legs
        inside = (
                    Layer(PRO(0), zw.Id(n_legs*2), PRO(0)),
                )

        super().__init__(inside, dom, cod)


class Swap(zw.Diagram):
    """
    Implements the Swap channel.

    Example
    -------

    """
    def __init__(self, permutation: list[int] = [1, 0]):
        """
        Args:
            n_legs: The number of legs.
        """
        self.permutation = permutation
        dom = PRO(len(permutation)*2)
        cod = PRO(len(permutation)*2)

        doubled_permutation = []
        for i in range(len(permutation)*2):
            if i % 2 == 0:
                doubled_permutation.append(permutation[i//2]*2)
            else:
                doubled_permutation.append(permutation[i//2]*2 + 1)
        
        inside = (
                    Layer(PRO(0), 
                          zw.Swap(len(permutation)*2, 
                                  len(permutation)*2, 
                                  doubled_permutation), 
                          PRO(0)),
                )
        
        super().__init__(inside, dom, cod)


class Create(zw.Diagram):
    """
    Implements the Create channel.

    Example
    -------

    """
    def __init__(self, n_photons: int):
        """
        Args:
            n_legs: The number of legs.
        """
        dom = PRO(0)
        cod = PRO(2)
        self.n_photons = n_photons
        inside = (
                    Layer(PRO(0), zw.Create(n_photons), PRO(0)),
                    Layer(PRO(1), zw.Create(n_photons), PRO(0)),
                )

        super().__init__(inside, dom, cod)

class Select(zw.Diagram):
    """
    Implements the Select channel.

    Example
    -------

    """
    def __init__(self, n_photons: int):
        """
        Args:
            n_photons: The number of photons.
            n_modes: The number of modes.
        """
        dom = PRO(2)
        cod = PRO(0)
        self.n_photons = n_photons
        inside = (
                    Layer(PRO(1), zw.Select(n_photons), PRO(0)),
                    Layer(PRO(0), zw.Select(n_photons), PRO(0)),
                )

        super().__init__(inside, dom, cod)

class Discard(zw.Diagram):
    """
    Implements the Discard channel.

    Example
    -------

    """
    def __init__(self):
        """
        Args:
            n_photons: The number of photons.
        """
        dom = PRO(2)
        cod = PRO(0)
        inside = (
                    Layer(PRO(0), zw.Z(lambda i: 1, 2, 0), PRO(0)),
                )

        super().__init__(inside, dom, cod)


class Measure(zw.Diagram):
    """
    Implements the Measure channel.

    Example
    -------

    """
    def __init__(self):
        """
        Args:
            n_photons: The number of photons.
        """
        dom = PRO(2)
        cod = PRO(1)
        inside = (
                    Layer(PRO(0), zw.Z(lambda i: 1, 2, 1), PRO(0)),
                )

        super().__init__(inside, dom, cod)


class Prepare(zw.Diagram):
    """
    Implements the Prepare channel.
    """

    def __init__(self):
        """
        Args:
            n_photons: The number of photons.
        """
        dom = PRO(1)
        cod = PRO(2)
        inside = (
                    Layer(PRO(0), zw.Z(lambda i: 1, 1, 2), PRO(0)),
                )

        super().__init__(inside, dom, cod)  