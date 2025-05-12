"""

Overview
--------

This module can be used to build arbitrary
linear optical unitary circuits using biased/tunable
beam splitters, phase shifters, and Mach-Zender interferometers.
The :class:`lo` generators have an underlying
matrix representation in :class:`path` [FC23]_,
which allows to evaluate the amplitudes of circuits
by computing permanents.

Generators and diagrams
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

Functions
----------

.. autosummary::
    :template: function.rst
    :nosignatures:
    :toctree:

    ansatz

Examples of usage
------------------

Let us check if a beam splitter showcases a valid Hang-Ou-Mandel effect:

>>> BS = BBS(0)
>>> diagram = Create(1, 1) >> BS
>>> assert np.isclose((diagram >> Select(0, 2)).to_path().prob().array, 0.5)
>>> diagram.draw(path='docs/_static/BS.png')

.. image:: /_static/BS.png
    :align: center

The function :code:`ansatz` generates a universal interferometer:

>>> ansatz(6, 4).draw(path='docs/_static/ansatz6_4.png')

.. image:: /_static/ansatz6_4.png
    :align: center

Each diagram of the module can be converted to a :class:`zw` diagram:

>>> from discopy.drawing import Equation
>>> BS = BBS(0)
>>> double_BS = BS.to_zw()
>>> Equation(BS, double_BS, symbol="$\\mapsto$").draw(\\
... path="docs/_static/double_BS.png")

.. image:: /_static/double_BS.png
    :align: center

**Evaluating linear optical circuits**

:class:`lo` generators correspond to physical linear
optical devices. We can use them to build photonic "chips"
to simulate quantum photonics experiments.

As an example, let us consider a beam splitter and the
Hong-Ou-Mandel effect.

First, let's create a beam splitter:

>>> BS = BBS(0)
>>> BS.draw(path='docs/_static/BS_hom.png', figsize=(2, 2))

.. image:: /_static/BS_hom.png
    :align: center

If we want to evaluate the effect of
inputting two photons using :code:`quimb`,
we need to feed the circuit with two photons.
Finally, let's check the effect of having both
photons on two output modes.

>>> diagram_qpath = Create(1, 1) >> BS >> Select(1, 1)
>>> diagram_qpath.draw(path='docs/_static/BS_hom_2.png', figsize=(3, 3))

.. image:: /_static/BS_hom_2.png
    :align: center

>>> np.round(diagram_qpath.to_zw().to_tensor().to_quimb()^..., 1)
-0.0

We can also do the same using :code:`Perceval`:

>>> diagram_qpath.to_path().prob_with_perceval().array[0, 0]
0j

We can also obtain the entire probability distribution (or amplitudes)
for all possible outcomes of an experiment. Let us use the interferometer
on three modes with three photons:

>>> circuit = Create(1, 1, 1) >> ansatz(3, 3)
>>> circuit.draw(path='docs/_static/ansatz3_3.png', figsize=(5, 5))

.. image:: /_static/ansatz3_3.png
    :align: center

>>> from optyx.utils import tensor_2_amplitudes
>>> symbs = list(circuit.free_symbols)
>>> s = [(i, 0) for i in symbs]
>>> circuit = circuit.subs(*s)
>>> assert np.allclose( \\
...  np.abs(np.round( \\
...  (circuit.to_zw().to_tensor(max_dim=4).to_quimb()^...).data, 1))**2, \\
...  circuit.to_path().prob_with_perceval(as_tensor=True).array)

**Differentiation**

We can also differentiate the expectation values of optical circuits.

>>> from sympy.abc import psi
>>> circuit = BS >> Phase(psi) @ Id(1) >> BS.dagger()
>>> state = Create(2, 0) >> circuit
>>> observable = num_op @ Id(1)
>>> expectation = state >> observable >> state.dagger()
>>> assert np.allclose(
...     expectation.subs((psi, 1/2)).to_path().eval().array, np.array([0.]))
>>> assert np.allclose(
...     expectation.subs((psi, 1/4)).to_path().eval().array, np.array([1.]))
>>> exp = expectation.grad(psi).subs((psi, 1/2))
>>> assert np.allclose(
...     sum([exp.terms[i].to_path().eval().array[0] \\
...      for i in range(len(exp.terms))]), 0.)
>>> exp = expectation.grad(psi).subs((psi, 1/4))
>>> assert np.allclose(
...     sum([exp.terms[i].to_path().eval().array[0] \\
...      for i in range(len(exp.terms))]),
...     -2*np.pi)
>>> exp = expectation.grad(psi).grad(psi).subs((psi, 1/4))
>>> assert np.allclose(
...     sum([exp.terms[i].to_path().eval().array[0] \\
...      for i in range(len(exp.terms))]),
...     np.array([0.]))

References
----------
.. [FC23] de Felice, G., & Coecke, B. (2023). Quantum Linear Optics \
    via String Diagrams. In Proceedings 19th International \
    Conference on Quantum Physics and Logic, Wolfson College, \
    Oxford, UK, 27 June - 1 July 2022 (pp. 83-100). \
        Open Publishing Association.
.. [FSP+23] de Felice, G., Shaikh, R., Poór, B., Yeh, L., Wang, Q., \
    & Coecke, B. (2023). Light-Matter Interaction in the \
    ZXW Calculus. In  Proceedings of the Twentieth \
    International Conference on Quantum Physics and Logic,  \
    Paris, France, 17-21st July 2023 (pp. 20-46). Open Publishing Association.
"""

from optyx.diagram.lo import (
    Gate as GateSingle,
    Phase as PhaseSingle,
    BBS as BBSSingle,
    TBS as TBSSingle,
    MZI as MZISingle,
    ansatz as ansatz_single,
    BS_hadamard as BS_hadamard_single,
)
from optyx.diagram.channel import Channel

import numpy as np

class Gate(Channel):
    def __init__(
        self,
        array,
        dom: int,
        cod: int,
        name: str,
        is_dagger = False
    ):
        super().__init__(
            name,
            GateSingle(array, dom, cod, is_dagger=is_dagger),
        )


    def dagger(self):
        return Gate(
            np.conjugate(self.array.T),
            len(self.cod),
            len(self.dom),
            self.name,
            is_dagger=not self.is_dagger,
        )


class Phase(Channel):
    def __init__(self, angle: float):
        super().__init__(
            f"Phase({angle})",
            PhaseSingle(angle)
        )


class BBS(Channel):
    def __init__(self, bias: float):
        super().__init__(
            f"BBS({bias})",
            BBSSingle(bias)
        )

    def dagger(self):
        return BBS(0.5 - self.bias)


class TBS(Channel):
    def __init__(self, theta: float, is_dagger=False):
        super().__init__(
            f"TBS({theta})",
            TBSSingle(theta, is_dagger=is_dagger)
        )

    def dagger(self):
        return TBS(self.theta, is_dagger=not self.is_dagger)


class MZI(Channel):
    def __init__(self, theta: float, phi: float, is_dagger=False):
        super().__init__(
            f"MZI({theta}, {phi})",
            MZISingle(theta, phi, is_dagger=is_dagger)
        )

    def dagger(self):
        return MZI(self.theta, self.phi, is_dagger=not self.is_dagger)


def ansatz(width, depth):
    return Channel(
        f"Ansatz({width}, {depth})",
        ansatz_single(width, depth)
    )


BS = BBS(0)

HadamardBS = Channel(
    "HadamardBS",
    BS_hadamard_single
)