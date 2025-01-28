"""

Overview
--------

The category :class:`Matrix` and the syntax :class:`Diagram`
of matrices with creations and post-selections. The module
supports representing the :class:`lo` fragment
of Optyx diagrams as matrices. It enables the
computation of the amplitudes and probabilities
of the diagrams by evaluting
permanents of underlying matrices (either directly
or via Perceval [FGL+23]_). The :code:`to_path` method
of Optyx diagrams which belong to the W fragment of the
:class:`zw` calculus (including :class:`lo` circuits,
n-photon states and effects)
returns a :class:`Matrix` object.

Classes
-------------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Matrix
    Amplitudes
    Probabilities


Functions
-------------

    .. autosummary::
        :template: function.rst
        :nosignatures:
        :toctree:

        npperm

Examples of usage
------------------

**Hong-Ou-Mandel effect**

We can check the Hong-Ou-Mandel effect by
evaluating the permanent of the underlying matrix:

>>> from optyx.zw import Create, Select, Split, Merge, Id
>>> from optyx.lo import BS
>>> HOM = Create(1, 1) >> BS
>>> assert np.allclose(HOM.to_path().eval().array,\\
... Amplitudes([0.+0.70710678j, -0.+0.j    , 0.+0.70710678j],\\
... dom=1, cod=3).array)

>>> assert(HOM.to_path().prob().array, \\
... Probabilities[complex]([0.5+0.j, 0. +0.j, 0.5+0.j], \\
... dom=1, cod=3))
>>> left = Create(1, 1) >> BS >> Select(2, 0)
>>> left.to_path().prob()
Probabilities[complex]([0.5+0.j], dom=1, cod=1)

We can also show the Hong-Ou-Mandel effect by
using the rules of the :class:`path` calculus:

>>> from optyx.zw import W, Endo, SWAP, Create, Select
>>> left_hs = (Create(1, 1) >> \\
... W(2) @ W(2) >> \\
... Endo(1j) @ Id(1) @ Id(1) @ Endo(1j) >> \\
... Id(1) @ SWAP @ Id(1) >> \\
... W(2).dagger() @ W(2).dagger() >> \\
... Select(1, 1))
>>> left_hs.draw(path="docs/_static/left_hs.png")

.. image:: /_static/left_hs.png
    :align: center

>>> left_hs.to_path().prob_with_perceval().array[0, 0]
0j

According to the rewrite rules of the :class:`path` calculus,
this should be equal to:

>>> right_hs = ((Create(1, 1) >> \\
... Endo(1j) @ Endo(1j) >> \\
... Select(1, 1)) +
... (Create(1, 1) >> \\
... SWAP >> \\
... Select(1, 1)))
>>> right_hs.draw(path="docs/_static/right_hs.png")

.. image:: /_static/right_hs.png

Let us now evaluate this using :code:`DisCoPy` tensor:

>>> right_hs.to_tensor().eval().array
array(0)

**Bell state**

We can construct a Bell state in dual rail encoding:

>>> plus = Create() >> Split(2)
>>> state = plus >> Id(1) @ plus @ Id(1)
>>> bell = state @ state\\
...     >> Id(2) @ (BS @ BS.dagger() >> state.dagger()) @ Id(2)
>>> H, V = Select(1, 0), Select(0, 1)
>>> assert np.allclose(
...     (bell >> H @ H).to_path().eval().array,
...     (bell >> V @ V).to_path().eval().array)
>>> assert np.allclose(
...     (bell >> V @ H).to_path().eval().array,
...     (bell >> H @ V).to_path().eval().array)

**Number operator**

We can define the number operator and compute its expectation.

>>> num_op = Split(2) >> Id(1) @ Select(1) >> Id(1) @ Create(1) >> Merge(2)
>>> expectation = lambda n: Create(n) >> num_op >> Select(n)
>>> assert np.allclose(expectation(5).to_path().eval().array, np.array([5.]))

References
----------
.. [FGL+23] Heurtel, N., Fyrillas, A., Gliniasty, G., Le Bihan, R., \
    Malherbe, S., Pailhas, M., Bertasi, E., Bourdoncle, B., Emeriau, \
    P.E., Mezher, R., Music, L., Belabas, N., Valiron, B., Senellart, \
    P., Mansfield, S., & Senellart, J. (2023). Perceval: A Software \
    Platform for Discrete Variable Photonic Quantum Computing. Quantum, 7, 931.

"""

from __future__ import annotations

from math import factorial

import numpy as np
import perceval as pcvl

from discopy.cat import assert_iscomposable
from discopy.utils import unbiased
import discopy.matrix as underlying
from discopy.tensor import Tensor
from optyx.utils import occupation_numbers, amplitudes_2_tensor


def npperm(matrix):
    """
    Numpy code for computing the permanent of a matrix,
    from https://github.com/scipy/scipy/issues/7151.
    """
    n = matrix.shape[0]
    d = np.ones(n)
    j = 0
    s = 1
    f = np.arange(n)
    v = matrix.sum(axis=0)
    p = np.prod(v)
    while j < n - 1:
        v -= 2 * d[j] * matrix[j]
        d[j] = -d[j]
        s = -s
        prod = np.prod(v)
        p += s * prod
        f[0] = 0
        f[j] = f[j + 1]
        f[j + 1] = j + 1
        j = f[0]
    return p / 2 ** (n - 1)


class Matrix(underlying.Matrix):
    """
    Matrix with photon creations and post-selections,
    evaluated as :class:`Amplitudes`.

    Parameters:
        array : underlying array
        dom : int
        cod : int
        creations : list of occupation numbers
        selections : list of occupation numbers
        normalisation : normalisation factor dependent on number of photons.
        scalar : global scalar independent of number of photons

    Example
    -------
    >>> from optyx.zw import Split, Select, Create, Merge, Id
    >>> array = np.array([[1, 1], [1, 0]])
    >>> matrix = Matrix(array, 1, 1, creations=(1,), selections=(1,))
    >>> matrix.eval(3)
    Amplitudes([3.+0.j], dom=1, cod=1)
    >>> num_op = Split(2) >> Select() @ Id(1) >> Create() @ Id(1) >> Merge(2)
    >>> assert np.allclose(num_op.to_path().eval(4).array,
    ...                    matrix.eval(4).array)
    """

    dtype = complex

    def __new__(
        cls,
        array,
        dom,
        cod,
        creations=(),
        selections=(),
        normalisation=1,
        scalar=1,
    ):
        return underlying.Matrix.__new__(cls, array, dom, cod)

    def __init__(
        self,
        array,
        dom: int,
        cod: int,
        creations: tuple[int, ...] = (),
        selections: tuple[int, ...] = (),
        normalisation=1,
        scalar=1,
    ):
        self.udom, self.ucod = dom + len(creations), cod + len(selections)
        super().__init__(array, self.udom, self.ucod)
        self.dom, self.cod = dom, cod
        self.creations, self.selections = creations, selections
        self.normalisation = normalisation
        self.scalar = scalar

    @property
    def umatrix(self) -> underlying.Matrix:
        """
        Underlying matrix with `len(creations) + dom` inputs and
        `len(selections) + cod` outputs.
        """
        return underlying.Matrix[self.dtype](self.array, self.udom, self.ucod)

    @unbiased
    def then(self, other: Matrix) -> Matrix:
        """Sequential composition of QPath matrices"""
        assert_iscomposable(self, other)
        M = underlying.Matrix[self.dtype]
        left, right = len(self.selections), len(other.creations)
        umatrix = (
            self.umatrix @ right
            >> self.cod @ M.swap(left, right)
            >> other.umatrix @ left
        )
        creations = self.creations + other.creations
        selections = other.selections + self.selections
        scalar = self.scalar * other.scalar
        normalisation = self.normalisation * other.normalisation
        return Matrix[self.dtype](
            umatrix.array,
            self.dom,
            other.cod,
            creations,
            selections,
            normalisation,
            scalar,
        )

    @unbiased
    def tensor(self, other: Matrix) -> Matrix:
        """Parallel composition of QPath matrices"""
        M = underlying.Matrix[self.dtype]
        a, b = len(self.creations), len(other.creations)
        c, d = len(self.selections), len(other.selections)
        umatrix = (
            self.dom @ M.swap(other.dom, a) @ b
            >> self.umatrix @ other.umatrix
            >> self.cod @ M.swap(c, other.cod) @ d
        )
        dom, cod = self.dom + other.dom, self.cod + other.cod
        creations = self.creations + other.creations
        selections = self.selections + other.selections
        normalisation = self.normalisation * other.normalisation
        scalar = self.scalar * other.scalar
        return Matrix[self.dtype](
            umatrix.array,
            dom,
            cod,
            creations,
            selections,
            normalisation,
            scalar,
        )

    def dagger(self) -> Matrix:
        """Adjoint QPath matrix"""
        array = self.umatrix.dagger().array
        return Matrix[self.dtype](
            array,
            self.cod,
            self.dom,
            self.selections,
            self.creations,
            self.normalisation.conjugate(),
            self.scalar,
        )

    def __repr__(self):
        return (
            super().__repr__()[:-1]
            + f", creations={self.creations}"
            + f", selections={self.selections}"
            + f", normalisation={self.normalisation}"
            + f", scalar={self.scalar})"
        )

    def dilate(self) -> Matrix:
        """
        Returns an equivalent :class:`Matrix` with unitary underlying matrix.

        Example
        -------
        >>> from optyx.zw import Split, Select, Create, Merge, Id
        >>> num_op = Split(2) >> Select() @ Id(1) \\
        ...           >> Create() @ Id(1) >> Merge(2)
        >>> U = num_op.to_path().dilate()
        >>> assert np.allclose(
        ...     (U.umatrix >> U.umatrix.dagger()).array, np.eye(4))
        >>> assert np.allclose(U.eval(5).array, num_op.to_path().eval(5).array)
        """
        dom, cod = self.umatrix.dom, self.umatrix.cod
        A = self.umatrix.array
        U, S, Vh = np.linalg.svd(A)
        s = max(S) if max(S) > 1 else 1
        defect0 = np.concatenate(
            [np.sqrt(1 - (S / s) ** 2), [1 for _ in range(dom - len(S))]]
        )
        defect1 = np.concatenate(
            [np.sqrt(1 - (S / s) ** 2), [1 for _ in range(cod - len(S))]]
        )
        defect_left = U.dot(np.diag(defect0)).dot(U.conj().T)
        defect_right = (Vh.conj().T).dot(np.diag(defect1)).dot(Vh)
        unitary = np.block(
            [[A / s, defect_left], [defect_right, -A.conj().T / s]]
        )
        creations = self.creations + cod * (0,)
        selections = self.selections + dom * (0,)

        return Matrix(
            unitary,
            self.dom,
            self.cod,
            creations,
            selections,
            normalisation=s,
        )

    def eval(
        self, n_photons=0, permanent=npperm, as_tensor=False
    ) -> Amplitudes:
        """Evaluates the :class:`Amplitudes` of a the QPath matrix"""
        dom_basis = occupation_numbers(n_photons, self.dom)
        n_photons_out = n_photons - sum(self.selections) + sum(self.creations)
        if n_photons_out < 0:
            raise ValueError("Expected a positive number of photons out.")
        cod_basis = occupation_numbers(n_photons_out, self.cod)

        result = Amplitudes[self.dtype].zero(len(dom_basis), len(cod_basis))
        normalisation = self.normalisation ** (n_photons + sum(self.creations))
        for i, open_creations in enumerate(dom_basis):
            for j, open_selections in enumerate(cod_basis):
                creations = open_creations + self.creations
                selections = open_selections + self.selections
                matrix = np.stack(
                    [
                        self.array[:, m]
                        for m, n in enumerate(selections)
                        for _ in range(n)
                    ],
                    axis=1,
                )
                matrix = np.stack(
                    [
                        matrix[m]
                        for m, n in enumerate(creations)
                        for _ in range(n)
                    ],
                    axis=0,
                )
                divisor = np.sqrt(
                    np.prod([factorial(n) for n in creations + selections])
                )
                val = self.scalar * normalisation * permanent(matrix) / divisor
                result.array[i, j] = val
        if as_tensor:
            return amplitudes_2_tensor(result.array, dom_basis, cod_basis)
        return result

    def prob(
        self,
        n_photons=0,
        permanent=npperm,
        with_perceval=False,
        as_tensor=False,
    ) -> Probabilities:
        """Computes the Born rule of the amplitudes of the :class:`Matrix`"""
        if with_perceval:
            return self.prob_with_perceval(n_photons, as_tensor=as_tensor)
        amplitudes = self.eval(n_photons, permanent, as_tensor)
        probabilities = np.abs(amplitudes.array) ** 2
        if as_tensor:
            return Tensor(probabilities, amplitudes.dom, amplitudes.cod)
        return Probabilities[self.dtype](
            probabilities, amplitudes.dom, amplitudes.cod
        )

    def prob_with_perceval(
        self, n_photons=0, simulator: str = "SLOS", as_tensor=False
    ) -> Probabilities:
        """
        Computes the Born rule of the amplitudes of the :class:`Matrix` using
        the perceval library

        Note
        ----
        If the :class:`Matrix` is non-unitary, first :meth:`dilate` is called
        to create a unitary.

        Example
        -------
        >>> import numpy as np
        >>> from optyx.zw import Split, Select, Create, Merge, Id, Endo, SWAP
        >>> theta, phi = np.pi / 4, 0
        >>> r = np.exp(1j * phi) * np.sin(theta)
        >>> t = np.cos(theta)
        >>> optyx_bs = Split(2) @ Split(2) >> Id(1) @ SWAP @ Id(1) \\
        ...            >> Endo(r) @ Endo(t) @ Endo(np.conj(t)) \\
        ...            @ Endo(-np.conj(r)) >> Merge(2) @ Merge(2)
        >>> assert optyx_bs.to_path().prob_with_perceval\\
        ...              (n_photons=1).round(1) == Probabilities[complex](
        ...         [0.5+0.j, 0.5+0.j, 0.5+0.j, 0.5+0.j], dom=2, cod=2)
        >>> z_spider = optyx_bs >> Endo(2) @ Id(1) >> optyx_bs
        >>> assert z_spider.to_path().prob_with_perceval\\
        ...               (n_photons=1).round(1)== Probabilities[complex](
        ...         [0.9+0.j, 0.1+0.j, 0.1+0.j, 0.9+0.j], dom=2, cod=2)
        """
        if not self._umatrix_is_unitary():
            self = self.dilate()

        circ = self._umatrix_to_perceval_circuit()
        post = self._to_perceval_post_select()

        proc = pcvl.Processor(simulator)
        proc.set_circuit(circ)
        proc.set_postselection(post)

        input_occ = occupation_numbers(n_photons, self.dom)
        output_occ = occupation_numbers(
            sum(self.creations) + n_photons,
            len(self.creations) + self.dom,
        )

        states = [pcvl.BasicState(o + self.creations) for o in input_occ]
        analyzer = pcvl.algorithm.Analyzer(proc, states, "*")

        permutation = [
            analyzer.col(pcvl.BasicState(o))
            for o in output_occ
            if post(pcvl.BasicState(o))
        ]
        result = analyzer.distribution[:, permutation]
        if as_tensor:
            return amplitudes_2_tensor(result, input_occ, output_occ)
        return Probabilities[self.dtype](
            result,
            dom=len(states),
            cod=len(permutation),
        )

    def _umatrix_to_perceval_circuit(self) -> pcvl.Circuit:
        _mzi_triangle = (
            pcvl.Circuit(2)
            // pcvl.BS()
            // (0, pcvl.PS(phi=pcvl.Parameter("phi_1")))
            // pcvl.BS()
            // (0, pcvl.PS(phi=pcvl.Parameter("phi_2")))
        )

        m = pcvl.MatrixN(self.array.T.conj())
        return pcvl.Circuit.decomposition(
            m,
            _mzi_triangle,
            phase_shifter_fn=pcvl.PS,
            shape="triangle",
            max_try=1,
        )

    def _to_perceval_post_select(self) -> pcvl.PostSelect:
        post_str = [
            f"[{self.cod + i}] == {p}" for i, p in enumerate(self.selections)
        ]
        return pcvl.PostSelect(" & ".join(post_str))

    def _umatrix_is_unitary(self) -> bool:
        m = self.umatrix.array
        return np.allclose(np.eye(m.shape[0]), m.dot(m.conj().T))


class Amplitudes(underlying.Matrix):
    """
    Operator on the Fock space represented as matrix over `occupation_numbers`.

    Example
    -------
    >>> from optyx.zw import Select, Id
    >>> from optyx.lo import BS
    >>> BS.to_path().eval(1)
    Amplitudes([0.    +0.70710678j, 0.70710678+0.j    , 0.70710678+0.j    ,
     0.    +0.70710678j], dom=2, cod=2)
    >>> assert isinstance(BS.to_path().eval(2), Amplitudes)
    >>> assert np.allclose((BS >> Select(1) @ \\
    ... Id(1)).to_path().eval(2).array,\\
    ... Amplitudes([0.+0.70710678j, -0.+0.j    , 0.+0.70710678j], \\
    ... dom=3, cod=1).array)
    """

    dtype = complex

    def __new__(cls, array, dom, cod):
        return underlying.Matrix.__new__(cls, array, dom, cod)


class Probabilities(underlying.Matrix):
    """
    Stochastic matrix of probabilities over `occupation_numbers`.

    Example
    -------
    >>> from optyx.zw import Create
    >>> from optyx.lo import BS
    >>> BS.to_path().prob(1).round(1)
    Probabilities[complex]([0.5+0.j, 0.5+0.j, 0.5+0.j, 0.5+0.j], dom=2, cod=2)
    >>> (Create(1, 1) >> BS).to_path().prob().round(1)
    Probabilities[complex]([0.5+0.j, 0. +0.j, 0.5+0.j], dom=1, cod=3)
    """

    dtype = float

    def __new__(cls, array, dom, cod):
        return underlying.Matrix.__new__(cls, array, dom, cod)

    def normalise(self) -> Probabilities:
        return self.__class__(
            array=self.array / self.array.sum(axis=1)[:, None],
            dom=self.dom,
            cod=self.cod,
        )
