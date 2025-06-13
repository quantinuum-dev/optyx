"""
Overview
--------

:class:`zw` diagrams [FSP+23]_ and their
mapping to :class:`tensor.Diagram` from
DisCoPy [FTC21]_. :class:`zw` enables
us to express a wider class of linear maps
on the bosonic Fock space than by simply using
the physically motivated diagarms of
:class:`lo`. This however means that some of the
maps might not be directly physically realisable.

The calculus is encompassing both the :class:`lo`
and :class:`path` calculi.

Generators and diagrams
------------------------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    W
    ZBox
    Create
    Select
    Endo
    Scalar


Examples of usage
------------------

Let's check the axioms of :class:`zw`. The examples also
showcase the option to map :class:`zw` diagrams to tensors
which can be evaluated using :code:`DisCoPy` (with :code:`tensor.eval()`)
or using :code:`quimb` (with :code:`tensor.to_quimb()`).

**W commutativity**

>>> from optyx._utils import compare_arrays_of_different_sizes
>>> from discopy.drawing import Equation
>>> bSym_l = W(2)
>>> bSym_r = W(2) >> SWAP
>>> assert compare_arrays_of_different_sizes(\\
...             (bSym_l.to_tensor().to_quimb()^...).data,\\
...             bSym_r.to_tensor().eval().array)
>>> Equation(bSym_l, bSym_r, symbol="$=$").draw(\\
... path="docs/_static/zw_commutativity.svg")

.. image:: /_static/zw_commutativity.svg
    :align: center

**W associativity**

>>> bAso_l = W(2) >> W(2) @ Id(1)
>>> bAso_r = W(2) >> Id(1) @ W(2)
>>> assert compare_arrays_of_different_sizes(\\
...             bAso_l.to_tensor().eval().array,\\
...             bAso_r.to_tensor().eval().array)
>>> Equation(bAso_l, bAso_r, symbol="$=$").draw(\\
... path="docs/_static/zw_associativity.svg")

.. image:: /_static/zw_associativity.svg
    :align: center

**W unit**

>>> bId_l = W(2) >> Select(0) @ Id(1)
>>> bId_r = W(2) >> Id(1) @ Select(0)
>>> assert compare_arrays_of_different_sizes(\\
...             bId_l.to_tensor().eval().array,\\
...             bId_r.to_tensor().eval().array)
>>> Equation(bId_l, bId_r, symbol="$=$").draw(\\
... path="docs/_static/zw_unit.svg")

.. image:: /_static/zw_unit.svg
    :align: center

**W bialgebra**

>>> bBa_l = W(2) @ W(2) >>\\
...             Id(1) @ SWAP @ Id(1) >>\\
...             W(2).dagger() @ W(2).dagger()
>>> bBa_r = W(2).dagger() >> W(2)
>>> assert compare_arrays_of_different_sizes(\\
...             bBa_l.to_tensor().eval().array,\\
...             bBa_r.to_tensor().eval().array)
>>> Equation(bBa_l, bBa_r, symbol="$=$").draw(\\
... path="docs/_static/zw_bialgebra.svg")

.. image:: /_static/zw_bialgebra.svg
    :align: center

**ZW bialgebra**

>>> from math import factorial
>>> N = [float(np.sqrt(factorial(i))) for i in range(5)]
>>> frac_N = [float(1/np.sqrt(factorial(i))) for i in range(5)]
>>> bZBA_l = ZBox(1,2,N) @ ZBox(1, 2, N) >>\\
...             Id(1) @ SWAP @ Id(1) >>\\
...             W(2).dagger() @ W(2).dagger() >>\\
...             Id(1) @ ZBox(1, 1, frac_N)
>>> bZBA_r = W(2).dagger() >> ZBox(1,2,[1, 1, 1, 1, 1])
>>> assert compare_arrays_of_different_sizes(\\
...             bZBA_l.to_tensor().eval().array,\\
...             bZBA_r.to_tensor().eval().array)
>>> Equation(bZBA_l, bZBA_r, symbol="$=$").draw(\\
... path="docs/_static/zw_zw_bialgebra.svg")

.. image:: /_static/zw_zw_bialgebra.svg
    :align: center


**Z copies n-photon states**

>>> K0_infty_l = Create(4) >> ZBox(1,2,[1, 1, 1, 1, 1])
>>> K0_infty_r = Create(4) @ Create(4)
>>> assert compare_arrays_of_different_sizes(\\
...             K0_infty_l.to_tensor().eval().array,\\
...             K0_infty_r.to_tensor().eval().array)
>>> Equation(K0_infty_l, K0_infty_r, symbol="$=$").draw(\\
... path="docs/_static/zw_z_copies.svg")

.. image:: /_static/zw_z_copies.svg
    :align: center


**Check Lemma B7 from 2306.02114**

>>> lemma_B7_l = Id(1) @ W(2).dagger() >> \\
...             ZBox(2,0,lambda i: 1)
>>> lemma_B7_r = W(2) @ Id(2) >>\\
...             Id(1) @ Id(1) @ SWAP >>\\
...             Id(1) @ SWAP @ Id(1) >>\\
...             ZBox(2,0,lambda i: 1) @ ZBox(2,0,lambda i: 1)
>>> assert compare_arrays_of_different_sizes(\\
...             lemma_B7_l.to_tensor().eval().array,\\
...             lemma_B7_r.to_tensor().eval().array)
>>> Equation(lemma_B7_l, lemma_B7_r, symbol="$=$").draw(\\
... path="docs/_static/zw_lemma_B7.svg")

.. image:: /_static/zw_lemma_B7.svg
    :align: center
"""

from typing import List, Union
import numpy as np
from discopy.frobenius import Dim
from discopy import tensor
from optyx.core import diagram
from optyx._utils import (
    occupation_numbers,
    multinomial,
    filter_occupation_numbers
)
from optyx.core.path import Matrix


class ZWDiagram(diagram.Diagram):
    pass

class ZWBox(diagram.Box, ZWDiagram):
    """Box in a :class:`Diagram`"""

    def __init__(self, name, dom, cod, **params):
        if isinstance(dom, int):
            dom = diagram.Mode(dom)
        if isinstance(cod, int):
            cod = diagram.Mode(cod)
        super().__init__(name=name, dom=dom, cod=cod, **params)


class IndexableAmplitudes:
    """Since the amplitudes can be an infinite list,
    we can specify them as a function instead of an explicit list.
    The class is a wrapper for the function which allows to
    index the function as if it was a list.

    >>> f = lambda i: i
    >>> amplitudes = IndexableAmplitudes(f)
    >>> assert amplitudes[0] == 0
    """

    def __init__(self, func, conj=False, name="Z(func)"):
        self.func, self.conj, self.name = func, conj, name

    def __getitem__(self, i):
        if not self.conj:
            return self.func(i)
        return np.conj(self.func(i))

    def __str__(self) -> str:
        return "function"

    def conjugate(self):
        """Conjugate the amplitudes"""
        return IndexableAmplitudes(
            self.func, conj=not self.conj, name=self.name
        )

    def __eq__(self, other: "IndexableAmplitudes") -> bool:
        if not isinstance(other, IndexableAmplitudes):
            return False
        return self.func.__code__.co_code == other.func.__code__.co_code


class W(ZWBox):
    """
    W node from the infinite ZW calculus - one input and n outputs
    """

    draw_as_spider = False
    color = "white"

    def __init__(self, n_legs: int, is_dagger: bool = False):
        dom = diagram.Mode(n_legs) if is_dagger else diagram.Mode(1)
        cod = diagram.Mode(1) if is_dagger else diagram.Mode(n_legs)
        super().__init__("W", dom, cod)
        self.n_legs = n_legs
        self.is_dagger = is_dagger
        self.shape = "triangle_up" if not is_dagger else "triangle_down"

    def conjugate(self):
        return self

    def truncation(
        self, input_dims: list[int] = None, output_dims: list[int] = None
    ) -> tensor.Box:
        """Create a truncated array like in 2306.02114."""
        if input_dims is None:
            raise ValueError("Input dimensions must be provided.")

        if output_dims is None:
            output_dims = self.determine_output_dimensions(input_dims)

        max_dim = output_dims[0] if self.is_dagger else input_dims[0]
        shape = (
            (np.prod(input_dims), output_dims[0])
            if self.is_dagger
            else (np.prod(output_dims), input_dims[0])
        )
        result_matrix = np.zeros(shape, dtype=complex)

        for n in range(max_dim):
            allowed_configs = occupation_numbers(n, self.n_legs)
            if self.is_dagger:
                allowed_configs = filter_occupation_numbers(
                    allowed_configs, np.array(input_dims) - 1
                )

            for config in allowed_configs:
                coef = np.sqrt(multinomial(config))

                if self.is_dagger:
                    row_idx = sum(
                        s * np.prod(input_dims[i + 1:], dtype=int)
                        for i, s in enumerate(config)
                    )
                else:
                    row_idx = sum(
                        s * np.prod(output_dims[i + 1:], dtype=int)
                        for i, s in enumerate(config)
                    )

                result_matrix[row_idx, n] += coef

        out_dims = Dim(*[int(i) for i in output_dims])
        in_dims = Dim(*[int(i) for i in input_dims])

        if self.is_dagger:
            return tensor.Box(self.name, in_dims, out_dims, result_matrix)
        return tensor.Box(
            self.name, in_dims, out_dims, result_matrix.conj().T
        )

    def determine_output_dimensions(self, input_dims: list[int]) -> list[int]:
        """Determine the output dimensions based on the input dimensions."""
        if self.is_dagger:
            dims_out = np.sum(np.array(input_dims) - 1) + 1
            return [dims_out]
        return [input_dims[0] for _ in range(len(self.cod))]

    def to_path(self, dtype=complex) -> Matrix:
        array = np.ones(self.n_legs)
        if self.is_dagger:
            return Matrix[dtype](array, self.n_legs, 1)
        return Matrix[dtype](array, 1, int(self.n_legs))

    def dagger(self) -> diagram.Diagram:
        return W(self.n_legs, not self.is_dagger)

    def __repr__(self):
        attr = ", dagger=True" if self.is_dagger else ""
        return f"W({self.n_legs}{attr})"

    def __eq__(self, other: "W") -> bool:
        if not isinstance(other, W):
            return False
        return (self.n_legs, self.is_dagger) == (
            other.n_legs,
            other.is_dagger,
        )


class ZBox(diagram.Spider, ZWBox):
    """
    Z spider from the ZW calculus.
    """

    def __init__(
        self,
        legs_in: int = 1,
        legs_out: int = 1,
        amplitudes: Union[
            np.ndarray, list, callable, IndexableAmplitudes
        ] = lambda i: i,
    ):
        # if amplitudes are a function then make it indexable, "conjugable"
        if callable(amplitudes):
            self.amplitudes = IndexableAmplitudes(amplitudes)
        else:
            self.amplitudes = amplitudes
        super().__init__(legs_in, legs_out, diagram.Mode(1))
        self.legs_in = legs_in
        self.legs_out = legs_out

    def conjugate(self):
        return ZBox(self.legs_in, self.legs_out, self.amplitudes.conjugate())

    def truncation(
        self, input_dims: list[int] = None, output_dims: list[int] = None
    ) -> tensor.Box:

        if input_dims is None:
            raise ValueError("Input dimensions must be provided.")

        # if a scalar
        if self.legs_out == 0 and self.legs_in == 0:
            amplitudes = (
                np.array([self.amplitudes], dtype=complex)
                if not isinstance(self.amplitudes, IndexableAmplitudes)
                else np.array([self.amplitudes[0]], dtype=complex)
            )
            return tensor.Box(self.name, Dim(1), Dim(1), amplitudes)

        spider_dim = min(input_dims) if self.legs_in > 0 else 2

        # create the array
        if not isinstance(
            self.amplitudes, IndexableAmplitudes
        ) and spider_dim > len(self.amplitudes):
            diag = list(self.amplitudes) + [0] * (
                spider_dim - len(self.amplitudes)
            )
        else:
            diag = [self.amplitudes[i] for i in range(spider_dim)]
        result_matrix = np.diag(diag)

        # get the embedding layer and the leg on which to put the Z box
        embedding_layer = tensor.Id(1)
        idx_leg_zbox = 0
        for i, input_dim in enumerate(input_dims):
            embedding_layer @= (
                diagram.EmbeddingTensor(input_dim, spider_dim)
                if input_dim > spider_dim
                else tensor.Id(Dim(int(input_dim)))
            )
            if input_dim == spider_dim:
                idx_leg_zbox = i

        # put the Zbox on the leg with min dimensions
        n_legs = self.legs_in if self.legs_in > 0 else self.legs_out

        layer_zbox = (
            Dim(*[spider_dim] * idx_leg_zbox)
            @ tensor.Box(
                self.name,
                Dim(int(spider_dim)),
                Dim(int(spider_dim)),
                result_matrix,
            )
            @ Dim(*[spider_dim] * (n_legs - idx_leg_zbox - 1))
        )

        spider_layer = tensor.Spider(
            self.legs_in, self.legs_out, Dim(int(spider_dim))
        )
        full_subdiagram = (
            embedding_layer >> layer_zbox >> spider_layer
            if self.legs_in > 0
            else spider_layer >> layer_zbox
        )

        return full_subdiagram

    def determine_output_dimensions(self, input_dims: list[int]) -> list[int]:
        """Determine the output dimensions based on the input dimensions."""
        if self.legs_in == 0:
            return [2 for _ in range(len(self.cod))]
        return [min(input_dims) for _ in range(len(self.cod))]

    def __repr__(self):
        if isinstance(self.amplitudes, IndexableAmplitudes):
            s = ", ".join(str(self.amplitudes[i]) for i in range(2)) + ", ..."
        else:
            s = ", ".join(str(a) for a in self.amplitudes)
        return f"Z({s})"

    __str__ = __repr__

    def __eq__(self, other: "ZBox") -> bool:
        if (
            not isinstance(other, ZBox)
            or self.legs_in != other.legs_in
            or self.legs_out != other.legs_out
        ):
            return False
        if isinstance(self.amplitudes, IndexableAmplitudes):
            return self.amplitudes == other.amplitudes
        return np.allclose(self.amplitudes, other.amplitudes)

    def dagger(self) -> diagram.Diagram:
        return ZBox(self.legs_out, self.legs_in, np.conj(self.amplitudes))


class Create(ZWBox):
    """
    Creation of photons on modes given a list of occupation numbers.

    Parameters:
        photons : Occupation numbers.

    Example
    -------
    >>> assert Create() == Create(1)
    >>> Create(1).to_path().eval()
    Amplitudes([1.+0.j], dom=1, cod=1)
    """

    draw_as_spider = True
    color = "blue"

    def __init__(self,
                 *photons: int,
                 internal_states: tuple[list[int]] = None):
        self.photons = photons or (1,)

        if internal_states is not None:
            # we define an internal state for each photon
            if not isinstance(internal_states, tuple):
                internal_states = (internal_states,)
            assert all(p in (0, 1) for p in photons), \
                "Only 0 or 1 photons per mode are allowed for internal states"
            assert sum(photons) == len(internal_states), \
                "The number of internal states must " \
                "match the total number of photons"
            assert len(set(len(i) for i in internal_states)) == 1, \
                "All internal states must be of the same length"

        self.internal_states = internal_states

        name = "Create(1)" if self.photons == (1,) else f"Create({photons})"
        super().__init__(name, 0, len(self.photons))

    def conjugate(self):
        return self

    def to_path(self, dtype=complex):
        array = np.eye(len(self.photons))
        return Matrix[dtype](
            array, 0, len(self.photons), creations=self.photons
        )

    def truncation(
        self, input_dims: list[int] = None, output_dims: list[int] = None
    ) -> tensor.Box:
        """Create an array like in 2306.02114"""

        if output_dims is None:
            output_dims = self.determine_output_dimensions()

        index = 0
        factor = 1
        for max_dim, occ_num in zip(
            reversed(output_dims), reversed(self.photons)
        ):
            index += occ_num * factor
            factor *= max_dim

        # Create the composite state vector with a 1 at the calculated index
        result_matrix = np.zeros((np.prod(output_dims), 1))
        result_matrix[index, 0] = 1

        out_dims = Dim(*[int(i) for i in output_dims])
        in_dims = Dim(1)

        return tensor.Box(self.name, in_dims, out_dims, result_matrix)

    def determine_output_dimensions(
        self, input_dims: list[int] = None
    ) -> list[int]:
        """Determine the output dimensions based on the input dimensions."""
        # for this class we don't need the input dimensions
        return [
            self.photons[i] + 1 if self.photons[i] != 0 else 2
            for i in range(len(self.cod))
        ]

    def dagger(self) -> diagram.Diagram:
        return Select(*self.photons)

    def inflate(self, d):

        if any(p == 1 for p in self.photons):
            assert self.internal_states is not None, \
                "Internal states in Create/Select must be " \
                "provided if there is at least one photon being created"
            assert all(len(state) == d for state in self.internal_states), \
                "All internal states must be of length d"

        dgrm = Id(diagram.Mode(0))

        photon_index = 0
        for i, n_photons in enumerate(self.photons):

            endo_layer = Id(diagram.Mode(0))
            if n_photons == 1:
                for j in self.internal_states[photon_index]:
                    endo_layer @= Endo(j)
                photon_index += 1
            else:
                endo_layer @= Id(diagram.Mode(d))

            dgrm @= (
                Create(self.photons[i]) >>
                W(d) >>
                endo_layer
            )

        return dgrm

class Select(ZWBox):
    """
    Post-selection of photons given a list of occupation numbers.

    Parameters:
        photons : Occupation numbers.

    Example
    -------
    >>> assert Select() == Select(1)
    >>> assert Select(2).dagger() == Create(2)
    """

    draw_as_spider = True
    color = "blue"

    def __init__(self,
                 *photons: int,
                 internal_states: tuple[list[int]] = None):

        if internal_states is not None:
            # we define an internal state for each photon
            if not isinstance(internal_states, tuple):
                internal_states = (internal_states,)
            assert all(p in (0, 1) for p in photons), \
                "Only 0 or 1 photons per mode are allowed for internal states"
            assert sum(photons) == len(internal_states), \
                "The number of internal states must " \
                "match the total number of photons"
            assert len(set(len(i) for i in internal_states)) == 1, \
                "All internal states must be of the same length"

        self.internal_states = internal_states
        self.photons = photons or (1,)
        name = "Select(1)" if self.photons == (1,) else f"Select({photons})"
        super().__init__(name, len(self.photons), 0)

    def inflate(self, d):

        dgrm = Create(
            *self.photons,
            internal_states=self.internal_states
        ).inflate(d).dagger()

        return dgrm

    def conjugate(self):
        return self

    def to_path(self, dtype=complex) -> Matrix:
        array = np.eye(len(self.photons))
        return Matrix[dtype](
            array, len(self.photons), 0, selections=self.photons
        )

    def truncation(
        self, input_dims: list[int] = None, output_dims: list[int] = None
    ) -> tensor.Box:
        """Create an array like in 2306.02114"""

        if input_dims is None:
            raise ValueError("Input dimensions must be provided.")

        result_matrix = np.zeros((1, np.prod(input_dims)), dtype=complex)
        index = 0
        factor = 1
        for max_dim, occ_num in zip(
            reversed(input_dims), reversed(self.photons)
        ):
            index += occ_num * factor
            factor *= max_dim

        out_dims = Dim(1)
        in_dims = Dim(*[int(i) for i in input_dims])

        # if the occupation number on which we
        # are postselecting is large than the
        # maximum dimension of the input, then we
        # return the zero matrix because
        # the inner product is zero anyway
        if index < np.prod(input_dims):
            result_matrix[0, index] = 1.0
            return tensor.Box(self.name, in_dims, out_dims, result_matrix)

        return tensor.Box(self.name, in_dims, out_dims, result_matrix)

    def determine_output_dimensions(self, _=None) -> list[int]:
        """Determine the output dimensions based on the input dimensions."""
        return []

    def dagger(self) -> diagram.Diagram:
        return Create(*self.photons)


class Endo(ZWBox):
    """
    Endomorphism with one input and one output.

    Parameters:
        scalar : complex

    Example
    -------
    >>> assert (Create(2) >> Split(2) >> Id(1) @ Endo(0.5)).to_path()\\
    ...     == Matrix(
    ...         [1. +0.j, 0.5+0.j], dom=0, cod=2,
    ...         creations=(2,), selections=(), normalisation=1)
    >>> from sympy import Expr
    >>> from sympy.abc import psi
    >>> import sympy as sp
    >>> assert Endo(3 * psi ** 2).to_path(Expr)\\
    ...     == Matrix[Expr]([3*psi**2], dom=1, cod=1)
    >>> phase = Endo(sp.exp(1j * psi * 2 * sp.pi))
    >>> derivative = phase.grad(psi).subs((psi, 0.5)).to_path().eval(2).array
    >>> assert np.allclose(derivative, 4 * np.pi * 1j)
    """

    def __init__(self, scalar: complex):
        try:
            scalar = complex(scalar)
        except TypeError:
            pass
        self.scalar = scalar
        super().__init__(f"Endo({scalar})", 1, 1, data=scalar)

    def conjugate(self):
        return Endo(self.scalar.conjugate())

    def to_path(self, dtype=complex) -> Matrix:
        """Returns an equivalent :class:`Matrix` object"""
        return Matrix[dtype]([self.scalar], 1, 1)

    def dagger(self) -> diagram.Diagram:
        return Endo(self.scalar.conjugate())

    def grad(self, var):
        """Compute the gradient of the scalar with respect to a variable."""
        if var not in self.free_symbols:
            return self.sum_factory((), self.dom, self.cod)
        s = self.scalar.diff(var) / self.scalar
        num_op = Split(2) >> Id(1) @ (Select() >> Create()) >> Merge(2)
        d = diagram.Scalar(s) @ (self >> num_op)
        return d

    def lambdify(self, *symbols, **kwargs):
        from sympy import lambdify

        return lambda *xs: type(self)(
            lambdify(symbols, self.scalar, **kwargs)(*xs)
        )

    def truncation(
        self, input_dims: list[int], output_dims: list[int] = None
    ) -> tensor.Box:
        return ZBox(1, 1, lambda x: self.scalar**x).truncation(
            input_dims, output_dims
        )

    def determine_output_dimensions(self, input_dims: list[int]) -> list[int]:
        return input_dims


class Add(ZWBox):
    """
    Adds multiple classical values using a W-dagger operation.
    Acts by adding the basis vectors without the binomial coefficient.
    Takes `n` mode inputs and returns a single summed mode output
    (or vice versa if daggered).

    Example
    -------
    >>> add_box = Add(2)
    >>> tensor = add_box.to_tensor(input_dims=[2, 2]).eval().array
    >>> import numpy as np
    >>> # Expect 4 one-hot outputs for all input combinations
    >>> assert np.allclose(tensor.sum(), 4)
    """

    def __init__(self, n: int, is_dagger: bool = False):
        dom = diagram.Mode(1) if is_dagger else diagram.Mode(n)
        cod = diagram.Mode(n) if is_dagger else diagram.Mode(1)

        super().__init__("Add", dom, cod)
        self.n = n
        self.is_dagger = is_dagger

    def truncation(
        self, input_dims: List[int], output_dims: List[int]
    ) -> tensor.Box:

        input_dims = [int(i) for i in input_dims]
        output_dims = [int(i) for i in output_dims]

        if self.is_dagger:
            input_dims, output_dims = output_dims, input_dims

        diag = W(self.n).dagger().to_tensor(input_dims)
        array = np.sign(
            (diag >> diagram.truncation_tensor(diag.cod.inside, output_dims))
            .eval()
            .array
        )
        if self.is_dagger:
            return tensor.Box(
                "Add", Dim(*input_dims), Dim(*output_dims), array
            ).dagger()

        return tensor.Box("Add", Dim(*input_dims), Dim(*output_dims), array)

    def determine_output_dimensions(self, input_dims: List[int]) -> List[int]:
        if self.is_dagger:
            return [int(input_dims[0])] * self.n
        return [int(sum(input_dims))]

    def dagger(self):
        return Add(self.n, not self.is_dagger)

    def conjugate(self):
        return self


class Multiply(ZWBox):
    """
    Multiplies two classical integers.

    Example
    -------
    >>> mbox = Multiply()
    >>> result = mbox.to_tensor(input_dims=[3, 3]).eval().array
    >>> import numpy as np
    >>> assert result.shape == (3, 3, 9)
    >>> nonzero = np.nonzero(result)
    >>> assert len(nonzero[0]) > 0
    """

    def __init__(self, is_dagger: bool = False):
        dom = diagram.Mode(1) if is_dagger else diagram.Mode(2)
        cod = diagram.Mode(2) if is_dagger else diagram.Mode(1)

        super().__init__("Multiply", dom, cod)

        self.is_dagger = is_dagger

    def truncation(
        self, input_dims: List[int], output_dims: List[int]
    ) -> tensor.Box:

        if self.is_dagger:
            input_dims, output_dims = output_dims, input_dims

        array = np.zeros((*input_dims, *output_dims), dtype=complex)

        for i in range(input_dims[0]):
            if i > 0:
                def multiply_diagram(n): return (diagram.Spider(1, n, diagram.Mode(1)) >>
                                                 Add(n))
            else:
                def multiply_diagram(n): return (diagram.Spider(1, 0, diagram.Mode(1)) >>
                                                 Create(0))

            d = multiply_diagram(i).to_tensor([input_dims[1]])
            d = d >> diagram.truncation_tensor(d.cod.inside, output_dims)

            array[i, :] = d.eval().array.reshape(array[i, :].shape)

        if self.is_dagger:
            return tensor.Box(
                self.name, Dim(*input_dims), Dim(*output_dims), array
            ).dagger()
        return tensor.Box(
            self.name, Dim(*input_dims), Dim(*output_dims), array
        )

    def determine_output_dimensions(self, input_dims: List[int]) -> List[int]:
        if self.is_dagger:
            return [int(input_dims[0])]
        return [int(np.prod(input_dims))]

    def conjugate(self):
        return self

    def dagger(self):
        return Multiply(not self.is_dagger)


class Divide(ZWBox):
    """
    Inverse of multiplication: decomposes a product into factors if possible.

    Example
    -------
    >>> dbox = Divide()
    >>> result = dbox.to_tensor(input_dims=[3, 3]).eval().array
    >>> import numpy as np
    >>> assert result.shape == (3, 3, 9)
    >>> assert np.all(result >= 0)
    """

    def __init__(self, is_dagger: bool = False):
        dom = diagram.Mode(1) if is_dagger else diagram.Mode(2)
        cod = diagram.Mode(2) if is_dagger else diagram.Mode(1)

        super().__init__("Divide", dom, cod)

        self.is_dagger = is_dagger

    def truncation(
        self, input_dims: List[int], output_dims: List[int]
    ) -> tensor.Box:

        if self.is_dagger:
            input_dims, output_dims = output_dims, input_dims

        array = np.zeros((*input_dims, *output_dims), dtype=complex)

        for i in range(input_dims[1]):
            if i > 0:
                def divide_diagram(n): return (diagram.Spider(1, n, diagram.Mode(1)) >>
                                               Add(n)).dagger()

                d = divide_diagram(i).to_tensor([input_dims[0]])
                d = d >> diagram.truncation_tensor(d.cod.inside, output_dims)

                array[:, i, :] = d.eval().array.reshape(array[:, i, :].shape)

        if self.is_dagger:
            return tensor.Box(
                self.name, Dim(*input_dims), Dim(*output_dims), array
            ).dagger()
        return tensor.Box(
            self.name, Dim(*input_dims), Dim(*output_dims), array
        )

    def determine_output_dimensions(self, input_dims: List[int]) -> List[int]:
        if self.is_dagger:
            return [int(input_dims[0])]
        return [int(np.prod(input_dims))]

    def conjugate(self):
        return self

    def dagger(self):
        return Divide(not self.is_dagger)


class Mod2(ZWBox):
    """
    Reduces a classical mode to its parity (even/odd), i.e., modulo 2.

    Example
    -------
    >>> m2 = Mod2()
    >>> array = m2.to_tensor(input_dims=[5]).eval().array
    >>> import numpy as np
    >>> assert np.allclose([np.argmax(array[i]) for i in range(5)],
    ...    [i % 2 for i in range(5)])
    """

    def __init__(self, is_dagger: bool = False):
        super().__init__("Mod2", diagram.Mode(1), diagram.Mode(1))
        self.is_dagger = is_dagger

    def truncation(
        self, input_dims: List[int], output_dims: List[int]
    ) -> tensor.Box:

        if self.is_dagger:
            input_dims, output_dims = output_dims, input_dims

        array = np.zeros((*input_dims, *output_dims), dtype=complex)

        for i in range(input_dims[0]):
            array[i, i % 2] = 1

        if self.is_dagger:
            return tensor.Box(
                self.name, Dim(*input_dims), Dim(*output_dims), array
            ).dagger()
        return tensor.Box(
            self.name, Dim(*input_dims), Dim(*output_dims), array
        )

    def determine_output_dimensions(self, input_dims: List[int]) -> List[int]:
        if self.is_dagger:
            return [input_dims[0]]
        return [2]

    def conjugate(self):
        return self

    def dagger(self):
        return Mod2(not self.is_dagger)


SWAP = diagram.Swap(diagram.Mode(1), diagram.Mode(1))


def Split(n):
    return W(n)


def Merge(n):
    return W(n).dagger()


def Id(n):
    return diagram.Diagram.id(n) if \
          isinstance(n, diagram.Ty) else diagram.Diagram.id(diagram.Mode(n))