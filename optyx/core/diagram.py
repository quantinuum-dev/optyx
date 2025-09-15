"""

Overview
--------

Optyx diagrams combine three diagrammatic calculi:

- :class:`zw` calculus: for infinite-dimensional systems (Mode type), \
with generators :class:`zw.Z`, :class:`zw.W`, creations and selections.
- :class:`lo` calculus: for linear optics (Mode type), with generators \
:class:`lo.BS` and :class:`lo.Phase`, or other .
- :class:`zx` calculus: for qubit systems (Bit type), with generators \
:class:`zx.Z` and :class:`zx.X`.

Mode and Bit types can moreover be combined using :class:`DualRail`
or other instances of :class:`diagram.core.Box`.
Note that the permanent method is only defined for a subclass
of :class:`zw` diagrams, including :class:`lo` circuits.
These are also known as QPath diagrams [FC23]_,
or matrices with creations and annihilation.
They are implemented in the :class:`path.Matrix` class,
with an interface :class:`to_perceval`
or the internal evaluation method :class:`eval`.
The DisCoPy class :class:`tensor.Diagram` is used as an
implementation of tensor networks,
with dimensions as types and tensors as boxes,
with an interface :class:`to_quimb`
or the internal evaluation method :class:`eval`.
Linear optical circuits, built from the generators of :class:`lo`,
can be evaluated as tensor networks
by first applying the method :class:`to_zw`.


Types
-------------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Mode
    Bit
    Ty

Generators and diagrams
------------------------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Diagram
    Box
    Swap
    Scalar
    DualRail

Other classes
-------------
.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    EmbeddingTensor

Functions
----------
.. autosummary::
    :template: function.rst
    :nosignatures:
    :toctree:

    dual_rail
    embedding_tensor

Examples of usage
------------------

**Creating diagrams**

We can create and draw Optyx diagrams using the syntax of
the Discopy package [FTC21]_. Sequential composition of
boxes is done using the :code:`<<` operator:

>>> from optyx.core.zw import Create, W
>>> split_photon = Create(1) >> W(2)
>>> split_photon.draw(path="docs/_static/seq_comp_example.png")

.. image:: /_static/seq_comp_example.png
    :align: center

We can also compose boxes in parallel (tensor) using the :code:`@` operator :

>>> from optyx.photonic import BS, Phase
>>> beam_splitter_phase = (BS @ Phase(0.5)).get_kraus()
>>> beam_splitter_phase.draw(path="docs/_static/parallel_comp_example.png")

.. image:: /_static/parallel_comp_example.png
    :align: center

A beam-splitter from the :class:`photonic` calculus can be
expressed using the :class:`zw` calculus:

>>> from optyx.photonic import BS
>>> beam_splitter = BS.get_kraus()
>>> beam_splitter.draw(path="docs/_static/bs_zw.png")

.. image:: /_static/bs_zw.png
    :align: center

Optyx diagrams can combine the generators from
:class:`zw` (Mode type),
:class:`photonic` (Mode type) and :class:`zx` calculi (Bit type).
We can check their equivalence as tensors.

**Branching Law**

Let's check the branching law from [FC23]_.

>>> from optyx.core.zw import Create, W
>>> from optyx.utils.utils import compare_arrays_of_different_sizes
>>> branching_l = Create(1) >> W(2)
>>> branching_r = Create(1) @ Create(0) + Create(0) @ Create(1)

>>> assert compare_arrays_of_different_sizes(\\
...     branching_l.to_tensor().eval().array,\\
...     branching_r.to_tensor().eval().array,\\
... )

**Hong-Ou-Mandel Effect**

The :code:`to_tensor` method supports evaluation of
diagrams like the Hong-Ou-Mandel effect:

>>> from optyx.core.zw import ZBox, SWAP, W, Select, Id
>>> Zb_i = ZBox(1,1,np.array([1, 1j/(np.sqrt(2))]))
>>> Zb_1 = ZBox(1,1,np.array([1, 1/(np.sqrt(2))]))
>>> beam_splitter = W(2) @ W(2) >> \\
...               Zb_i @ Zb_1 @ Zb_1 @ Zb_i >> \\
...               Id(1) @ SWAP @ Id(1) >> \\
...               W(2).dagger() @ W(2).dagger()
>>> Hong_Ou_Mandel = Create(1) @ Create(1) >> \\
...                beam_splitter >> \\
...                Select(1) @ Select(1)
>>> assert compare_arrays_of_different_sizes(\\
...             Hong_Ou_Mandel.to_tensor().eval().array,\\
...             np.array([0]))

**:class:`zw` diagrams from Bosonic Operators**

The :code:`from_bosonic_operator` method
supports creating :class:`path` diagrams:

>>> from optyx.core.zw import Split, Select, Id
>>> d1 = Diagram.from_bosonic_operator(
...     n_modes= 2,
...     operators=((0, False), (1, False), (0, True)),
...     scalar=2.1
... )

>>> annil = Split(2) >> Select(1) @ Id(Mode(1))
>>> create = annil.dagger()

>>> d2 = Scalar(2.1) @ annil @ Id(Mode(1)) >> \\
... Id(Mode(1)) @ annil >> create @ Id(Mode(1))

>>> assert d1 == d2

**Permanent evaluation for QPath diagrams**

The :code:`to_path` method supports evaluation by
calculating a permanent of an underlying matrix:

>>> from optyx.core.zw import Create, W
>>> counit_l = W(2) >> Select(0) @ Id(Mode(1))
>>> counit_r = W(2) >> Id(Mode(1)) @ Select(0)
>>> assert counit_l.to_path().eval(2) == counit_r.to_path().eval(2)

References
-----------
.. [FC23] de Felice, G., & Coecke, B. (2023). Quantum Linear Optics via \
String Diagrams. In Proceedings 19th International Conference on \
Quantum Physics and Logic, Wolfson College, Oxford, UK, \
27 June - 1 July 2022 (pp. 83-100). Open Publishing Association.
.. [KW20] Kissinger, A., & Wetering, J. (2020). PyZX: Large Scale \
Automated Diagrammatic Reasoning. In  Proceedings 16th \
International Conference on Quantum Physics and Logic, \
Chapman University, Orange, CA, USA., 10-14 June 2019 \
(pp. 229-241). Open Publishing Association.
.. [Gray18] Gray, J. (2018). quimb: A python package \
for quantum information and many-body calculations. \
Journal of Open Source Software, 3(29), 819.
.. [FGL+23] Heurtel, N., Fyrillas, A., Gliniasty, G., \
Le Bihan, R., Malherbe, S., Pailhas, M., Bertasi, E., \
Bourdoncle, B., Emeriau, P.E., Mezher, R., Music, L., \
Belabas, N., Valiron, B., Senellart, P., Mansfield, S., \
& Senellart, J. (2023). Perceval: A Software Platform \
for Discrete Variable Photonic Quantum Computing. Quantum, 7, 931.
.. [FTC21] de Felice, G., Toumi, A., & Coecke, B. (2021). \
DisCoPy: Monoidal Categories in Python. In  Proceedings Z \
of the 3rd Annual International Applied Category Theory \
Conference 2020,  Cambridge, USA, 6-10th July 2020 (pp. \
183-197). Open Publishing Association.
.. [FSP+23] de Felice, G., Shaikh, R., Poór, B., Yeh, L., \
Wang, Q., & Coecke, B. (2023). Light-Matter Interaction \
in the ZXW Calculus. In  Proceedings of the Twentieth \
International Conference on Quantum Physics and Logic, \
Paris, France, 17-21st July 2023 (pp. 20-46). \
Open Publishing Association.
.. [FPY+24] de Felice, G., Poór, B., Yeh, L., & \
Cashman, W. (2024). Fusion and flow: formal protocols to \
reliably build photonic graph states. arXiv \
preprint arXiv:2409.13541.
"""

from __future__ import annotations

import numpy as np
from sympy.core import Symbol, Mul
from discopy import (
    symmetric, frobenius, tensor, hypergraph
)
from discopy.cat import factory, rsubs
from discopy.frobenius import Dim
from discopy.quantum.gates import format_number
from optyx.utils.utils import (
    modify_io_dims_against_max_dim,
    BasisTransition,
    calculate_right_offset,
    get_max_dim_for_box
)
from typing import List, Tuple, Iterable

MAX_DIM = 10


class Ob(frobenius.Ob):
    """Basic object in an optyx Diagram: bit or mode"""


@factory
class Ty(frobenius.Ty):
    """Optical and (qu)bit types."""

    ob_factory = Ob


class Mode(Ty):
    """Optical mode interpreted as the infinite space with countable basis"""

    # pylint: disable=invalid-name
    def __init__(self, n=0):
        self.n = n
        super().__init__(*["mode" for _ in range(n)])


class Bit(Ty):
    """Qubit type interpreted as the two dimensional complext vector space"""

    # pylint: disable=invalid-name
    def __init__(self, n=0):
        self.n = n
        super().__init__(*["bit" for _ in range(n)])


@factory
class Diagram(frobenius.Diagram):
    """Optyx diagram combining :class:`zw`,
    :class:`zx` and
    :class:`lo` calculi."""

    grad = tensor.Diagram.grad

    def conjugate(self) -> Diagram:
        """Conjugates every box in the diagram"""
        return symmetric.Functor(
            ob=lambda x: x,
            ar=lambda f: f.conjugate(),
            cod=symmetric.Category(Ty, Diagram),
            dom=symmetric.Category(Ty, Diagram),
        )(self)

    def to_path(self, dtype: type = complex):
        """Returns the :class:`Matrix` normal form
        of a :class:`Diagram`.
        In other words, it is the underlying matrix
        representation of a :class:`path` and :class:`lo` diagrams."""
        # pylint: disable=import-outside-toplevel
        from optyx.core import path

        return symmetric.Functor(
            ob=len,
            ar=lambda f: f.to_path(dtype),
            cod=symmetric.Category(int, path.Matrix[dtype]),
        )(self)

    # pylint: disable=too-many-locals
    def to_tensor(
        self, input_dims: list = None
    ) -> tensor.Diagram:
        """Returns a :class:`tensor.Diagram` for evaluation"""
        from optyx.core import zw

        prev_layers: List[Tuple[int, Box]] = []

        # pylint: disable=import-outside-toplevel
        def list_to_dim(dims: np.ndarray | list) -> Dim:
            """Converts a list of dimensions to a Dim object"""
            return Dim(*[int(i) for i in dims])

        number_of_input_layer_wires = len(self.dom)

        if input_dims is None:
            layer_dims = [2 for _ in range(len(self.dom))]
        else:
            assert len(self.dom) == len(input_dims), (
                f"Input dims length {len(input_dims)} does not match number of input wires {len(self.dom)}"
            )
            layer_dims = input_dims

        if len(self.boxes) == 0 and len(self.offsets) == 0:
            return tensor.Diagram.id(list_to_dim(layer_dims))

        for i, (box, left_offset) in enumerate(zip(self.boxes, self.offsets)):
            right_offset = calculate_right_offset(
                number_of_input_layer_wires, left_offset, len(box.dom)
            )

            max_dim = get_max_dim_for_box(
                left_offset,
                box,
                right_offset,
                layer_dims,
                prev_layers
            )

            dims_in = layer_dims[left_offset:left_offset + len(box.dom)]
            dims_out, _ = modify_io_dims_against_max_dim(
                box.determine_output_dimensions(dims_in),
                None,
                max_dim
            )
            if isinstance(box, zw.LO_ELEMENTS):
                prev_layers.append((left_offset, box))
            else:
                if len(dims_in) > 0:
                    prev_layers.append(
                        (
                            left_offset,
                            zw.Select(*[int(i) for i in np.array(dims_in)-1])
                        )
                    )
                if len(dims_out) > 0:
                    prev_layers.append(
                        (
                            left_offset,
                            zw.Create(*[int(i) for i in np.array(dims_out)-1])
                        )
                    )

            left = Dim()
            if left_offset > 0:
                left = list_to_dim(layer_dims[0:left_offset])
            right = Dim()
            if left_offset + len(box.dom) < number_of_input_layer_wires:
                right = list_to_dim(
                    layer_dims[left_offset + len(box.dom): number_of_input_layer_wires]
                )

            number_of_input_layer_wires += -len(box.dom) + len(box.cod)
            cod_layer_dims = (
                layer_dims[0:left_offset]
                + dims_out
                + layer_dims[left_offset + len(box.dom):]
            )
            diagram_ = left @ box.truncation(dims_in, dims_out) @ right

            if i == 0:
                diagram = diagram_
            else:
                diagram = diagram >> diagram_

            layer_dims = cod_layer_dims

        zboxes = tensor.Id(Dim(1))

        # print("------------------")
        # for d_ in prev_layers:
        #     d_[1].draw()
        # print("------------------")

        # pylint: disable=invalid-name
        for c in diagram.cod:
            zboxes @= zw.ZBox(1, 1, lambda i: 1).truncation(
                input_dims=[int(c.inside[0])], output_dims=[int(c.inside[0])]
            )
        diagram >>= zboxes
        return diagram


    @classmethod
    def from_bosonic_operator(cls, n_modes, operators, scalar=1):
        """Create a :class:`zw` diagram from a bosonic operator."""
        # pylint: disable=import-outside-toplevel
        from optyx.core import zw

        # pylint: disable=invalid-name
        d = cls.id(Mode(n_modes))
        annil = zw.Split(2) >> zw.Select(1) @ zw.Id(1)
        create = annil.dagger()
        for idx, dagger in operators:
            if not 0 <= idx < n_modes:
                raise ValueError(f"Index {idx} out of bounds.")
            box = create if dagger else annil
            d = d >> zw.Id(idx) @ box @ zw.Id(n_modes - idx - 1)

        if scalar != 1:
            # pylint: disable=invalid-name
            d = Scalar(scalar) @ d
        return d

    def to_pyzx(self):
        # pylint: disable=import-outside-toplevel
        from optyx.core import zx

        try:
            zx_diagram = zx.ZXDiagram(
                dom=self.dom,
                cod=self.cod,
                inside=self.inside,
            )
        except TypeError:
            raise NotImplementedError(
                "Conversion to PyZX is not implemented for this diagram."
            )
        return zx_diagram.to_pyzx()

    # pylint: disable=invalid-name
    def inflate(self, d):
        r"""
        Translates from an indistinguishable setting
        to a distinguishable one. For a map on :math:`F(\mathbb{C})`,
        obtain a map on :math:`F(\mathbb{C})^{\widetilde{\otimes} d}`.
        """
        assert isinstance(d, int), "Dimension must be an integer"
        assert d > 0, "Dimension must be positive"

        # pylint: disable=invalid-name
        def ob(x):
            return Ty.tensor(
                *(o**d if o.name == "mode" else o for o in x)
            )

        return symmetric.Functor(
            ob=ob,
            ar=lambda f: f.inflate(d),
            cod=symmetric.Category(Ty, Diagram),
            dom=symmetric.Category(Ty, Diagram),
        )(self)


class Box(frobenius.Box, Diagram):
    """A box in an optyx diagram"""

    __ambiguous_inheritance__ = (frobenius.Box,)

    def __init__(self, name, dom, cod, array=None, **params):
        self._array = array
        self.photon_sum_preserving = True
        super().__init__(name, dom, cod, **params)

    @classmethod
    def get_perm(self, n, d):
        return sorted(sorted(list(range(n))), key=lambda i: i % d)

    def inflate(self, d):
        r"""
        Translates from an indistinguishable setting
        to a distinguishable one. For a map on :math:`F(\mathbb{C})`,
        obtain a map on :math:`F(\mathbb{C})^{\widetilde{\otimes} d}`.
        """

        return (
            Diagram.permutation(self.get_perm(len(self.dom)*d, d),
                                self.dom**d) >>
            self**d >>
            Diagram.permutation(self.get_perm(len(self.cod)*d, d),
                                self.cod**d).dagger()
        )

    def conjugate(self) -> Box:
        """Conjugate the box.
        Inheriting boxes should implement this method.
        Otherwise it is defined by the array."""
        if self._array is not None:
            return type(self)(
                self.name + ".dagger()",
                dom=self.cod,
                cod=self.dom,
                array=self._array.conjugate(),
            )
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support conjugation"
        )

    def dagger(self) -> Box:
        """Return the dagger of the box.
        Inheriting boxes should implement this method.
        Otherwise it is defined by the array."""
        if self._array is not None:
            return type(self)(
                self.name + ".dagger()",
                dom=self.cod,
                cod=self.dom,
                array=self._array.T.conjugate(),
            )
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support dagger"
        )

    def truncation(
        self, input_dims: list[int] = None, output_dims: list[int] = None
    ) -> tensor.Box:
        """Create a tensor in the semantics of a ZW diagram.
        Inheriting boxes should implement this method.
        Otherwise it is defined by the array."""
        if self._array is not None:
            return tensor.Box(
                self.name,
                dom=tensor.Dim(2) ** len(self.dom),
                cod=tensor.Dim(2) ** len(self.cod),
                data=self._array,
            )

        if input_dims is None:
            raise ValueError("Input dimensions must be provided.")

        if output_dims is None:
            output_dims = self.determine_output_dimensions(input_dims)

        if self.is_dagger:
            input_dims, output_dims = output_dims, input_dims

        shape = (
            *[int(i) for i in output_dims],
            *[int(i) for i in input_dims]
        )
        result_matrix = np.zeros(shape, dtype=complex)

        input_ranges = [range(int(d)) for d in input_dims]
        input_combinations = np.array(
            np.meshgrid(*input_ranges)).T.reshape(-1, len(input_dims)) \
            if input_ranges else np.array([[]])

        non_zero_indices = []
        for inp in map(tuple, input_combinations):
            for trans in self.truncation_specification(
                inp, tuple(output_dims)
            ):
                idx = tuple(trans.out + inp)
                non_zero_indices.append((idx, trans.amp))

        if non_zero_indices:
            configs, coeffs = map(np.array, zip(*non_zero_indices))
            idx = tuple(configs.T)
            result_matrix[idx] = coeffs

        out_dims = Dim(*[int(i) for i in output_dims])
        in_dims = Dim(*[int(i) for i in input_dims])

        if self.is_dagger:
            return tensor.Box(self.name, out_dims, in_dims, result_matrix)
        return tensor.Box(
            self.name, out_dims, in_dims, result_matrix
        ).dagger()

    def determine_output_dimensions(self, input_dims: list[int]) -> list[int]:
        """Determine the output dimensions based on the input dimensions.
        The generators of ZW affect the dimensions
        of the output tensor diagrams.
        Inheriting boxes should implement this method.
        Otherwise it is defined by the array."""
        if self._array is not None:
            return input_dims
        str = "does not support determine_output_dimensions"
        raise NotImplementedError(
            f"{self.__class__.__name__} {str}"
        )

    def to_path(self, dtype=complex):
        """Convert the box to a (Q)path representation.
        It can only be defined for zw boxes which are also part of
        the QPath graphical language
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support to_path"
        )

    def grad(self, var):
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support grad"
        )

    def lambdify(self, *symbols, **kwargs):
        # Non-symbolic gates can be returned directly
        return lambda *xs: self

    def subs(self, *args) -> Diagram:
        syms, exprs = zip(*args)
        return self.lambdify(*syms)(*exprs)

    @property
    def array(self):
        return self._array

    @array.setter
    def array(self, value):
        """
        A :code:`diagram.Box` can be defined through an array
        which will inform :code:`Box.truncation()`, :code:`Box.dagger()`,
        :code:`Box.conjugate()` and :code:`Box.determine_output_dimensions()`.
        The box need to have fixed dom and cod. The tensor should also have
        fixed dimensions. Usually used for zx boxes
        (Bit) with tensor with dims of 2.
        """
        self._array = value

    def truncation_specification(
        self,
        inp: Tuple[int, ...] = None,
        max_output_dims: Tuple[int, ...] = None
    ) -> Iterable[BasisTransition]:
        pass

    def __pow__(self, n):
        if n == 1:
            return self
        return self @ self ** (n - 1)


class Spider(frobenius.Spider, Box):
    """Abstract spider (dagger-SCFA)

    No amplitudes, or no phases.
    """

    draw_as_spider = True
    color = "green"

    def conjugate(self):
        return self

    def determine_output_dimensions(self, input_dims: list[int]) -> list[int]:
        if isinstance(self.cod, Bit):
            return [2] * len(self.cod)
        else:
            if len(self.dom) == 0:
                return [2 for _ in range(len(self.cod))]
            return [min(input_dims) for _ in range(len(self.cod))]

    def truncation(
        self, input_dims: list[int] = None, output_dims: list[int] = None
    ) -> tensor.Box:
        """
        Create a tensor in the semantics of a ZW/ZX diagram depending
        on the domain and codomain type.

        The truncation is defined as a tensor.Spider with
        EmbeddingTensor layers to fix the dimensions of the spider to
        the lowest dimension of the input wires.
        """
        if isinstance(self.cod, Bit) and isinstance(self.dom, Bit):
            return tensor.Spider(len(self.dom), len(self.cod), Dim(2))

        if input_dims is None:
            raise ValueError("Input dimensions must be provided.")

        spider_dim = min(input_dims) if len(self.dom) > 0 else 2

        # get the embedding layer
        embedding_layer = tensor.Id(1)
        for input_dim in input_dims:
            embedding_layer @= (
                EmbeddingTensor(input_dim, spider_dim)
                if input_dim > spider_dim
                else tensor.Id(Dim(int(input_dim)))
            )

        return embedding_layer >> tensor.Spider(
            len(self.dom), len(self.cod), Dim(int(spider_dim))
        )


class Sum(symmetric.Sum, Box):
    """
    Formal sum of optyx diagrams
    """

    __ambiguous_inheritance__ = (symmetric.Sum,)

    def conjugate(self):
        return sum(term.conjugate() for term in self.terms)

    def to_path(self, dtype: type = complex):
        return sum(term.to_path(dtype) for term in self.terms)

    def eval(self, n_photons=0, permanent=None, dtype=complex):
        """Evaluate the sum of diagrams."""
        # we need to implement the proper sums of qpath diagrams
        # this is only a temporary solution, so that the grad tests pass
        if permanent is None:
            # pylint: disable=import-outside-toplevel
            from optyx.core.path import npperm

            permanent = npperm
        return sum(
            term.to_path(dtype).eval(n_photons, permanent)
            for term in self.terms
        )

    def to_tensor(self, input_dims=None):

        terms = [t.to_tensor(input_dims) for t in self]
        cods = [list(t.cod.inside) for t in terms]

        # figure out the max dims for each idx and set it for all the terms
        max_dims = [
            max(c[i] if len(c) > 0 else 0 for c in cods)
            for i in range(len(cods[0]))
        ]

        # modify the diagrams for all the terms
        # add an embedding layer for each wire to fix the cods
        for i, term in enumerate(terms):
            embedding_layer = tensor.Id(1)
            for wire, d in enumerate(term.cod):
                embedding_layer = embedding_layer @ EmbeddingTensor(
                    d.inside[0], max_dims[wire]
                )
            terms[i] = terms[i] >> embedding_layer
            terms[i].cod = Dim(*max_dims)

        # assemble the diagram
        for i, term in enumerate(terms):
            if i == 0:
                diagram = term
            else:
                diagram += term
        return diagram

    def grad(self, var, **params):
        """Gradient with respect to :code:`var`."""
        if var not in self.free_symbols:
            return self.sum_factory((), self.dom, self.cod)
        return sum(term.grad(var, **params) for term in self.terms)


class Swap(frobenius.Swap, Box):
    """Swap in optyx diagram"""

    def conjugate(self):
        return self

    def to_path(self, dtype: type = complex):
        # pylint: disable=import-outside-toplevel
        from optyx.core.path import Matrix

        return Matrix([0, 1, 1, 0], 2, 2)

    def determine_output_dimensions(self, input_dims: list[int]) -> list[int]:
        """Determine the output dimensions based on the input dimensions."""
        return input_dims[::-1]

    def truncation(
        self, input_dims: list[int] = None, output_dims: list[int] = None
    ) -> tensor.Box:
        return tensor.Swap(Dim(int(input_dims[0])), Dim(int(input_dims[1])))


class Scalar(Box):
    """
    Scalar in a diagram

    Example
    -------
    >>> from optyx.core.path import Matrix
    >>> from optyx.core.zw import Create, Select
    >>> from optyx.photonic import BS
    >>> assert Scalar(0.45).to_path() == Matrix(
    ...     [], dom=0, cod=0,
    ...     creations=(), selections=(),
    ...     normalisation=1, scalar=0.45)
    >>> s = Scalar(- 1j * 2 ** (1/2)) @ Create(1, 1) >> \\
    ...     BS.get_kraus() >> Select(2, 0)
    >>> assert np.isclose(s.to_path().eval().array[0], 1)
    """

    def __init__(self, scalar: complex | Symbol):
        if not isinstance(scalar, (Symbol, Mul)):
            self.scalar = complex(scalar)
        else:
            self.scalar = scalar
        super().__init__(
            name="scalar", dom=Mode(0), cod=Mode(0), data=self.scalar
        )

    def conjugate(self):
        return Scalar(self.scalar.conjugate())

    def __str__(self):
        return f"scalar({format_number(self.data)})"

    def to_path(self, dtype: type = complex):
        # pylint: disable=import-outside-toplevel
        from optyx.core.path import Matrix

        return Matrix[dtype]([], 0, 0, scalar=self.scalar)

    def dagger(self) -> Diagram:
        return Scalar(self.scalar.conjugate())

    def subs(self, *args):
        data = rsubs(self.scalar, *args)
        return Scalar(data)

    def grad(self, var, **params):
        """Gradient with respect to :code:`var`."""
        if var not in self.free_symbols:
            return Sum((), self.dom, self.cod)
        return Scalar(self.scalar.diff(var))

    def lambdify(self, *symbols, **kwargs):
        # pylint: disable=import-outside-toplevel
        from sympy import lambdify

        return lambda *xs: type(self)(
            lambdify(symbols, self.scalar, **kwargs)(*xs)
        )

    def truncation(
        self, input_dims: list[int] = None, output_dims: list[int] = None
    ) -> tensor.Box:
        return tensor.Box(self.name, Dim(1), Dim(1), [self.scalar])

    def determine_output_dimensions(
        self, input_dims: list[int] = None
    ) -> list[int]:
        """Determine the output dimensions"""
        return []


class DualRail(Box):
    """
    A map from :code:`Bit` to :code:`Mode` using the dual rail encoding.
    """

    def __init__(self,
                 is_dagger=False,
                 internal_state=None):
        dom = Mode(2) if is_dagger else Bit(1)
        cod = Bit(1) if is_dagger else Mode(2)
        super().__init__("2R", dom, cod)
        self.internal_state = internal_state
        self.is_dagger = is_dagger
        self.photon_sum_preserving = False

    def conjugate(self):
        return self

    def truncation_specification(
        self,
        inp: Tuple[int, ...] = None,
        max_output_dims: Tuple[int, ...] = None
    ) -> Iterable[BasisTransition]:
        out = (1, 0) if inp[0] == 0 else (0, 1)
        yield BasisTransition(
            out=out,
            amp=1.0
        )

    def determine_output_dimensions(self,
                                    input_dims: list[int]) -> list[int]:
        """Determine the output dimensions"""
        if self.is_dagger:
            return [2]
        return [2, 2]

    def inflate(self, d):
        # pylint: disable=import-outside-toplevel
        from optyx.core.zw import W, Endo

        assert self.internal_state is not None, \
            "Internal state must be provided"
        assert len(self.internal_state) == d, \
            "Internal state must be of len d"

        dgrm = DualRail() >> Diagram.tensor(
            *[
                (W(d) >>
                 Diagram.tensor(*[Endo(d_i) for d_i in self.internal_state]))
                for _ in range(2)
            ]
        )
        if self.is_dagger:
            return dgrm.dagger()
        return dgrm

    def dagger(self) -> Diagram:
        return DualRail(not self.is_dagger,
                        internal_state=self.internal_state)


class PhotonThresholdDetector(Box):
    """
    Photon-number non-resolving detector from mode to bit.
    Detects whether one or more photons are present.
    """

    def __init__(self, is_dagger=False):
        if is_dagger:
            super().__init__("PTD dagger", Bit(1), Mode(1))
        else:
            super().__init__("PTD", Mode(1), Bit(1))
        self.is_dagger = is_dagger

    def truncation_specification(
        self,
        inp: Tuple[int, ...] = None,
        max_output_dims: Tuple[int, ...] = None
    ) -> Iterable[BasisTransition]:
        yield BasisTransition(
            out=(0,) if inp[0] == 0 else (1,),
            amp=1.0
        )

    def determine_output_dimensions(self, input_dims):
        if self.is_dagger:
            return [MAX_DIM] * len(input_dims)
        return [2] * len(input_dims)

    def conjugate(self):
        return self

    def dagger(self):
        return PhotonThresholdDetector(not self.is_dagger)

    def inflate(self, d):
        # pylint: disable=import-outside-toplevel
        from optyx.core.zw import Add
        dgrm = Add(d) >> PhotonThresholdDetector()
        if self.is_dagger:
            return dgrm.dagger()
        return dgrm


class EmbeddingTensor(tensor.Box):
    """
    Embedding tensor for fixing the dimensions of the output tensor.
    """

    def __init__(self, input_dim: int, output_dim: int):

        embedding_array = np.zeros((output_dim, input_dim), dtype=complex)

        if input_dim < output_dim:
            embedding_array[:input_dim, :input_dim] = np.eye(input_dim)
        else:
            embedding_array[:output_dim, :output_dim] = np.eye(output_dim)

        super().__init__(
            "Embedding",
            Dim(int(input_dim)),
            Dim(int(output_dim)),
            embedding_array.T,
        )

    def conjugate(self):
        return self


def dual_rail(n, internal_states=None):
    """
    Encode n qubits into 2n modes via the dual-rail encoding.
    """

    if internal_states is None:
        d = DualRail()
        for i in range(n - 1):
            d @= DualRail()
    else:
        d = DualRail(internal_state=internal_states[0])
        for i, state in enumerate(internal_states[1:]):
            d @= DualRail(internal_state=state)
    return d


def embedding_tensor(n, dim):
    """
    Obtain 2->dim embedding tensors on n wires.
    """
    d = EmbeddingTensor(2, dim)
    for i in range(n - 1):
        d @= EmbeddingTensor(2, dim)
    return d


def truncation_tensor(
    input_dims: List[int], output_dims: List[int]
) -> tensor.Box:

    assert len(input_dims) == len(
        output_dims
    ), "input_dims and output_dims must have the same length"

    tensor = EmbeddingTensor(input_dims[0], output_dims[0])

    for i in zip(input_dims[1:], output_dims[1:]):

        tensor = tensor @ EmbeddingTensor(i[0], i[1])
    return tensor


class Category(frobenius.Category):  # pragma: no cover
    """
    A hypergraph category is a compact category with a method :code:`spiders`.
    Parameters:
        ob : The objects of the category, default is :class:`Ty`.
        ar : The arrows of the category, default is :class:`Diagram`.
    """
    ob, ar = Ty, Diagram


class Functor(frobenius.Functor):  # pragma: no cover
    """
    A hypergraph functor is a compact functor that preserves spiders.
    Parameters:
        ob (Mapping[Ty, Ty]) : Map from atomic :class:`Ty` to :code:`cod.ob`.
        ar (Mapping[Box, Diagram]) : Map from :class:`Box` to :code:`cod.ar`.
        cod (Category) : The codomain of the functor.
    """
    dom = cod = Category()

    def __call__(self, other):
        return frobenius.Functor.__call__(self, other)


class Hypergraph(hypergraph.Hypergraph):  # pragma: no cover
    category, functor = Category, Functor


bit = Bit(1)
mode = Mode(1)

Diagram.hypergraph_factory = Hypergraph
Diagram.braid_factory, Diagram.spider_factory = Swap, Spider
Diagram.ty_factory = Ty
Diagram.sum_factory = Sum
Id = Diagram.id
