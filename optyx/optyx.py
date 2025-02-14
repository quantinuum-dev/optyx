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
or other instances of :class:`optyx.Box`.
We can evaluate arbitrary compositions of the above generators via:

- exact tensor network contraction with quimb [Gray18]_ \
using the method :class:`to_tensor`.
- exact permanent evaluation with Perceval [FGL+23]_ \
using the method :class:`to_path`.

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

>>> from optyx.zw import Create, W
>>> split_photon = Create(1) >> W(2)
>>> split_photon.draw(path="docs/_static/seq_comp_example.png")

.. image:: /_static/seq_comp_example.png
    :align: center

We can also compose boxes in parallel (tensor) using the :code:`@` operator :

>>> from optyx.lo import BS, Phase
>>> beam_splitter_phase = BS @ Phase(0.5)
>>> beam_splitter_phase.draw(path="docs/_static/parallel_comp_example.png")

.. image:: /_static/parallel_comp_example.png
    :align: center

A beam-splitter from the :class:`lo` calculus can be
expressed using the :class:`zw` calculus:

>>> from optyx.lo import BS
>>> beam_splitter = BS.to_zw()
>>> beam_splitter.draw(path="docs/_static/bs_zw.png")

.. image:: /_static/bs_zw.png
    :align: center

Optyx diagrams can combine the generators from
:class:`zw` (Mode type),
:class:`lo` (Mode type) and :class:`zx` calculi (Bit type).
We can check their equivalence as tensors.

**Branching Law**

Let's check the branching law from [FC23]_.

>>> from optyx.zw import Create, W
>>> from optyx.utils import compare_arrays_of_different_sizes
>>> branching_l = Create(1) >> W(2)
>>> branching_r = Create(1) @ Create(0) + Create(0) @ Create(1)

>>> assert compare_arrays_of_different_sizes(\\
...     branching_l.to_tensor().eval().array,\\
...     branching_r.to_tensor().eval().array,\\
... )

**Hong-Ou-Mandel Effect**

The :code:`to_tensor` method supports evaluation of
diagrams like the Hong-Ou-Mandel effect:

>>> from optyx.zw import ZBox, SWAP, W, Select, Id
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

**Converting :class:`zx` Diagrams**

The :code:`to_pyzx` and :code:`from_pyzx` methods
enable conversion between Optyx and PyZX [KW20]_:

>>> from optyx.zx import Z, SWAP
>>> assert Diagram.from_pyzx(Z(0, 2).to_pyzx()) == Z(0, 2) >> SWAP

**:class:`zw` diagrams from Bosonic Operators**

The :code:`from_bosonic_operator` method
supports creating :class:`path` diagrams:

>>> from optyx.zw import Split, Select, Id, Mode, Scalar
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

>>> from optyx.zw import Create, W
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
from discopy import symmetric, frobenius, tensor
from discopy.cat import factory, rsubs
from discopy.frobenius import Dim
from discopy.quantum.gates import format_number
from optyx.utils import modify_io_dims_against_max_dim


class Ob(frobenius.Ob):
    """Basic object in an optyx Diagram: bit or mode"""


@factory
class Ty(frobenius.Ty):
    """Optical and (qu)bit types."""
    ob_factory = Ob


class Mode(Ty):
    """Optical mode interpreted as the infinite space with countable basis"""

    def __init__(self, n=0):
        self.n = n
        super().__init__(*["mode" for _ in range(n)])


class Bit(Ty):
    """Qubit type interpreted as the two dimensional complext vector space"""

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
        """ Conjugates every box in the diagram"""
        return symmetric.Functor(
            ob=lambda x: x,
            ar=lambda f: f.conjugate(),
            cod=symmetric.Category(Ty, Diagram),
            dom=symmetric.Category(Ty, Diagram),
        )(self)

    def to_zw(self) -> Diagram:
        """To be used with :class:`lo` diagrams which can
        be decomposed into the underlying
        :class:`zw` generators."""
        return symmetric.Functor(
            ob=lambda x: x,
            ar=lambda f: f.to_zw(),
            cod=symmetric.Category(Ty, Diagram),
            dom=symmetric.Category(Ty, Diagram),
        )(self)

    def to_path(self, dtype: type = complex):
        """Returns the :class:`Matrix` normal form
        of a :class:`Diagram`.
        In other words, it is the underlying matrix
        representation of a :class:`path` and :class:`lo` diagrams."""
        from optyx import path

        return symmetric.Functor(
            ob=len,
            ar=lambda f: f.to_path(dtype),
            cod=symmetric.Category(int, path.Matrix[dtype]),
        )(self)

    def to_tensor(
        self, input_dims: list = None, max_dim: int = None
    ) -> tensor.Diagram:
        """Returns a :class:`tensor.Diagram` for evaluation"""

        def list_to_dim(dims: np.ndarray | list) -> Dim:
            """Converts a list of dimensions to a Dim object"""
            return Dim(*[int(i) for i in dims])

        if input_dims is None:
            layer_dims = [2 for _ in range(len(self.dom))]
        else:
            layer_dims = input_dims

            if max_dim is not None:
                layer_dims, _ = modify_io_dims_against_max_dim(
                    layer_dims, None, max_dim
                )

        if len(self.boxes) == 0 and len(self.offsets) == 0:
            return tensor.Diagram.id(list_to_dim(layer_dims))

        right_dim = len(self.dom)
        for i, (box, off) in enumerate(zip(self.boxes, self.offsets)):
            dims_in = layer_dims[off:off + len(box.dom)]

            dims_out = box.determine_output_dimensions(dims_in)

            if max_dim is not None:
                dims_out, _ = modify_io_dims_against_max_dim(
                    dims_out, None, max_dim
                )

            left = Dim()
            if off > 0:
                left = list_to_dim(layer_dims[0:off])
            right = Dim()
            if off + len(box.dom) < right_dim:
                right = list_to_dim(layer_dims[off + len(box.dom):right_dim])

            cod_right_dim = right_dim - len(box.dom) + len(box.cod)
            cod_layer_dims = (
                layer_dims[0:off] + dims_out + layer_dims[off + len(box.dom):]
            )

            diagram_ = left @ box.truncation(dims_in, dims_out) @ right

            if i == 0:
                diagram = diagram_
            else:
                diagram = diagram >> diagram_

            right_dim = cod_right_dim
            layer_dims = cod_layer_dims
        return diagram

    @classmethod
    def from_bosonic_operator(cls, n_modes, operators, scalar=1):
        """Create a :class:`zw` diagram from a bosonic operator."""
        from optyx import zw

        d = cls.id(Mode(n_modes))
        annil = zw.Split(2) >> zw.Select(1) @ zw.Id(Mode(1))
        create = annil.dagger()
        for idx, dagger in operators:
            if not 0 <= idx < n_modes:
                raise ValueError(f"Index {idx} out of bounds.")
            box = create if dagger else annil
            d = d >> zw.Id(idx) @ box @ zw.Id(n_modes - idx - 1)

        if scalar != 1:
            d = zw.Scalar(scalar) @ d
        return d

    def to_pyzx(self):
        """
        Returns a :class:`pyzx.Graph`.

        >>> import optyx.zx as zx
        >>> bialgebra = zx.Z(1, 2, .25) @ zx.Z(1, 2, .75) >> Id(Bit(1)) @ \\
        ...   zx.SWAP @ Id(Bit(1)) >> zx.X(2, 1, .5) @ zx.X(2, 1, .5)
        >>> graph = bialgebra.to_pyzx()
        >>> assert len(graph.vertices()) == 8
        >>> assert (graph.inputs(), graph.outputs()) == ((0, 1), (6, 7))
        >>> from pyzx import VertexType
        >>> assert graph.type(2) == graph.type(3) == VertexType.Z
        >>> assert graph.phase(2) == 2 * .25 and graph.phase(3) == 2 * .75
        >>> assert graph.type(4) == graph.type(5) == VertexType.X
        >>> assert graph.phase(4) == graph.phase(5) == 2 * .5
        >>> assert graph.graph == {
        ...     0: {2: 1},
        ...     1: {3: 1},
        ...     2: {0: 1, 4: 1, 5: 1},
        ...     3: {1: 1, 4: 1, 5: 1},
        ...     4: {2: 1, 3: 1, 6: 1},
        ...     5: {2: 1, 3: 1, 7: 1},
        ...     6: {4: 1},
        ...     7: {5: 1}}
        """
        from pyzx import Graph, VertexType, EdgeType
        from optyx import zx

        graph, scan = Graph(), []
        for i, _ in enumerate(self.dom):
            node, hadamard = graph.add_vertex(VertexType.BOUNDARY), False
            scan.append((node, hadamard))
            graph.set_inputs(graph.inputs() + (node,))
            graph.set_position(node, i, 0)
        for row, (box, offset) in enumerate(zip(self.boxes, self.offsets)):
            if isinstance(box, zx.Spider):
                node = graph.add_vertex(
                    (VertexType.Z if isinstance(box, zx.Z) else VertexType.X),
                    phase=box.phase * 2 if box.phase else None,
                )
                graph.set_position(node, offset, row + 1)
                for i, _ in enumerate(box.dom):
                    source, hadamard = scan[offset + i]
                    etype = EdgeType.HADAMARD if hadamard else EdgeType.SIMPLE
                    graph.add_edge((source, node), etype)
                scan = (
                    scan[:offset]
                    + len(box.cod) * [(node, False)]
                    + scan[offset + len(box.dom):]
                )
            elif isinstance(box, Swap):
                scan = (
                    scan[:offset]
                    + [scan[offset + 1], scan[offset]]
                    + scan[offset + 2:]
                )
            elif isinstance(box, zx.Scalar):
                graph.scalar.add_float(box.data)
            elif box == zx.H:
                node, hadamard = scan[offset]
                scan[offset] = (node, not hadamard)
            else:
                raise NotImplementedError
        for i, _ in enumerate(self.cod):
            target = graph.add_vertex(VertexType.BOUNDARY)
            source, hadamard = scan[i]
            etype = EdgeType.HADAMARD if hadamard else EdgeType.SIMPLE
            graph.add_edge((source, target), etype)
            graph.set_position(target, i, len(self) + 1)
            graph.set_outputs(graph.outputs() + (target,))
        return graph

    @staticmethod
    def from_pyzx(graph):
        """
        Takes a :class:`pyzx.Graph` returns a :class:`zx.Diagram`.

        Examples
        --------

        >>> import optyx.zx as zx
        >>> bialgebra = zx.Z(1, 2, .25) @ zx.Z(1, 2, .75) >> \\
        ...    zx.Id(Bit(1)) @ zx.SWAP @ zx.Id(Bit(1)) >> \\
        ...    zx.X(2, 1, .5) @ zx.X(2, 1, .5)
        >>> graph = bialgebra.to_pyzx()
        >>> assert Diagram.from_pyzx(graph) == bialgebra

        Note
        ----

        Raises :code:`ValueError` if either:
        * a boundary node is not in :code:`graph.inputs() + graph.outputs()`,
        * or :code:`set(graph.inputs()).intersection(graph.outputs())`.
        """
        from pyzx import VertexType, EdgeType
        from optyx import zx

        def node2box(node, n_legs_in, n_legs_out):
            if graph.type(node) not in {VertexType.Z, VertexType.X}:
                raise NotImplementedError  # pragma: no cover
            return (
                zx.Z if graph.type(node) is VertexType.Z else zx.X
            )(  # noqa: E721
                n_legs_in, n_legs_out, graph.phase(node) * 0.5
            )

        def move(scan, source, target):
            if target < source:
                swaps = (
                    Id(Bit(target))
                    @ Diagram.swap(Bit(source - target), Bit(1))
                    @ Id(Bit(len(scan) - source - 1))
                )
                scan = (
                    scan[:target]
                    + (scan[source],)
                    + scan[target:source]
                    + scan[source + 1:]
                )
            elif target > source:
                swaps = (
                    Id(Bit(source))
                    @ Diagram.swap(Bit(1), Bit(target - source))
                    @ Id(Bit(len(scan) - target - 1))
                )
                scan = (
                    scan[:source]
                    + scan[source + 1:target]
                    + (scan[source],)
                    + scan[target:]
                )
            else:
                swaps = Id(Bit(len(scan)))
            return scan, swaps

        def make_wires_adjacent(scan, diagram, inputs):
            if not inputs:
                return scan, diagram, len(scan)
            offset = scan.index(inputs[0])
            for i, _ in enumerate(inputs[1:]):
                source, target = scan.index(inputs[i + 1]), offset + i + 1
                scan, swaps = move(scan, source, target)
                diagram = diagram >> swaps
            return scan, diagram, offset

        missing_boundary = any(
            graph.type(node) == VertexType.BOUNDARY  # noqa: E721
            and node not in graph.inputs() + graph.outputs()
            for node in graph.vertices()
        )
        if missing_boundary:
            raise ValueError
        duplicate_boundary = set(graph.inputs()).intersection(graph.outputs())
        if duplicate_boundary:
            raise ValueError
        diagram, scan = Id(Bit(len(graph.inputs()))), graph.inputs()
        for node in [
            v
            for v in graph.vertices()
            if v not in graph.inputs() + graph.outputs()
        ]:
            inputs = [
                v
                for v in graph.neighbors(node)
                if v < node and v not in graph.outputs() or v in graph.inputs()
            ]
            inputs.sort(key=scan.index)
            outputs = [
                v
                for v in graph.neighbors(node)
                if v > node and v not in graph.inputs() or v in graph.outputs()
            ]
            scan, diagram, offset = make_wires_adjacent(scan, diagram, inputs)
            hadamards = Id(Bit(0)).tensor(
                *[
                    (
                        zx.H
                        if graph.edge_type((i, node)) == EdgeType.HADAMARD
                        else Id(Bit(1))
                    )
                    for i in scan[offset:offset + len(inputs)]
                ]
            )
            box = node2box(node, len(inputs), len(outputs))
            diagram = diagram >> Id(Bit(offset)) @ (hadamards >> box) @ Id(
                Bit(len(diagram.cod) - offset - len(inputs))
            )
            scan = (
                scan[:offset]
                + len(outputs) * (node,)
                + scan[offset + len(inputs):]
            )
        for target, output in enumerate(graph.outputs()):
            (node,) = graph.neighbors(output)
            etype = graph.edge_type((node, output))
            hadamard = zx.H if etype == EdgeType.HADAMARD else Id(Bit(1))
            scan, swaps = move(scan, scan.index(node), target)
            diagram = (
                diagram
                >> swaps
                >> Id(Bit(target)) @ hadamard @ Id(Bit(len(scan) - target - 1))
            )
        return diagram


class Box(frobenius.Box, Diagram):
    """A box in an optyx diagram"""

    _array = None

    __ambiguous_inheritance__ = (frobenius.Box,)

    def conjugate(self):
        raise NotImplementedError

    def to_zw(self):
        raise NotImplementedError

    def to_path(self, dtype: type = complex):
        raise NotImplementedError

    def truncation(
        self, input_dims: list[int] = None, output_dims: list[int] = None
    ) -> tensor.Box:
        """Create a tensor in the semantics of a ZW diagram"""
        raise NotImplementedError

    @property
    def array(self):
        """Create an array in the semantics of a ZX diagram"""
        raise NotImplementedError

    @array.setter
    def array(self, value):
        self._array = value

    @array.getter
    def array(self):
        return self._array

    def determine_output_dimensions(self, input_dims: list[int]) -> list[int]:
        """Determine the output dimensions based on the input dimensions.
        The generators of ZW affect the dimensions
        of the output tensor diagrams."""
        raise NotImplementedError

    def lambdify(self, *symbols, **kwargs):
        # Non-symbolic gates can be returned directly
        return lambda *xs: self

    def subs(self, *args) -> Diagram:
        syms, exprs = zip(*args)
        return self.lambdify(*syms)(*exprs)


class Spider(frobenius.Spider, Box):
    """Abstract spider (dagger-SCFA)"""

    draw_as_spider = True
    color = "green"

    def conjugate(self):
        return self

    def to_zw(self):
        return self

    def determine_output_dimensions(self,
                                    input_dims: list[int]) -> list[int]:
        if isinstance(self.cod, Bit):
            return [2]*len(self.cod)
        else:
            if len(self.dom) == 0:
                return [2 for _ in range(len(self.cod))]
            return [min(input_dims) for _ in range(len(self.cod))]

    def truncation(
            self,
            input_dims: list[int] = None,
            output_dims: list[int] = None
    ) -> tensor.Box:
        """
        Create a tensor in the semantics of a ZW/ZX diagram depending
        on the domain and codomain type
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
        """Convert the sum to a path diagram.
        For this function to fully work,
        formal sums of Path matrices need to be implemented."""
        return sum(term.to_path(dtype) for term in self.terms)

    def eval(self, n_photons=0, permanent=None, dtype=complex):
        """Evaluate the sum of diagrams."""
        # we need to implement the proper sums of qpath diagrams
        # this is only a temporary solution, so that the grad tests pass
        if permanent is None:
            from optyx.path import npperm

            permanent = npperm
        return sum(
            term.to_path(dtype).eval(n_photons, permanent)
            for term in self.terms
        )

    def to_tensor(self, input_dims=None, max_dim=None):

        terms = [t.to_tensor(input_dims, max_dim) for t in self]
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
        from optyx.path import Matrix

        return Matrix([0, 1, 1, 0], 2, 2)

    def to_zw(self):
        return self

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
    >>> from optyx.path import Matrix
    >>> from optyx.zw import Create, Select
    >>> from optyx.lo import BS
    >>> assert Scalar(0.45).to_path() == Matrix(
    ...     [], dom=0, cod=0,
    ...     creations=(), selections=(), normalisation=1, scalar=0.45)
    >>> s = Scalar(- 1j * 2 ** (1/2)) @ Create(1, 1) >> BS >> Select(2, 0)
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

    @property
    def array(self):
        return np.array([self.scalar], dtype=complex)

    def __str__(self):
        return f"scalar({format_number(self.data)})"

    def to_path(self, dtype: type = complex):
        from optyx.path import Matrix

        return Matrix[dtype]([], 0, 0, scalar=self.scalar)

    def dagger(self) -> Diagram:
        return Scalar(self.scalar.conjugate())

    def to_zw(self):
        from optyx.zw import ZBox

        return ZBox(0, 0, self.array)

    def subs(self, *args):
        data = rsubs(self.scalar, *args)
        return Scalar(data)

    def grad(self, var, **params):
        """Gradient with respect to :code:`var`."""
        if var not in self.free_symbols:
            return Sum((), self.dom, self.cod)
        return Scalar(self.scalar.diff(var))

    def lambdify(self, *symbols, **kwargs):
        from sympy import lambdify

        return lambda *xs: type(self)(
            lambdify(symbols, self.scalar, **kwargs)(*xs)
        )

    def truncation(
        self, input_dims: list[int] = None, output_dims: list[int] = None
    ) -> tensor.Box:
        return tensor.Box(self.name, Dim(1), Dim(1), self.array)

    def determine_output_dimensions(
        self, input_dims: list[int] = None
    ) -> list[int]:
        """Determine the output dimensions"""
        return [1]


class DualRail(Box):
    """
    A map from :code:`Bit` to :code:`Mode` using the dual rail encoding.
    """

    def __init__(self, is_dagger=False):
        dom = Mode(2) if is_dagger else Bit(1) 
        cod = Bit(1) if is_dagger else Mode(2)
        self.is_dagger = is_dagger
        super().__init__("2R", dom, cod)

    def conjugate(self):
        return self

    def truncation(
        self, input_dims: list[int] = None, output_dims: list[int] = None
    ) -> tensor.Box:
        """:class:`tensor.Box` for the dual rail encoding."""
        if self.is_dagger:
            array = np.zeros((2, input_dims[0], input_dims[1]), dtype=complex)
        else:
            array = np.zeros((2, 2, 2), dtype=complex)
        array[0, 1, 0] = 1
        array[1, 0, 1] = 1
        if self.is_dagger:
            return tensor.Box(self.name, Dim(2), 
                              Dim(*[int(i) for i in input_dims]), 
                              array).dagger()
        return tensor.Box(self.name, Dim(2), Dim(2, 2), array)

    def determine_output_dimensions(self, input_dims: list[int]) -> list[int]:
        """Determine the output dimensions"""
        if self.is_dagger:
            return [2]
        return [2, 2]

    def to_zw(self):
        return self

    def dagger(self) -> Diagram:
        return DualRail(not self.is_dagger)


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


def dual_rail(n):
    '''
    Encode n qubits into 2n modes via the dual-rail encoding.
    '''
    d = DualRail()
    for i in range(n-1):
        d @= DualRail()
    return d


def embedding_tensor(n, dim):
    '''
    Obtain 2->dim embedding tensors on n wires.
    '''
    d = EmbeddingTensor(2, dim)
    for i in range(n-1):
        d @= EmbeddingTensor(2, dim)
    return d


bit = Bit(1)
mode = Mode(1)

Diagram.braid_factory, Diagram.spider_factory = Swap, Spider
Diagram.ty_factory = Ty
Diagram.sum_factory = Sum
Id = Diagram.id
