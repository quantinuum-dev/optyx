"""
Optics diagrams combine ZW_infty, QPath and ZX calculi.

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Mode
    Bit
    Diagram
    Box
    Swap
    Permutation
    Scalar

Example
-------

Optyx diagrams can combine the generators from ZW_infty (Mode type),
QPath (Mode type) and ZX calculi (Bit type).

Branching law from QPath and ZW_infty:

>>> from optyx.zw import Create, W
>>> from optyx.utils import compare_arrays_of_different_sizes
>>> branching_l = Create(1) >> W(2)
>>> branching_r = Create(1) @ Create(0) + Create(0) @ Create(1)

>>> assert compare_arrays_of_different_sizes(\\
...     branching_l.to_tensor().eval().array,\\
...     branching_r.to_tensor().eval().array,\\
... )

Specific methods of the optyx.Diagram class are applicable to
specific types of diagrams.

The to_tensor method is applicable to all types of diagrams. For example,
the Hong-Ou-Mandel effect can be implemented as follows:

>>> from optyx.zw import Z, SWAP, W, Select, Id
>>> Zb_i = Z(np.array([1, 1j/(np.sqrt(2))]), 1, 1)
>>> Zb_1 = Z(np.array([1, 1/(np.sqrt(2))]), 1, 1)
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

The to_pyzx and from_pyzx methods are to applicable to ZX diagrams:

>>> from optyx.zx import Z, SWAP
>>> assert Diagram.from_pyzx(Z(0, 2).to_pyzx()) == Z(0, 2) >> SWAP

The method from_bosonic_operator is applicable to QPath diagrams:

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

Finally, to_path method is applicable to QPath diagrams:

>>> from optyx.zw import Create, W
>>> counit_l = W(2) >> Select(0) @ Id(Mode(1))
>>> counit_r = W(2) >> Id(Mode(1)) @ Select(0)
>>> assert counit_l.to_path().eval(2) == counit_r.to_path().eval(2)
"""

from __future__ import annotations

import numpy as np
from sympy.core import Symbol, Mul
from discopy import symmetric, frobenius, tensor
from discopy.cat import factory, rsubs
from discopy.frobenius import Dim
from discopy.quantum.gates import format_number


class Ty(frobenius.Ty):
    """Optical and (qu)bit types."""


mode = Ty("mode")
bit = Ty("bit")


class Mode(Ty):
    """Optical mode type with infinite dimensions."""

    def __init__(self, n=0):
        self.n = n
        super().__init__(*["mode" for _ in range(n)])


class Bit(Ty):
    """(Qu)bit type."""

    def __init__(self, n=0):
        self.n = n
        super().__init__(*["bit" for _ in range(n)])


@factory
class Diagram(frobenius.Diagram):
    """Optyx diagram combining ZX, ZW_infty and QPath calculi."""

    grad = tensor.Diagram.grad

    def to_zw(self) -> Diagram:
        """ To be used with optyx.circuit diagrams """
        return symmetric.Functor(
            ob=lambda x: x,
            ar=lambda f: f.to_zw(),
            cod=symmetric.Category(Ty, Diagram),
        )(self)

    def to_path(self, dtype: type = complex):
        """Returns the :class:`Matrix` normal form of a :class:`Diagram`."""
        from optyx import qpath

        return symmetric.Functor(
            ob=len,
            ar=lambda f: f.to_path(dtype),
            cod=symmetric.Category(int, qpath.Matrix[dtype]),
        )(self)

    def to_tensor(self, input_dims: list = None) -> tensor.Diagram:
        """Returns a tensor.Diagram for evaluation"""

        if input_dims is None:
            layer_dims = [2 for _ in range(len(self.dom))]
        else:
            layer_dims = input_dims

        def f_ob(dims: np.ndarray | list) -> Dim:
            """Converts a list of dimensions to a Dim object"""
            return Dim(*[int(i) for i in dims])

        def f_ar(box: Box, dims_in: list, dims_out: list) -> tensor.Box:
            """Converts a box to a tensor.Box object
            with the correct dimensions and array"""
            arr = box.truncated_array(np.array(dims_in))
            return tensor.Box(box.name, f_ob(dims_in), f_ob(dims_out), arr)

        def get_embedding_tensor(
            input_dim: int, output_dim: int
        ) -> np.ndarray:
            """Returns the embedding tensor for the given
            input and output dimensions."""
            embedding_array = np.zeros(
                (output_dim, input_dim), dtype=complex
            )
            embedding_array[:input_dim, :input_dim] = np.eye(input_dim)
            embedding_tensor = tensor.Box(
                "Embedding",
                Dim(input_dim),
                Dim(output_dim),
                embedding_array.T,
            )
            return embedding_tensor

        def get_diagram_for_identities(layer_dims):
            dims_in = layer_dims[: len(self.dom)]
            dims_out = dims_in

            diagram = tensor.Box(
                "Id",
                f_ob(dims_in),
                f_ob(dims_out),
                np.eye(int(np.prod(np.array(dims_in)))),
            )

            return diagram

        def get_diagram_for_a_single_term(layer_dims):
            right_dim = len(self.dom)
            for i, (box, off) in enumerate(zip(self.boxes, self.offsets)):
                dims_in = layer_dims[off: off + len(box.dom)]

                dims_out = box.determine_output_dimensions(dims_in)

                left = Dim()
                if off > 0:
                    left = f_ob(layer_dims[0:off])
                right = Dim()
                if off + len(box.dom) < right_dim:
                    right = f_ob(
                        layer_dims[off + len(box.dom): right_dim]
                    )

                cod_right_dim = right_dim - len(box.dom) + len(box.cod)
                cod_layer_dims = (
                    layer_dims[0:off]
                    + dims_out
                    + layer_dims[off + len(box.dom):]
                )

                diagram_ = left @ f_ar(box, dims_in, dims_out) @ right

                if i == 0:
                    diagram = diagram_
                else:
                    diagram = diagram >> diagram_

                right_dim = cod_right_dim
                layer_dims = cod_layer_dims
            return diagram

        def get_diagram_for_sums(input_dims):
            """If the diagram is a sum,
            need to find the common dimensions for the cod for all the terms -
            can do it by finding the max dims for each idx
            and set it for all the terms
            - then we can apply the new dims to all
            the last boxes on each wire
            for all the terms"""

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
                    embedding_layer = (
                        embedding_layer
                        @ get_embedding_tensor(d.inside[0], max_dims[wire])
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

        if isinstance(self, Sum):
            return get_diagram_for_sums(input_dims)
        if len(self.boxes) == 0 and len(self.offsets) == 0:
            return get_diagram_for_identities(layer_dims)

        return get_diagram_for_a_single_term(layer_dims)

    @classmethod
    def from_bosonic_operator(cls, n_modes, operators, scalar=1):
        """ Create a QPath diagram from a bosonic operator."""
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
                    (
                        VertexType.Z
                        if isinstance(box, zx.Z)
                        else VertexType.X
                    ),
                    phase=box.phase * 2 if box.phase else None,
                )
                graph.set_position(node, offset, row + 1)
                for i, _ in enumerate(box.dom):
                    source, hadamard = scan[offset + i]
                    etype = (
                        EdgeType.HADAMARD if hadamard else EdgeType.SIMPLE
                    )
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
                zx.Z if graph.type(node) == VertexType.Z else zx.X
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
                    + scan[source + 1: target]
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
        duplicate_boundary = set(graph.inputs()).intersection(
            graph.outputs()
        )
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
                if v < node
                and v not in graph.outputs()
                or v in graph.inputs()
            ]
            inputs.sort(key=scan.index)
            outputs = [
                v
                for v in graph.neighbors(node)
                if v > node
                and v not in graph.inputs()
                or v in graph.outputs()
            ]
            scan, diagram, offset = make_wires_adjacent(
                scan, diagram, inputs
            )
            hadamards = Id(Bit(0)).tensor(
                *[
                    (
                        zx.H
                        if graph.edge_type((i, node)) == EdgeType.HADAMARD
                        else Id(Bit(1))
                    )
                    for i in scan[offset: offset + len(inputs)]
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
                >> Id(Bit(target))
                @ hadamard
                @ Id(Bit(len(scan) - target - 1))
            )
        return diagram


class Box(frobenius.Box, Diagram):
    """A box in an optyx diagram"""

    _array = None

    __ambiguous_inheritance__ = (frobenius.Box,)

    def to_zw(self):
        raise NotImplementedError

    def to_path(self, dtype: type = complex):
        raise NotImplementedError

    def truncated_array(self, input_dims):
        """ Create array in the semantics of a ZW diagram """
        raise NotImplementedError

    @property
    def array(self):
        """ Create array in the semantics of a ZX diagram """
        raise NotImplementedError

    @array.setter
    def array(self, value):
        self._array = value

    @array.getter
    def array(self):
        return self._array

    def determine_output_dimensions(self, input_dims):
        """ Determine the output dimensions based on the input dimensions.
        The generators of ZW affect the dimensions
        of the output tensor diagrams. """
        raise NotImplementedError

    def lambdify(self, *symbols, **kwargs):
        # Non-symbolic gates can be returned directly
        return lambda *xs: self

    def subs(self, *args) -> Diagram:
        syms, exprs = zip(*args)
        return self.lambdify(*syms)(*exprs)


class Sum(symmetric.Sum, Box):
    """
    Formal sum of optyx diagrams
    """

    __ambiguous_inheritance__ = (symmetric.Sum,)

    def to_path(self, dtype: type = complex):
        """ Convert the sum to a path diagram.
        For this function to fully work,
        formal sums of Path matrices need to be implemented."""
        return sum(term.to_path(dtype) for term in self.terms)

    def eval(self, n_photons=0, permanent=None, dtype=complex):
        """ Evaluate the sum of diagrams. """
        # we need to implement the proper sums of qpath diagrams
        # this is only a temporary solution, so that the grad tests pass
        if permanent is None:
            from optyx.qpath import npperm

            permanent = npperm
        return sum(
            term.to_path(dtype).eval(n_photons, permanent)
            for term in self.terms
        )

    def grad(self, var, **params):
        """Gradient with respect to :code:`var`."""
        if var not in self.free_symbols:
            return self.sum_factory((), self.dom, self.cod)
        return sum(term.grad(var, **params) for term in self.terms)


class Swap(frobenius.Swap, Box):
    """Swap in optyx diagram"""

    def to_path(self, dtype: type = complex):
        from optyx.qpath import Matrix

        return Matrix([0, 1, 1, 0], 2, 2)

    def to_zw(self):
        return self

    def determine_output_dimensions(
        self, input_dims: list[int]
    ) -> list[int]:
        """Determine the output dimensions based on the input dimensions."""
        return input_dims[::-1]

    def truncated_array(self, input_dims: list[int]) -> np.ndarray:
        return Permutation(self.dom, [1, 0]).truncated_array(input_dims)


class Permutation(Box):
    """Permute wires in an optyx diagram"""

    def __init__(
        self, dom: Ty, permutation: list[int], is_dagger: bool = False
    ):
        """
        Args:
            dom: The input type
            permutation: List of indices representing the permutation.
                         Each entry indicates where the
                         corresponding input goes in the output.
        """
        assert len(permutation) == len(dom)

        cod = Ty.tensor(*[dom[i] for i in permutation])
        super().__init__(str(permutation), dom, cod)
        self.is_dagger = is_dagger
        self.permutation = permutation

    def truncated_array(
        self, input_dims: list[int], output_dims: list[int] = None
    ) -> np.ndarray:
        """Create an array that permutes the occupation
        numbers based on the input dimensions."""

        if output_dims is None:
            output_dims = self.determine_output_dimensions(input_dims)

        input_total_dim = int(np.prod(input_dims))
        output_total_dim = int(np.prod(output_dims))
        perm_matrix = np.zeros(
            (output_total_dim, input_total_dim), dtype=complex
        )

        for input_index in np.ndindex(*input_dims):
            permuted_index = tuple(
                input_index[self.permutation[i]]
                for i in range(len(self.permutation))
            )
            input_flat_index = np.ravel_multi_index(
                input_index, input_dims
            )
            permuted_flat_index = np.ravel_multi_index(
                permuted_index, output_dims
            )
            perm_matrix[permuted_flat_index, input_flat_index] = 1

        return perm_matrix.T

    def determine_output_dimensions(
        self, input_dims: list[int]
    ) -> list[int]:
        """Determine the output dimensions based on the permutation."""
        return [input_dims[i] for i in self.permutation]

    def dagger(self) -> Diagram:
        n = len(self.permutation)
        inverse_permutation = [0] * n
        for i, j in enumerate(self.permutation):
            inverse_permutation[j] = i

        return Permutation(
            self.dom,
            inverse_permutation,
            not self.is_dagger,
        )

    def to_path(self, dtype: type = complex):
        from optyx.qpath import Matrix

        array = np.zeros(
            (len(self.cod.inside), len(self.dom.inside)), dtype=complex
        )
        for i, p in enumerate(self.permutation):
            array[p, i] = 1
        return Matrix(array, len(self.dom), len(self.cod))


class Scalar(Box):
    """
    Scalar in a diagram

    Example
    -------
    >>> from optyx.qpath import Matrix
    >>> from optyx.zw import Create, Select, BS
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

    @property
    def array(self):
        return np.array([self.scalar], dtype=complex)

    def __str__(self):
        return f"scalar({format_number(self.data)})"

    def to_path(self, dtype: type = complex):
        from optyx.qpath import Matrix

        return Matrix[dtype]([], 0, 0, scalar=self.scalar)

    def dagger(self) -> Diagram:
        return Scalar(self.scalar.conjugate())

    def subs(self, *args):
        data = rsubs(self.scalar, *args)
        return Scalar(data)

    def grad(self, var, **params):
        """ Gradient with respect to :code:`var`."""
        if var not in self.free_symbols:
            return Sum((), self.dom, self.cod)
        return Scalar(self.scalar.diff(var))

    def lambdify(self, *symbols, **kwargs):
        from sympy import lambdify

        return lambda *xs: type(self)(
            lambdify(symbols, self.scalar, **kwargs)(*xs)
        )

    def truncated_array(self, _=None, __=None) -> np.ndarray[complex]:
        return self.array

    def determine_output_dimensions(self, _=None) -> list[int]:
        return [1]


Diagram.swap_factory = Swap
Diagram.swap = Swap
Diagram.sum_factory = Sum
Id = Diagram.id
