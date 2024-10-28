from __future__ import annotations

import numpy as np
from sympy.core import Symbol, Mul
from discopy import symmetric, frobenius, tensor
from discopy.cat import factory, rsubs
from discopy.frobenius import Dim
from discopy.monoidal import Layer
from discopy.quantum.gates import (format_number)

class Ty(frobenius.Ty):
    pass

mode = Ty('mode')
bit = Ty('bit')

class Mode(Ty):
    def __init__(self, n=0):
        self.n = n
        super().__init__(*['mode' for _ in range(n)])

class Bit(Ty):
    def __init__(self, n=0):
        self.n = n
        super().__init__(*['bit' for _ in range(n)])

@factory
class Diagram(frobenius.Diagram):
    """ Optyx diagram """

    grad = tensor.Diagram.grad

    def to_zw(self) -> Diagram:
        return symmetric.Functor(
            ob= lambda x: x,
            ar= lambda f: f.to_zw(),
            cod=symmetric.Category(Ty, Diagram),
        )(self)

    def to_path(self, dtype: type = complex) -> Matrix:
        """Returns the :class:`Matrix` normal form of a :class:`Diagram`."""
        from optyx import qpath
        return symmetric.Functor(
            ob=len,
            ar=lambda f: f.to_path(dtype),
            cod=symmetric.Category(int, qpath.Matrix[dtype]),
        )(self)

    def to_tensor(self, input_dims: list = None) -> tensor.Diagram:
        """Returns a a tensor.Diagram for evaluation"""

        def f_ob(dims: np.ndarray | list) -> Dim:
            """Converts a list of dimensions to a Dim object"""
            return Dim(*[int(i) for i in dims])

        def f_ar(box: Box, dims_in: list, dims_out: list) -> tensor.Box:
            """Converts a box to a tensor.Box object
            with the correct dimensions and array"""
            arr = box.truncated_array(np.array(dims_in))
            return tensor.Box(
                box.name, f_ob(dims_in), f_ob(dims_out), arr
            )

        if input_dims is None:
            layer_dims = [2 for _ in range(len(self.dom))]
        else:
            layer_dims = input_dims

        if not isinstance(self, Sum):
            right_dim = len(self.dom)

            if len(self.boxes) == 0 and len(self.offsets) == 0:
                dims_in = layer_dims[:len(self.dom)]
                dims_out = dims_in

                diagram = tensor.Box(
                    "Id", f_ob(dims_in), f_ob(dims_out), np.eye(int(np.prod(np.array(dims_in)))) 
                )
                
                return diagram

            for i, (box, off) in enumerate(zip(self.boxes, self.offsets)):
                dims_in = layer_dims[off: off + len(box.dom)]

                dims_out = box.determine_dimensions(
                    dims_in
                )

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
        
        else:
            # find the common dimensions for all the terms
            terms = [t.to_tensor(input_dims) for t in self]
            cods = [list(t.cod.inside) for t in terms]

            #figure out the max dims for each idx and set it for all the terms
            max_dims = [max([c[i] if len(c) > 0 else 0 for c in cods]) for i in range(len(cods[0]))]

            for i in range(len(terms)):
                terms[i].cod = Dim(*max_dims)
                boxes_dims_offsets = self._find_boxes_to_modify_for_sums(terms[i], max_dims.copy())
                terms[i] = self._modify_boxes_for_sums(terms[i], boxes_dims_offsets)

            for i, term in enumerate(terms):
                if i == 0:
                    diagram = term
                else:
                    diagram += term   
            return diagram

    def _find_boxes_to_modify_for_sums(self, term, max_dims):
        boxes_dims_offsets = []
        len_max_dims = len(max_dims)
        #get the boxes which need to be modified together 
        #with all the position "parameters"
        for j, (box, off) in enumerate(zip(term.boxes[::-1], term.offsets[::-1])):   
            # if the box is a postselection - no output wire               
            if box.cod == Dim(1) and box.dom != Dim(1):
                for _ in range(len(box.dom)):
                    max_dims.insert(off, 0)
                continue

            #for all other wires
            else:
                boxes_dims_offsets.append((box, 
                                           max_dims[off:off + len(box.cod)], 
                                           off, 
                                           len(term.boxes) - j - 1))

                if len(box.cod) > len(box.dom):                            
                    # replace the max_dims[off:off + len(box.cod)] with [0]*len(box.cod)
                    max_dims[off:off + len(box.cod)] = [0]*len(box.cod)
                elif len(box.cod) < len(box.dom):
                    # insert 0s in the max_dims[off:off + len(box.cod)]
                    for _ in range(len(box.dom) - len(box.cod)):
                        max_dims.insert(off, 0)
                else:
                    pass

            if len(boxes_dims_offsets) == len_max_dims:
                break
        
        return boxes_dims_offsets


    def _modify_boxes_for_sums(self, term, boxes_dims_offsets):
        inside = list(term.inside)

        # modify the diagrams for all the terms
        for box, dims, off, idx in boxes_dims_offsets:
            old_layer = term.inside[idx]
            #get new arrays for all the terms
            arr = np.reshape(box.array, (int(np.prod(box.cod.inside)), 
                                         int(np.prod(box.dom.inside)))) 
            # embed old array in new array
            new_array = np.zeros((int(np.prod(dims)), int(arr.shape[1])))
            new_array[:arr.shape[0], :arr.shape[1]] = arr

            new_box = tensor.Box(
                box.name, box.dom, Dim(*[int(i) for i in dims]), new_array
            )
            new_layer = Layer(old_layer[0], new_box, old_layer[2])
            inside[idx] = new_layer

        term.inside = tuple(inside)   
        
        return term

    def grad(self, var, **params) -> Diagram.Sum:
        """
        Gradient with respect to `var`.

        Parameters
        ----------
        var : sympy.Symbol
            Differentiated variable.

        Examples
        --------
        >>> from sympy.abc import phi
        >>> from optyx.zx import scalar, Z
        >>> assert Z(1, 1, phi).grad(phi) == scalar(np.pi) @ Z(1, 1, phi + .5)
        """
        """ Gradient with respect to :code:`var`. """
        if var not in self.free_symbols:
            return self.sum_factory((), self.dom, self.cod)
        left, box, right, tail = tuple(self.inside[0]) + (self[1:], )
        t1 = self.id(left) @ box.grad(var, **params) @ self.id(right) >> tail
        t2 = self.id(left) @ box @ self.id(right) >> tail.grad(var, **params)
        return t1 + t2


    @classmethod
    def from_bosonic_operator(cls, n_modes, operators, scalar=1):
        from optyx.zw import Split, Select, Id, Mode, Scalar
        d = cls.id(Mode(n_modes))
        annil = Split(2) >> Select(1) @ Id(Mode(1))
        create = annil.dagger()
        for idx, dagger in operators:
            if not (0 <= idx < n_modes):
                raise ValueError(f"Index {idx} out of bounds.")
            box = create if dagger else annil
            d = d >> Id(idx) @ box @ Id(n_modes - idx - 1)

        if scalar != 1:
            d = Scalar(scalar) @ d
        return d


    def to_pyzx(self):
        """
        Returns a :class:`pyzx.Graph`.

        >>> import optyx.zx as zx
        >>> bialgebra = zx.Z(1, 2, .25) @ zx.Z(1, 2, .75)\\
        ...     >> Id(Bit(1)) @ zx.SWAP @ Id(Bit(1)) >> zx.X(2, 1, .5) @ zx.X(2, 1, .5)
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
        import optyx.zx as zx

        graph, scan = Graph(), []
        for i, _ in enumerate(self.dom):
            node, hadamard = graph.add_vertex(VertexType.BOUNDARY), False
            scan.append((node, hadamard))
            graph.set_inputs(graph.inputs() + (node,))
            graph.set_position(node, i, 0)
        for row, (box, offset) in enumerate(zip(self.boxes, self.offsets)):
            if isinstance(box, zx.Spider):
                node = graph.add_vertex(
                    VertexType.Z if isinstance(box, zx.Z) else VertexType.X,
                    phase=box.phase * 2 if box.phase else None)
                graph.set_position(node, offset, row + 1)
                for i, _ in enumerate(box.dom):
                    source, hadamard = scan[offset + i]
                    etype = EdgeType.HADAMARD if hadamard else EdgeType.SIMPLE
                    graph.add_edge((source, node), etype)
                scan = scan[:offset] + len(box.cod) * [(node, False)]\
                    + scan[offset + len(box.dom):]
            elif isinstance(box, Swap):
                scan = scan[:offset] + [scan[offset + 1], scan[offset]]\
                    + scan[offset + 2:]
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
        >>> bialgebra = zx.Z(1, 2, .25) @ zx.Z(1, 2, .75)\\
        ...     >> zx.Id(Bit(1)) @ zx.SWAP @ zx.Id(Bit(1)) >> zx.X(2, 1, .5) @ zx.X(2, 1, .5)
        >>> graph = bialgebra.to_pyzx()
        >>> assert Diagram.from_pyzx(graph) == bialgebra

        Note
        ----

        Raises :code:`ValueError` if either:
        * a boundary node is not in :code:`graph.inputs() + graph.outputs()`,
        * or :code:`set(graph.inputs()).intersection(graph.outputs())`.
        """
        from pyzx import VertexType, EdgeType
        import optyx.zx as zx

        def node2box(node, n_legs_in, n_legs_out):
            if graph.type(node) not in {VertexType.Z, VertexType.X}:
                raise NotImplementedError  # pragma: no cover
            return \
                (zx.Z if graph.type(node) == VertexType.Z else zx.X)(  # noqa: E721
                    n_legs_in, n_legs_out, graph.phase(node) * .5)

        def move(scan, source, target):
            if target < source:
                swaps = Id(Bit(target))\
                    @ Diagram.swap(Bit(source - target), Bit(1))\
                    @ Id(Bit(len(scan) - source - 1))
                scan = scan[:target] + (scan[source],)\
                    + scan[target:source] + scan[source + 1:]
            elif target > source:
                swaps = Id(Bit(source))\
                    @ Diagram.swap(1, target - source)\
                    @ Id(Bit(len(scan) - target - 1))
                scan = scan[:source] + scan[source + 1:target]\
                    + (scan[source],) + scan[target:]
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
            for node in graph.vertices())
        if missing_boundary:
            raise ValueError
        duplicate_boundary = set(graph.inputs()).intersection(graph.outputs())
        if duplicate_boundary:
            raise ValueError
        diagram, scan = Id(Bit(len(graph.inputs()))), graph.inputs()
        for node in [v for v in graph.vertices()
                     if v not in graph.inputs() + graph.outputs()]:
            inputs = [v for v in graph.neighbors(node) if v < node
                      and v not in graph.outputs() or v in graph.inputs()]
            inputs.sort(key=scan.index)
            outputs = [v for v in graph.neighbors(node) if v > node
                       and v not in graph.inputs() or v in graph.outputs()]
            scan, diagram, offset = make_wires_adjacent(scan, diagram, inputs)
            hadamards = Id(Bit(0)).tensor(*[
                zx.H if graph.edge_type((i, node)) == EdgeType.HADAMARD
                else Id(Bit(1)) for i in scan[offset: offset + len(inputs)]])
            box = node2box(node, len(inputs), len(outputs))
            diagram = diagram >> Id(Bit(offset)) @ (hadamards >> box)\
                @ Id(Bit(len(diagram.cod) - offset - len(inputs)))
            scan = scan[:offset] + len(outputs) * (node,)\
                + scan[offset + len(inputs):]
        for target, output in enumerate(graph.outputs()):
            node, = graph.neighbors(output)
            etype = graph.edge_type((node, output))
            hadamard = zx.H if etype == EdgeType.HADAMARD else Id(Bit(1))
            scan, swaps = move(scan, scan.index(node), target)
            diagram = diagram >> swaps\
                >> Id(Bit(target)) @ hadamard @ Id(Bit(len(scan) - target - 1))
        return diagram


class Box(frobenius.Box, Diagram):
    """A box in an optyx diagram"""

    __ambiguous_inheritance__ = (frobenius.Box,)

    def to_zw(self):
        raise NotImplementedError

    def to_path(self):
        raise NotImplementedError

    def truncated_array(self, input_dims):
        raise NotImplementedError
    
    def array(self):
        raise NotImplementedError
    
    def determine_dimensions(self, input_dims):
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

    def to_path(self, dtype=complex):
        return sum(term.to_path(dtype) for term in self.terms)

    def eval(self, n_photons=0, permanent=None, dtype=complex):
        # we need to implement the proper sums of qpath diagrams
        # this is only a temporary solution, so that the grad tests pass
        if permanent is None:
            from optyx.qpath import npperm
            permanent = npperm
        return sum(
            term.to_path(dtype).eval(n_photons, permanent) for term in self.terms
        )

    def grad(self, var, **params):
        """Gradient with respect to :code:`var`."""
        if var not in self.free_symbols:
            return self.sum_factory((), self.dom, self.cod)
        return sum(term.grad(var, **params) for term in self.terms)


class Swap(frobenius.Swap, Box):
    """Swap in optyx diagram"""

    def to_path(self, dtype=complex):
        from optyx.qpath import Matrix
        return Matrix([0, 1, 1, 0], 2, 2)

    def to_zw(self):
        return self

    def determine_dimensions(self, input_dims: list[int]) -> list[int]:
        """Determine the output dimensions based on the input dimensions."""
        return input_dims[::-1]

    def truncated_array(self, input_dims: list[int]) -> np.ndarray:
        return Permutation(self.dom, [1, 0]).truncated_array(input_dims)


class Permutation(Box):
    """Permute wires in an optyx diagram"""

    def __init__(self, dom: Ty,
                 permutation: list[int],
                 is_dagger: bool = False):
        """
        Args:
            dom: The input type
            permutation: List of indices representing the permutation.
                         Each entry indicates where the
                         corresponding input goes in the output.
        """
        assert len(permutation) == len(dom)

        cod = Ty.tensor(*[dom[i] for i in permutation])
        self.is_dagger = is_dagger
        super().__init__(str(permutation), dom, cod)
        self.permutation = permutation

    def truncated_array(self, input_dims: list[int]) -> np.ndarray:
        """Create an array that permutes the occupation
        numbers based on the input dimensions."""

        input_total_dim = int(np.prod(input_dims))

        perm_matrix = np.zeros((input_total_dim, input_total_dim),
                               dtype=complex)

        output_dims = [input_dims[self.permutation[i]]
                       for i in range(len(self.permutation))]

        for input_index in np.ndindex(*input_dims):
            permuted_index = tuple(input_index[self.permutation[i]]
                                   for i in range(len(self.permutation)))
            input_flat_index = np.ravel_multi_index(input_index, input_dims)
            permuted_flat_index = np.ravel_multi_index(permuted_index,
                                                       output_dims)
            perm_matrix[permuted_flat_index, input_flat_index] = 1

        return perm_matrix.T

    def determine_dimensions(self, input_dims: list[int]) -> list[int]:
        """Determine the output dimensions based on the permutation."""
        return [input_dims[i] for i in self.permutation]

    def dagger(self) -> Diagram:
        n = len(self.permutation)
        inverse_permutation = [0] * n
        for i, j in enumerate(self.permutation):
            inverse_permutation[j] = i

        return Permutation(int(np.sum(self.dom.inside)),
                    int(np.sum(self.cod.inside)),
                    inverse_permutation, 
                    not self.is_dagger)
    
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
        super().__init__(name=f"scalar", dom=Mode(0), cod=Mode(0), data=self.scalar)

    @property
    def array(self):
        return np.array([self.scalar], dtype=complex)

    def __str__(self):
        return f"scalar({format_number(self.data)})"

    def to_path(self, dtype=complex):
        from optyx.qpath import Matrix
        return Matrix[dtype]([], 0, 0, scalar=self.scalar)

    def dagger(self) -> Diagram:
        return Scalar(self.scalar.conjugate())

    def subs(self, *args):
        data = rsubs(self.scalar, *args)
        return Scalar(data)

    def grad(self, var, **params):
        if var not in self.free_symbols:
            return Sum((), self.dom, self.cod)
        return Scalar(self.scalar.diff(var))

    def lambdify(self, *symbols, **kwargs):
        from sympy import lambdify

        return lambda *xs: type(self)(
            lambdify(symbols, self.scalar, **kwargs)(*xs)
        )

    def truncated_array(self, _: list[int]) -> np.ndarray[complex]:
        return self.array
    
    def determine_dimensions(self, _: list[int]) -> list[int]:
        return [1]

Diagram.swap_factory = Swap
Diagram.swap = Swap
Diagram.sum_factory = Sum
Id = Diagram.id