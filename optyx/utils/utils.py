"""
Utility functions which are used in the package.

.. admonition:: Functions
    .. autosummary::
        :template: function.rst
        :nosignatures:
        :toctree:
"""

import numpy as np
from typing import (
    NamedTuple,
    Tuple,
    List
)
from numbers import Number


def _build_w_layer(n_nonzero_counts, dagger=False):
    # pylint: disable=import-outside-toplevel
    from optyx.core import zw
    layer = zw.Id(0)
    for count in n_nonzero_counts:
        if count > 1:
            w_gate = zw.W(count)
            layer @= w_gate.dagger() if dagger else w_gate
        elif count == 1:
            layer @= zw.Id(1)
    return layer


def matrix_to_zw(U):
    # pylint: disable=import-outside-toplevel
    from optyx.core import zw
    from sympy import Expr

    n = U.shape[0]
    diagram = zw.Id(0)

    # initial W layer
    if isinstance(U[0, 0], Expr):
        n_cols_nonzero = [U.shape[1]]*n
    else:
        n_cols_nonzero = np.abs(np.sign(U)).sum(axis=1).astype(int)
    diagram @= _build_w_layer(n_cols_nonzero, dagger=False)

    # endomorphism layer
    endo_layer = zw.Id(0)
    rows, cols = np.nonzero(U)
    for r, c in zip(rows, cols):
        endo_layer @= zw.Endo(U[r, c])

    diagram >>= endo_layer

    # permutation
    nonzero_indices = np.nonzero(U)
    row_indices = nonzero_indices[0]
    col_indices = nonzero_indices[1]
    sorted_indices = np.lexsort((row_indices, col_indices))
    sorted_rows = row_indices[sorted_indices]
    sorted_cols = col_indices[sorted_indices]

    swap_list = []
    for r, c in zip(sorted_rows, sorted_cols):
        swap_list.append(int(n * r + c))

    if not isinstance(U[0, 0], Expr):
        n_s_output_flat = np.abs(np.sign(U)).flatten()
        adjusted_swap_list = []
        for idx in swap_list:
            sum_missing = np.abs(np.array(n_s_output_flat)[:idx] - 1).sum()
            adjusted_swap_list.append(int(idx - sum_missing))
    else:
        adjusted_swap_list = swap_list

    if adjusted_swap_list:
        diagram = diagram.permute(*adjusted_swap_list)

    # W-dagger layer
    if isinstance(U[0, 0], Expr):
        n_rows_nonzero = [U.shape[0]]*U.shape[1]
    else:
        n_rows_nonzero = np.abs(np.sign(U)).sum(axis=0).astype(int)
    diagram >>= _build_w_layer(n_rows_nonzero, dagger=True)

    return diagram


def occupation_numbers(n_photons, m_modes):
    """
    Returns vectors of occupation numbers for n_photons in m_modes.

    Example
    -------
    >>> occupation_numbers(3, 2)
    [(3, 0), (2, 1), (1, 2), (0, 3)]
    >>> occupation_numbers(2, 3)
    [(2, 0, 0), (1, 1, 0), (1, 0, 1), (0, 2, 0), (0, 1, 1), (0, 0, 2)]
    """
    if not n_photons:
        return [m_modes * (0,)]
    if not m_modes:
        raise ValueError(f"Can't put {n_photons} photons in zero modes!")
    if m_modes == 1:
        return [(n_photons,)]
    return [
        (head,) + tail
        for head in range(n_photons, -1, -1)
        for tail in occupation_numbers(n_photons - head, m_modes - 1)
    ]


def multinomial(lst: list) -> int:
    """Returns the multinomial coefficient for a given list of numbers"""
    # https://stackoverflow.com/questions/46374185/does-python-have-a-function-which-computes-multinomial-coefficients
    res, i = 1, sum(lst)
    i0 = lst.index(max(lst))
    for a in lst[:i0] + lst[i0 + 1:]:
        for j in range(1, a + 1):
            res *= i
            res //= j
            i -= 1
    return res


def compare_arrays_of_different_sizes(
    array_1: list | np.ndarray, array_2: list | np.ndarray, tol: float = 1e-08
) -> bool:
    """ZW diagrams which are equal in infinite dimensions
    might be intrepreted as arrays of different dimensions
    if we truncate them to a finite number of dimensions"""

    # See https://stackoverflow.com/questions/46042469/compare-two-arrays-with-different-size-python-numpy  # noqa: E501
    a, b = np.array(array_1).flatten(), np.array(array_2).flatten()
    n = min(len(a), len(b))
    return np.flatnonzero(np.abs(a[:n] - b[:n]) > tol).size == 0


def basis_vector_from_kets(
    indices: list | np.ndarray, max_index_sizes: list | np.ndarray
):
    """Each index from indices specifies the index
    of a "1" in a state basis vector (the occupation number)
    - max_index_sizes specifies the maximum index size (not the maximum index)
    """

    if any(i >= j for i, j in zip(indices, max_index_sizes)):
        raise ValueError(
            "Each index must be smaller than "
            "the corresponding max index size"
        )

    j = 0
    for k, i_k in enumerate(indices):
        j += i_k * (np.prod(np.array(max_index_sizes[k + 1:]), dtype=int))
    return j


def amplitudes_2_tensor(perceval_result, input_occ, output_occ):
    # pylint: disable=import-outside-toplevel
    from discopy.tensor import Tensor
    from discopy.frobenius import Dim

    dom_dims = [
        int(max(np.array(input_occ)[:, i]) + 1)
        for i in range(len(input_occ[0]))
    ]
    cod_dims = [
        int(max(np.array(output_occ)[:, i]) + 1)
        for i in range(len(output_occ[0]))
    ]

    tensor_result_array = np.zeros(
        (int(np.prod(dom_dims)), int(np.prod(cod_dims))), dtype=complex
    )

    for i, o in enumerate(input_occ):
        for j, o_out in enumerate(output_occ):
            i_basis = basis_vector_from_kets(o, dom_dims)
            j_basis = basis_vector_from_kets(o_out, cod_dims)
            tensor_result_array[i_basis, j_basis] = perceval_result[i, j]
    return Tensor(tensor_result_array, Dim(*dom_dims), Dim(*cod_dims))


def tensor_2_amplitudes(
    tn_diagram,
    n_photons_out,
) -> np.ndarray:
    """Convert the prob output of the tensor
    network to the perceval prob output"""
    # pylint: disable=import-outside-toplevel
    import warnings

    output = tn_diagram.eval().array.flatten()
    idxs = list(occupation_numbers(n_photons_out, len(tn_diagram.cod)))
    cod = list(tn_diagram.cod.inside)

    if sum(cod) < n_photons_out:
        warnings.warn(
            "It is likely that the Tensor diagram has been "
            "truncated with dimensions which are "
            "too low for the n_photons_out. "
            "The results might be incorrect."
        )

    res = []
    for i in idxs:
        try:
            basis = basis_vector_from_kets(i, cod)
            res.append(output[basis])
        except ValueError:
            res.append(0.0)
            warnings.warn(
                f"The basis vector {i} is out of bounds of "
                f"the codomain {cod}. Setting to 0."
            )

    return np.array(res)


def explode_channel(
    kraus,
    channel_class=None,
    circuit_class=None,
):
    # pylint: disable=import-outside-toplevel
    from optyx.core.channel import Channel, Ty, Diagram

    if channel_class is None:
        channel_class = Channel
    if circuit_class is None:
        circuit_class = Diagram

    arrows = []
    for layer in kraus:
        generator = layer.inside[0][1]
        channel = channel_class(
            generator.name,
            generator,
        )

        arrows.append(
            Ty.from_optyx(layer.inside[0][0]) @
            channel @
            Ty.from_optyx(layer.inside[0][2])
        )

    if len(arrows) == 0:
        return channel_class("Id", kraus)

    return channel_class.then(*arrows)


def calculate_num_creations_selections(dgrm) -> tuple:
    """Calculate the number of creations and selections in the diagram"""
    # pylint: disable=import-outside-toplevel
    from optyx.core import diagram, zw

    n_selections = 0
    n_creations = 0

    if not isinstance(dgrm, diagram.Sum):
        for box, _ in zip(dgrm.boxes, dgrm.offsets):
            if isinstance(box, zw.Create):
                n_creations += sum(box.photons)
            elif isinstance(box, zw.Select):
                n_selections += sum(box.photons)
    else:
        arr_selections_creations = []
        for term in dgrm:
            arr_selections_creations.append(
                calculate_num_creations_selections(term)
            )
        n_selections = max(i[0] for i in arr_selections_creations)
        n_creations = max(i[1] for i in arr_selections_creations)
    return n_selections, n_creations


def filter_occupation_numbers(
    allowed_occupation_configurations: list[list[int]],
    input_dims: list[int],
) -> list[list[int]]:
    """Filter the occupation numbers based on the input dimensions"""
    return [
        config
        for config in allowed_occupation_configurations
        if all(
            list(config[i] <= input_dims[i] for i in range(len(input_dims)))
        )
    ]


def invert_perm(p):
    q = [0] * len(p)
    for out, inn in enumerate(p):
        q[inn] = out
    return q


class BasisTransition(NamedTuple):
    """
    A single non-zero transition emitted by `truncation_specification`.

    For a fixed input basis index `input_index`, a box yields zero or more
    `Transition` objects, each describing one reachable output index and its
    amplitude. The caller (`Box.truncation`) writes `amp` into the result
    tensor at index `out + input_index` (output indices first, then input).

    Attributes
    ----------
    out : Tuple[int, ...]
        Output multi-index in the box's codomain for this transition.
        Must satisfy: len(out) == len(self.cod) and
        0 <= out[i] < output_dims[i] for all i.
    amp : Union[complex, float, int, sympy.Expr]
        The complex (or symbolic) amplitude associated with this output.
    """
    out: Tuple[int, ...]
    amp: Number


def preprocess_quimb_tensors_safe(tn, epsilon=1e-12, value_limit=1e10):
    for t in tn:
        data = t.data

        data = np.array(data, copy=True)

        if data.dtype.kind in {'i', 'u'}:
            t.modify(data=data.astype('complex128'))
            continue

        if data.ndim == 2 and np.linalg.matrix_rank(data) < min(data.shape):
            data += np.random.normal(0, epsilon, size=data.shape)

        if np.any(data == 0):
            data[data == 0] = epsilon

        if np.max(np.abs(data)) > value_limit:
            data = np.clip(data, -value_limit, value_limit)

        t.modify(data=data)

    return tn


def update_connections(
    wires_in_light_cone: List[bool],
    previous_left_offset: int,
    previous_box,
    previous_right_offset: int,
) -> List[bool]:
    from optyx.core.diagram import mode
    """
    Replace the previous box's cod segment in the light-cone by a dom-length
    segment that is either all True (if connected) or all False.
    This pulls the cone one layer backward.
    """
    connected = is_previous_box_connected_to_current_box(
        wires_in_light_cone,
        previous_left_offset,
        len(previous_box.cod),
        previous_right_offset,
    )

    start = previous_left_offset
    end = len(wires_in_light_cone) - previous_right_offset

    return (
        wires_in_light_cone[:start]
        + [connected if t == mode else False for t in previous_box.dom]
        + wires_in_light_cone[end:]
    )


def calculate_right_offset(
        total_wires: int,
        left_offset: int,
        span_len: int
) -> int:
    """Right offset = number of wires to the right of a span."""
    return total_wires - span_len - left_offset


def is_previous_box_connected_to_current_box(
    wires_in_light_cone: List[bool],
    previous_left_offset: int,
    previous_box_cod_len: int,
    previous_right_offset: int,
) -> bool:
    """
    Do the current light-cone wires intersect the COD of the previous box?
    """
    mask = (
        [False] * previous_left_offset
        + [True] * previous_box_cod_len
        + [False] * previous_right_offset
    )
    # lengths should match by construction
    assert len(mask) == len(wires_in_light_cone), (
        (f"Mask/wires length mismatch: {len(mask)}"
         + f" != {len(wires_in_light_cone)}")
    )

    return any(w and m for w, m in zip(wires_in_light_cone, mask))


def get_previous_box_cod_index_in_light_cone(
    wires_in_light_cone: List[bool],
    previous_left_offset: int,
    previous_box_cod_len: int,
    previous_right_offset: int,
) -> List[int]:
    """
    Get the indices of the cod of the previous box
    that are in the current light-cone.
    """
    mask = (
        [False] * previous_left_offset
        + [True] * previous_box_cod_len
        + [False] * previous_right_offset
    )
    # lengths should match by construction
    assert len(mask) == len(wires_in_light_cone), (
        (f"Mask/wires length mismatch: {len(mask)}"
         + f" != {len(wires_in_light_cone)}")
    )

    return [
        i - previous_left_offset for i, (w, m) in
        enumerate(zip(wires_in_light_cone, mask))
        if w and m
    ]


def get_max_dim_for_box(
    left_offset: int,
    box,
    right_offset: int,
    input_dims: List[int],
    prev_layers,
):
    from optyx.core.diagram import Swap, mode
    from optyx.core.zw import Create, Endo, Divide, Multiply

    if (
        len(box.dom) == 0 or
        isinstance(box, (Swap, Endo, Divide, Multiply))
    ):
        return 1e20

    dim_for_box = 0

    # light-cone at the current layer [connected] * len(previous_box.dom)inputs
    wires_in_light_cone: List[bool] = (
        [False] * left_offset
        + [True if t == mode else False for t in box.dom]
        + [False] * right_offset
    )

    # walk previous layers from nearest to farthest
    for previous_left_offset, previous_box in prev_layers[::-1]:
        total = len(wires_in_light_cone)
        cod_len = len(previous_box.cod)

        # Clamp the left offset so the (left, cod_len) span fits this frame
        # this guarantees previous_right_offset>= 0 and len(mask) == total
        max_left = max(0, total - cod_len)
        adj_left = previous_left_offset
        if adj_left < 0:
            adj_left = 0
        elif adj_left > max_left:
            adj_left = max_left

        previous_right_offset = \
            calculate_right_offset(total, adj_left, cod_len)

        if is_previous_box_connected_to_current_box(
            wires_in_light_cone,
            adj_left,
            cod_len,
            previous_right_offset,
        ):
            if isinstance(previous_box, Create):
                idxs = get_previous_box_cod_index_in_light_cone(
                    wires_in_light_cone,
                    adj_left,
                    cod_len,
                    previous_right_offset,
                )
                if idxs:
                    dim_for_box += sum(previous_box.photons[i] for i in idxs)

        if isinstance(previous_box, Swap):
            wires_in_light_cone = (
                wires_in_light_cone[:adj_left]
                + [wires_in_light_cone[adj_left + 1]]
                + [wires_in_light_cone[adj_left]]
                + wires_in_light_cone[adj_left + 2:]
            )
        else:
            wires_in_light_cone = update_connections(
                wires_in_light_cone,
                adj_left,
                previous_box,
                previous_right_offset,
            )
    dim_for_box += sum(2 * dim for wire, dim in
                       zip(wires_in_light_cone, input_dims) if wire) + 1
    return max(dim_for_box, 2)


def is_diagram_LO(diagram):
    from optyx.core.zw import LO_ELEMENTS
    if is_identity(diagram):
        return True

    for box in diagram.boxes:
        if is_identity(box):
            continue
        if not isinstance(box, LO_ELEMENTS):
            return False

    return True


is_identity = lambda box: (len(box.boxes) == 0 and  # noqa: E731
                           len(box.offsets) == 0)
