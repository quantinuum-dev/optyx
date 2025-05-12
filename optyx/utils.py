"""
Utility functions which are used in the package.

.. admonition:: Functions
    .. autosummary::
        :template: function.rst
        :nosignatures:
        :toctree:
"""

import numpy as np

def _build_w_layer(n_nonzero_counts, dagger=False):
    from optyx import zw

    layer = zw.Id(0)
    for count in n_nonzero_counts:
        if count > 1:
            w_gate = zw.W(count)
            layer @= w_gate.dagger() if dagger else w_gate
        elif count == 1:
            layer @= zw.Id(1)
    return layer


def matrix_to_zw(U):
    from optyx.diagram import zw

    n = U.shape[0]
    diagram = zw.Id(0)

    # initial W layer
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
        swap_list.append(n * r + c)

    n_s_output_flat = np.abs(np.sign(U)).flatten()
    adjusted_swap_list = []
    for idx in swap_list:
        sum_missing = np.abs(np.array(n_s_output_flat)[:idx] - 1).sum()
        adjusted_swap_list.append(int(idx - sum_missing))

    if adjusted_swap_list:
        diagram = diagram.permute(*adjusted_swap_list)

    # W-dagger layer
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


def modify_io_dims_against_max_dim(input_dims, output_dims, max_dim):
    """Modify the input and output dimensions against the maximum dimension"""
    if input_dims is not None:
        input_dims = [max_dim if i > max_dim else i for i in input_dims]
    if output_dims is not None:
        output_dims = [max_dim if i > max_dim else i for i in output_dims]
    return input_dims, output_dims


def amplitudes_2_tensor(perceval_result, input_occ, output_occ):

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
    from optyx.diagram.channel import Channel, Ty, Circuit

    if channel_class is None:
        channel_class = Channel
    if circuit_class is None:
        circuit_class = Circuit

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