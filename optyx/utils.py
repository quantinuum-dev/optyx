"""
Utility functions which are used in the package.

.. admonition:: Functions
    .. autosummary::
        :template: function.rst
        :nosignatures:
        :toctree:
"""

import numpy as np


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


def compare_arrays_of_different_sizes(array_1: list | np.ndarray,
                                      array_2: list | np.ndarray) -> bool:
    """ZW diagrams which are equal in infinite dimensions
    might be intrepreted as arrays of different dimensions
    if we truncate them to a finite number of dimensions"""
    if not isinstance(array_1, np.ndarray):
        array_1 = np.array([array_1])
    if not isinstance(array_2, np.ndarray):
        array_2 = np.array([array_2])
    if len(array_1.flatten()) < len(array_2.flatten()):
        ax_0 = array_1.shape[0]
        if len(array_1.shape) == 1:
            array_2 = array_2[:ax_0]
        else:
            ax_1 = array_1.shape[1]
            array_2 = array_2[:ax_0, :ax_1]
    elif len(array_1.flatten()) > len(array_2.flatten()):
        ax_0 = array_2.shape[0]
        if len(array_2.shape) == 1:
            array_1 = array_1[:ax_0]
        else:
            ax_1 = array_2.shape[1]
            array_1 = array_1[:ax_0, :ax_1]
    else:
        pass
    return np.allclose(array_1, array_2)


def get_index_from_list(indices: list | np.ndarray,
                        max_index_sizes: int = 0):
    """Each index from indices specifies the index
    of a "1" in a state basis vector (the occupation number),
    max_sum is the length of the vector minus 1 (the max occupation number),
    the function returns the index of the "1"
    if we tensor the state basis vectors
    """

    j = 0
    for k, i_k in enumerate(indices):
        j += i_k * (
            np.prod(np.array(max_index_sizes[k+1:]), dtype=int)
        )
    return j
