from optyx.zw import *
import pytest
import itertools
import random

@pytest.mark.skip(reason="Helper function for testing")
def kron_truncated_array_swap(input_dims: list[int]) -> np.ndarray[complex]:
    input_total_dim = (input_dims[0]) * (input_dims[1])   # Total input dimension

    swap = np.zeros((input_total_dim, input_total_dim), dtype=complex)

    for i in range(input_dims[0]):
        for j in range(input_dims[1]):
            i_ = np.eye(input_dims[0])[i]
            j_ = np.eye(input_dims[1])[j]

            ij = np.kron(i_, j_)
            ji = np.kron(j_, i_)
            swap += np.outer(ij, ji)
    return swap

@pytest.mark.skip(reason="Helper function for testing")
class Swap_basic(monoidal.Box, Diagram):
    """Swap in a ZW diagram"""

    def __init__(self, cod=2, dom=2):
        super().__init__("SWAP", PRO(cod), PRO(dom))

    # create an array like in 2306.02114
    def truncated_array(self, input_dims: list[int]) -> np.ndarray[complex]:
        """Create an array that swaps the occupation
        numbers based on the input dimensions."""

        input_total_dim = (input_dims[0]) * (input_dims[1])

        swap = np.zeros((input_total_dim, input_total_dim), dtype=complex)

        # Iterate over the dimensions for both wires
        for i in range(input_dims[1]):
            for j in range(input_dims[0]):
                swap[i * (input_dims[0]) + j, j * (input_dims[1]) + i] = 1

        return swap.T

    def dagger(self) -> Diagram:
        return Swap()

@pytest.mark.skip(reason="Helper function for testing")
def permutation_to_swaps(p: list[int]) -> list[tuple[int, int]]:
    swaps = []
    p = p[:]
    
    for _ in range(len(p)):
        for j in range(len(p) - 1):
            if p[j] > p[j + 1]:
                swaps.append((j, j + 1))
                p[j], p[j + 1] = p[j + 1], p[j]
    swaps.reverse()
    return swaps

@pytest.mark.skip(reason="Helper function for testing")
def from_perms_get_swaps(permutation: list[int], dims: list[int]):
    zw_diagram = np.eye(np.prod(dims))
    swaps = permutation_to_swaps(permutation)
    perm_for_dims = np.arange(len(permutation))
    for i, _ in swaps:

        permuted_dims = [dims[perm_for_dims[i]] for i in range(len(permutation))]
        left = Id(i).truncated_array(permuted_dims[:i])
        right = Id(len(permutation) - i - 2).truncated_array(permuted_dims[i + 2:])
        swap = Swap_basic(2, 2).truncated_array(permuted_dims[i:i + 2]).T
        layer = np.kron(np.kron(left, swap), right)
        zw_diagram = layer @ zw_diagram

        perm_for_dims[i], perm_for_dims[i + 1] = perm_for_dims[i + 1], perm_for_dims[i]
        
    return zw_diagram.T

@pytest.mark.skip(reason="Helper function for testing")
def manual_swap(p, d):
    arrays_of_dims = [np.arange(dim) for dim in d]


    all_combs = itertools.product(*arrays_of_dims)

    array = np.zeros((np.prod(d), np.prod(d)), dtype=complex)
    for comb in list(all_combs):
        ket = 1
        for j, i in enumerate(comb):
            ket = np.kron(ket, np.eye(d[j])[i])

        permuted_comb = tuple(comb[p[i]] for i in range(len(p)))
        permuted_dims = tuple(d[p[i]] for i in range(len(p)))
        bra = 1
        for j, i in enumerate(permuted_comb):
            bra = np.kron(bra, np.eye(permuted_dims[j])[i])

        outer = np.outer(ket, bra)
        array = np.add(array, outer)

    return array


test_pairs = [(i, j) for i in range(0, 10) for j in range(0, 10, 2)]

@pytest.mark.parametrize("i, j", test_pairs)
def test_swap(i, j):
    assert np.allclose(kron_truncated_array_swap([i, j]), Swap().truncated_array([i, j]))

@pytest.mark.skip(reason="Helper function for testing")
def kron_truncated_array_W(diagram, input_dims: list[int]) -> np.ndarray[complex]:
    if not diagram.is_dagger:
        max_occupation_num = input_dims[0]

        total_map = np.zeros(
            ((max_occupation_num) ** diagram.n_legs, max_occupation_num),
            dtype=complex,
        )   

        for n in range(max_occupation_num):

            # get all allowed occupation configurations for n photons
            # (symmetric Fock space basis states)
            allowed_occupation_configurations = occupation_numbers(
                n, diagram.n_legs
            )

            for configuration in allowed_occupation_configurations:
                coef = np.sqrt(multinomial(configuration))
                vec = 1
                for i, d in enumerate(configuration):
                    vec = np.kron(vec, np.eye(max_occupation_num)[d])
                total_map += coef * np.outer(vec, np.eye(max_occupation_num)[n])
            
        return total_map.T
    else:
        max_occupation_num = np.sum(np.array(input_dims) - 1) + 1

        total_map = np.zeros(
            (np.prod(np.array(input_dims)), max_occupation_num),
            dtype=complex,    
        )

        for n in range(max_occupation_num):

            # get all allowed occupation configurations for n photons
            # (symmetric Fock space basis states)
            allowed_occupation_configurations = occupation_numbers(
                n, diagram.n_legs
            )
            #print(allowed_occupation_configurations)
            allowed_occupation_configurations = filter_occupation_numbers(
                allowed_occupation_configurations, np.array(input_dims) - 1
            )
            for configuration in allowed_occupation_configurations:

                # get the coefficient for the configuration
                coef = np.sqrt(multinomial(configuration))

                # find idx of the matrix where to put the coefficient
                vec = 1
                for i, d in enumerate(configuration):
                    vec = np.kron(vec, np.eye(input_dims[i])[d])
                        
                total_map += coef * np.outer(vec, np.eye(max_occupation_num)[n])

        return total_map
    
test_pairs = [[i] for i in range(1, 10, 2)]
test_pairs += [[i, j] for i in range(1, 10, 2) for j in range(1, 10, 2)]
test_pairs += [[i, j, k] for i in range(1, 10, 2) for j in range(1, 10, 2) for k in range(1, 10, 2)]

@pytest.mark.parametrize("comb", test_pairs)
def test_W(comb):
    assert np.allclose(kron_truncated_array_W(W(len(comb)), comb), W(len(comb)).truncated_array(comb))

@pytest.mark.parametrize("comb", test_pairs)
def test_W_dagger(comb):
    assert np.allclose(kron_truncated_array_W(W(len(comb)).dagger(), comb), W(len(comb)).dagger().truncated_array(comb))

@pytest.mark.skip(reason="Helper function for testing")
def kron_truncated_array_Z(diagram, input_dims: list[int]) -> np.ndarray[complex]:
    max_occupation_num = min(input_dims)

    result_matrix = np.zeros(
        (max_occupation_num**diagram.legs_out, np.prod(np.array(input_dims))), dtype=complex
    )

    if diagram.legs_in == 0 and diagram.legs_out == 0:
        if not isinstance(diagram.amplitudes, IndexableAmplitudes):
            return np.array([diagram.amplitudes], dtype=complex)
        return np.array([diagram.amplitudes[0]], dtype=complex)  

    for i in range(max_occupation_num):
        vec_in = 1
        for j in range(diagram.legs_in):
            vec_in = np.kron(vec_in, np.eye(input_dims[j])[i])
        vec_out = 1
        for j in range(diagram.legs_out):
            vec_out = np.kron(vec_out, np.eye(max_occupation_num)[i])
        if not isinstance(diagram.amplitudes, IndexableAmplitudes):
            if i >= len(diagram.amplitudes):
                pass
            else:
                result_matrix += np.outer(vec_out, vec_in) * diagram.amplitudes[i]
        else:
            result_matrix += np.outer(vec_out, vec_in) * diagram.amplitudes[i]
    return result_matrix

test_pairs = [[i] for i in range(0, 10, 2)]
test_pairs += [[i, j] for i in range(0, 10, 2) for j in range(0, 10, 2)]
test_pairs += [[i, j, k] for i in range(0, 10, 2) for j in range(0, 10, 2) for k in range(0, 10, 2)]

@pytest.mark.parametrize("comb", test_pairs)
def test_Z(comb):
    assert np.allclose(kron_truncated_array_Z(Z(lambda i : np.sqrt(i), len(comb), 3), comb), 
                       Z(lambda i : np.sqrt(i), len(comb), 3).truncated_array(comb))
    

perms = [np.random.permutation(random.randint(2, 5)).tolist() for _ in range(50)]
@pytest.mark.parametrize("p", perms)
def test_bubble_sort_permutations(p):
    p_start = np.sort(p)
    for i, j in permutation_to_swaps(p):
        p_start[i], p_start[j] = p_start[j], p_start[i]
    assert np.allclose(p, p_start)  


perms = [np.random.permutation(random.randint(2, 5)).tolist() for _ in range(20)]
dims = [[random.randint(1, 4) for _ in range(len(p))] for p in perms]

@pytest.mark.parametrize("p, d", zip(perms, dims))
def test_from_perms_get_swaps(p, d):
    perm = Swap(len(p), len(p), p)
    assert np.allclose(perm.truncated_array(d), from_perms_get_swaps(p, d))    

@pytest.mark.parametrize("p, d", zip(perms, dims))
def test_manual_swap(p, d):
    assert np.allclose(manual_swap(p, d), from_perms_get_swaps(p, d))