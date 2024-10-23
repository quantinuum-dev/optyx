from optyx.zw import *
import pytest

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

test_pairs = [(i, j) for i in range(0, 10) for j in range(0, 10, 2)]

@pytest.mark.parametrize("i, j", test_pairs)
def test_swap(i, j):
    assert np.allclose(kron_truncated_array_swap([i, j]), Swap(mode, mode).truncated_array([i, j]))

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

test_pairs = []
test_pairs = [[i] for i in range(0, 10, 2)]
test_pairs += [[i, j] for i in range(0, 10, 2) for j in range(0, 10, 2)]
test_pairs += [[i, j, k] for i in range(0, 10, 2) for j in range(0, 10, 2) for k in range(0, 10, 2)]

@pytest.mark.parametrize("comb", test_pairs)
def test_Z(comb):
    assert np.allclose(kron_truncated_array_Z(Z(lambda i : np.sqrt(i), len(comb), 3), comb), 
                       Z(lambda i : np.sqrt(i), len(comb), 3).truncated_array(comb))