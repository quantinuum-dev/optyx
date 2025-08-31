import math

from optyx.core.zw import *
from optyx.core.diagram import mode, DualRail, EmbeddingTensor, Swap, Diagram, Mode, Spider, Scalar
from optyx.utils.utils import compare_arrays_of_different_sizes, calculate_num_creations_selections
import optyx.core.zx as zx
from optyx import photonic
import itertools
import pytest
import numpy as np
# Axioms

legs_a_in = range(1, 3)
legs_b_in = range(1, 3)
legs_a_out = range(1, 3)
legs_b_out = range(1, 3)
legs_between = range(1, 3)
max_dim = [None, 2, 4, 6]

# a set of arbitrary functions of i
fs = [
    lambda i: i,
    lambda i: math.factorial(i),
    lambda i: np.exp(i)
]

# get all combinations of legs etc
fs_legs_combinations = list(
    itertools.product(
        fs, legs_a_in, legs_b_in, legs_a_out, legs_b_out, legs_between, max_dim
    )
)


@pytest.mark.parametrize(
    "fs, legs_a_in, legs_b_in, legs_a_out, legs_b_out, legs_between, max_dim",
    fs_legs_combinations,
)
def test_spider_fusion(
    fs: int,
    legs_a_in: int,
    legs_b_in: int,
    legs_a_out: int,
    legs_b_out: int,
    legs_between: int,
    max_dim: int
):

    S1_infty_l = ZBox(legs_a_in, legs_a_out + legs_between, fs) @ Id(legs_b_in)

    S1_infty_l = S1_infty_l >> Id(legs_a_out + legs_between + legs_b_in)

    S1_infty_l = S1_infty_l >> Id(legs_a_out) @ ZBox(legs_b_in + legs_between, legs_b_out, fs)

    fn_mult = lambda i: fs(i) * fs(i)

    S1_infty_r = ZBox(legs_a_in + legs_b_in, legs_a_out + legs_b_out, fn_mult)

    assert compare_arrays_of_different_sizes(
        S1_infty_l.to_tensor(max_dim = max_dim).eval().array,
        S1_infty_r.to_tensor(max_dim = max_dim).eval().array,
    )


@pytest.mark.parametrize("legs", range(1, 3))
def test_Z_conj(legs: int):
    Z_conj_l = ZBox(legs, legs, lambda i: 1j).dagger()
    Z_conj_r = ZBox(legs, legs, lambda i: -1j)

    assert compare_arrays_of_different_sizes(
        Z_conj_l.to_tensor().eval().array,
        Z_conj_r.to_tensor().eval().array,
    )


def test_IndexableAmplitudes():
    i_a_1 = IndexableAmplitudes(lambda i: 1j)
    i_a_2 = IndexableAmplitudes(lambda i: 1.0j)

    assert i_a_1 == i_a_2


def test_bSym():
    bSym_l = W(2)
    bSym_r = W(2) >> Swap(mode, mode)

    assert compare_arrays_of_different_sizes(
        bSym_l.to_tensor().eval().array,
        bSym_r.to_tensor().eval().array,
    )


def test_bSym_input_dims():
    bSym_l = W(2)
    bSym_r = W(2) >> Swap(mode, mode)

    assert compare_arrays_of_different_sizes(
        bSym_l.to_tensor([5]).eval().array,
        bSym_r.to_tensor([5]).eval().array,
    )


def test_bAso():
    bAso_l = W(2) >> W(2) @ Id(1)
    bAso_r = W(2) >> Id(1) @ W(2)

    assert compare_arrays_of_different_sizes(
        bAso_l.to_tensor().eval().array,
        bAso_r.to_tensor().eval().array,
    )


def test_doubleSWAP():
    swap_l = Swap(mode, mode) >> Swap(mode, mode)
    swap_r = Id(2)

    assert compare_arrays_of_different_sizes(
        swap_l.to_tensor().eval().array,
        swap_r.to_tensor().eval().array,
    )


def test_SWAP_dagger_SWAP():
    swap_l = Swap(mode, mode).dagger()
    swap_r = Swap(mode, mode)

    assert compare_arrays_of_different_sizes(
        swap_l.to_tensor().eval().array,
        swap_r.to_tensor().eval().array,
    )


@pytest.mark.parametrize("k", range(1, 3))
def test_Id_eq(k: int):
    id_l = Id(k)
    id_r = Id(k).dagger()

    assert id_l == id_r


def test_permutation_dagger():
    perm = Diagram.permutation([1, 0], Mode(2))

    assert compare_arrays_of_different_sizes(
        perm.to_tensor().eval().array,
        perm.dagger().to_tensor().eval().array,

    )


def test_permutation_path_dagger():
    perm = Diagram.permutation([1, 0], Mode(2))

    assert compare_arrays_of_different_sizes(
        perm.to_path().array,
        perm.dagger().to_path().array,
    )

@pytest.mark.parametrize("legs", range(1, 3))
def test_Z_eq_IndexableAmplitudes(legs: int):
    Z_l = ZBox(legs, legs, lambda i: 1)
    Z_r = ZBox(legs, legs, lambda i: 1).dagger()

    assert Z_l == Z_r


@pytest.mark.parametrize("legs", range(1, 3))
def test_Z_eq(legs: int):
    Z_l = ZBox(legs, legs, [1, 1])
    Z_r = ZBox(legs, legs, [1, 1]).dagger()

    assert Z_l == Z_r


@pytest.mark.parametrize("k", range(1, 3))
def test_Id_dagger_Id(k: int):
    id_l = Id(k).dagger()
    id_r = Id(k)

    assert compare_arrays_of_different_sizes(
        id_l.to_tensor().eval().array,
        id_r.to_tensor().eval().array,
    )


@pytest.mark.parametrize("max_dim", [None, 3, 4, 6])
def test_bBa(max_dim):
    bBa_l = (
        W(2) @ W(2) >> Id(1) @ Swap(mode, mode) @ Id(1) >> W(2).dagger() @ W(2).dagger()
    )
    bBa_r = W(2).dagger() >> W(2)

    assert compare_arrays_of_different_sizes(
        bBa_l.to_tensor(max_dim=max_dim).eval().array,
        bBa_r.to_tensor(max_dim=max_dim).eval().array,
    )


def test_bId():
    bId_l = W(2) >> Select(0) @ Id(1)
    bId_r = Id(1)

    assert compare_arrays_of_different_sizes(
        bId_l.to_tensor().eval().array,
        bId_r.to_tensor().eval().array,
    )


@pytest.mark.parametrize("max_dim", [None, 2, 4, 6])
def test_bZBA(max_dim):
    from math import factorial

    N = [float(np.sqrt(factorial(i))) for i in range(5)]
    frac_N = [float(1 / np.sqrt(factorial(i))) for i in range(5)]

    bZBA_l = (
            ZBox(1, 2, N) @ ZBox(1, 2, N)
            >> Id(1) @ Swap(mode, mode) @ Id(1)
            >> W(2).dagger() @ W(2).dagger()
            >> Id(1) @ ZBox(1, 1, frac_N)
    )
    bZBA_r = W(2).dagger() >> ZBox(1, 2, lambda i: 1)

    assert compare_arrays_of_different_sizes(
        bZBA_l.to_tensor(max_dim=max_dim).eval().array,
        bZBA_r.to_tensor(max_dim=max_dim).eval().array,
    )

@pytest.mark.parametrize("max_dim", [None, 2, 4, 6])
def test_bZBA_optyx_Spider(max_dim):
    from math import factorial

    N = [float(np.sqrt(factorial(i))) for i in range(5)]
    frac_N = [float(1 / np.sqrt(factorial(i))) for i in range(5)]

    bZBA_l = (
            ZBox(1, 1, N) @ ZBox(1, 1, N)
            >> Spider(1, 2, Mode(1)) @ Spider(1, 2, Mode(1))
            >> Id(1) @ Swap(mode, mode) @ Id(1)
            >> W(2).dagger() @ W(2).dagger()
            >> Id(1) @ ZBox(1, 1, frac_N)
    )
    bZBA_r = W(2).dagger() >> Spider(1, 2, Mode(1))

    assert compare_arrays_of_different_sizes(
        bZBA_l.to_tensor(max_dim=max_dim).eval().array,
        bZBA_r.to_tensor(max_dim=max_dim).eval().array,
    )

def test_K0_infty():
    K0_infty_l = Create(4) >> ZBox(1, 2, lambda i: 1)
    K0_infty_r = Create(4) @ Create(4)

    assert compare_arrays_of_different_sizes(
        K0_infty_l.to_tensor().eval().array,
        K0_infty_r.to_tensor().eval().array,
    )


def test_scalar():
    scalar_l = Create(1) >> ZBox(1, 1, [1, 2]) >> Select(1)
    scalar_r = ZBox(0, 0, [2])

    assert compare_arrays_of_different_sizes(
        scalar_l.to_tensor().eval().array,
        scalar_r.to_tensor().eval().array,
    )


def test_scalar_with_IndexableAmplitudes():
    scalar_l = Create(1) >> ZBox(1, 1, lambda i: i) >> Select(1)
    scalar_r = ZBox(0, 0, lambda i: i + 1)

    assert compare_arrays_of_different_sizes(
        scalar_l.to_tensor().eval().array,
        scalar_r.to_tensor().eval().array,
    )


def test_bone():
    bone_l = Create(1) >> Select(0)
    bone_r = Create(0) >> Select(1)

    assert compare_arrays_of_different_sizes(bone_l.to_tensor().eval().array, [0])
    assert compare_arrays_of_different_sizes([0], bone_r.to_tensor().eval().array)


def test_branching():
    branching_l = Create(1) >> W(2)
    branching_r = Create(1) @ Create(0) + Create(0) @ Create(1)

    assert compare_arrays_of_different_sizes(
        branching_l.to_tensor().eval().array,
        branching_r.to_tensor().eval().array,
    )


@pytest.mark.parametrize("k", [1, 2, 3])
def test_normalisation(k: int):
    from math import factorial

    normalisation_l = Create(k) @ ZBox(0, 0, [np.sqrt(factorial(k))])

    normalisation_r = Id(0)
    for _ in range(k):
        normalisation_r = normalisation_r @ Create(1)

    normalisation_r = normalisation_r >> W(k).dagger()

    assert compare_arrays_of_different_sizes(
        normalisation_l.to_tensor().eval().array,
        normalisation_r.to_tensor().eval().array,
    )


# Some lemmas


@pytest.mark.parametrize("k", [1, 2, 3])
def test_lemma_B6(k: int):

    lemma_B6_l = Create(k) >> ZBox(1, 1, lambda i: i + 1)
    lemma_B6_r = Create(k) @ ZBox(0, 0, [k + 1])

    assert compare_arrays_of_different_sizes(
        lemma_B6_l.to_tensor().eval().array,
        lemma_B6_r.to_tensor().eval().array,
    )


def test_w_eq():
    assert W(2) == W(2).dagger().dagger()


def test_create_eq():
    assert Create(2) != Create(1)


def test_select_eq():
    assert Select(2) != Select(1)


def test_calculate_num_creations_selections():
    d1 = Create(1) + Create(1)
    d2 = Create(1)
    assert calculate_num_creations_selections(
        d1
    ) == calculate_num_creations_selections(d2)
    d3 = Create(1) >> Select(0) @ Create(1)
    assert calculate_num_creations_selections(
        d1
    ) != calculate_num_creations_selections(d3)
    assert calculate_num_creations_selections(
        d2 + d3
    ) == calculate_num_creations_selections(d3)


@pytest.mark.parametrize("max_dim", [None, 2, 4, 6])
def test_lemma_B8(max_dim):
    lemma_B8_l = (
            Create(1)
            >> ZBox(1, 2, [1, 1])
            >> W(2) @ W(2)
            >> Id(1) @ ZBox(2, 0, [1, 1]) @ Id(1)
    )

    lemma_B8_r = Create(1) >> W(2) >> ZBox(1, 2, [1, 1]) @ ZBox(1, 0, [1, 1])

    assert compare_arrays_of_different_sizes(
        lemma_B8_l.to_tensor(max_dim=max_dim).eval().array,
        lemma_B8_r.to_tensor(max_dim=max_dim).eval().array,
    )


@pytest.mark.parametrize("max_dim", [None, 2, 4, 6])
def test_lemma_B7(max_dim):
    lemma_B7_l = Id(1) @ W(2).dagger() >> ZBox(2, 0, lambda i: 1)

    lemma_B7_r = (
            W(2) @ Id(2)
            >> Id(1) @ Id(1) @ Swap(mode, mode)
            >> Id(1) @ Swap(mode, mode) @ Id(1)
            >> ZBox(2, 0, lambda i: 1) @ ZBox(2, 0, lambda i: 1)
    )

    assert compare_arrays_of_different_sizes(
        lemma_B7_l.to_tensor(max_dim=max_dim).eval().array,
        lemma_B7_r.to_tensor(max_dim=max_dim).eval().array,
    )


@pytest.mark.parametrize("max_dim", [None, 3, 4, 6])
def test_prop_54(max_dim):
    prop_54_l = (
            Create(1) @ Id(1)
            >> ZBox(1, 2, lambda i: 1) @ Id(1)
            >> Id(1) @ W(2).dagger()
            >> Id(1) @ W(2)
            >> ZBox(2, 0, lambda i: 1) @ Id(1)
    )

    prop_54_r = (
            Create(1) @ Id(1)
            >> ZBox(1, 2, lambda i: 1) @ Id(1)
            >> Swap(mode, mode) @ Id(1)
            >> W(2) @ W(2) @ W(2)
            >> Id(1) @ ZBox(2, 0, lambda i: 1) @ ZBox(2, 0, lambda i: 1) @ Id(1)
            >> W(2).dagger()
    )

    assert compare_arrays_of_different_sizes(
        prop_54_l.to_tensor(max_dim=max_dim).eval().array,
        prop_54_r.to_tensor(max_dim=max_dim).eval().array,
    )


# Hong-Ou-Mandel
@pytest.mark.parametrize(
    "postselect_and_prob",
    [
        [1, 1, np.array([0])],
        [2, 0, np.array(np.sqrt(2) * 1j / 2)],
        [0, 2, np.array(np.sqrt(2) * 1j / 2)],
    ],
)
def test_hom(postselect_and_prob: list):
    select_1 = postselect_and_prob[0]
    select_2 = postselect_and_prob[1]
    prob = postselect_and_prob[2]

    Zb_i = ZBox(1, 1, lambda i: (np.sin(0.25 * np.pi) * 1j) ** i)
    Zb_1 = ZBox(1, 1, lambda i: (np.cos(0.25 * np.pi)) ** i)

    beam_splitter = (
        W(2) @ W(2)
        >> Zb_i @ Zb_1 @ Zb_1 @ Zb_i
        >> Id(1) @ Swap(mode, mode) @ Id(1)
        >> W(2).dagger() @ W(2).dagger()
    )

    Hong_Ou_Mandel = (
        Create(1) @ Create(1)
        >> beam_splitter
        >> Select(select_1) @ Select(select_2)
    )

    assert compare_arrays_of_different_sizes(
        Hong_Ou_Mandel.to_tensor().eval().array,
        prob,
    )

def test_DR_X():
    left = Create(1, 0)
    right = zx.X(0, 1) @ Scalar((1/2)**(1/2)) >> DualRail()
    assert np.allclose(right.to_tensor().eval().array, left.to_tensor().eval().array)

def test_DR_X_pi():
    left = Create(0, 1)
    right = zx.X(0, 1, phase=0.5) @ Scalar((1/2)**(1/2)) >> DualRail()
    assert np.allclose(right.to_tensor().eval().array, left.to_tensor().eval().array)

def test_DR_beamsplitter():
    beam_splitter = (
            W(2) @ W(2)
            >> ZBox(1, 1, lambda i: ((1 / 2) ** (1 / 2)) ** i) @ ZBox(1, 1, lambda i: ((1 / 2) ** (1 / 2)) ** i) @ ZBox(
        1, 1, lambda i: ((1 / 2) ** (1 / 2)) ** i) @ ZBox(1, 1, lambda i: (-(1 / 2) ** (1 / 2)) ** i)
            >> Id(1) @ SWAP @ Id(1)
            >> W(2).dagger() @ W(2).dagger()
    )


    left = DualRail() >> beam_splitter
    right = zx.H >> DualRail()
    assert compare_arrays_of_different_sizes((right.to_tensor() >>
                                            EmbeddingTensor(2, 3) @ EmbeddingTensor(2, 3)).eval().array.flatten(),
                                            left.to_tensor().eval().array.flatten())

phases = phases = [0.0, 0.3, 0.6]
@pytest.mark.parametrize("phase", phases)
def test_DR_phase_shift(phase):
    left = DualRail() >> Mode(1) @ photonic.Phase(phase).get_kraus()
    right = zx.Z(1, 1, phase=phase) >> DualRail()
    assert np.allclose(right.to_tensor().eval().array, left.to_tensor().eval().array)