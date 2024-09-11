import math

from optyx.zw import *
import itertools
import pytest


@pytest.mark.skip(reason="Helper function for testing")
def test_arrays_of_different_sizes(array_1, array_2):
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


# Axioms

legs_a_in = range(1, 3)
legs_b_in = range(1, 3)
legs_a_out = range(1, 3)
legs_b_out = range(1, 3)
legs_between = range(1, 3)
max_occupation_nums = [1, 2, 3]

# a set of arbitrary functions of i
fs = [
    lambda i: i,
    lambda i: math.factorial(i),
    lambda i: np.exp(i),
    lambda i: np.sqrt(i),
    lambda i: i**2,
    lambda i: np.cos(i),
]

# get all combinations of legs etc
fs_legs_combinations = list(
    itertools.product(
        fs,
        legs_a_in,
        legs_b_in,
        legs_a_out,
        legs_b_out,
        legs_between,
        max_occupation_nums,
    )
)


@pytest.mark.parametrize(
    "fs, legs_a_in, legs_b_in, legs_a_out, legs_b_out, legs_between, max_occupation_num",
    fs_legs_combinations,
)
def test_spider_fusion(
    fs: int,
    legs_a_in: int,
    legs_b_in: int,
    legs_a_out: int,
    legs_b_out: int,
    legs_between: int,
    max_occupation_num: int,
):

    S1_infty_l = Z(fs, legs_a_in, legs_a_out + legs_between) @ Id(legs_b_in)

    S1_infty_l = S1_infty_l >> Id(legs_a_out + legs_between + legs_b_in)

    S1_infty_l = S1_infty_l >> Id(legs_a_out) @ Z(
        fs, legs_b_in + legs_between, legs_b_out
    )

    fn_mult = lambda i: fs(i) * fs(i)

    S1_infty_r = Z(fn_mult, legs_a_in + legs_b_in, legs_a_out + legs_b_out)

    assert test_arrays_of_different_sizes(
        S1_infty_l.to_tensor(
            max_occupation_num=max_occupation_num,
            print_max_occupation_number=False,
        )
        .eval()
        .array,
        S1_infty_r.to_tensor(
            max_occupation_num=max_occupation_num,
            print_max_occupation_number=False,
        )
        .eval()
        .array,
    )


def test_bSym():
    bSym_l = W(2)
    bSym_r = W(2) >> Swap()

    assert test_arrays_of_different_sizes(
        bSym_l.to_tensor(print_max_occupation_number=False).eval().array,
        bSym_r.to_tensor(print_max_occupation_number=False).eval().array,
    )


def test_bAso():
    bAso_l = W(2) >> W(2) @ Id(1)
    bAso_r = W(2) >> Id(1) @ W(2)

    assert test_arrays_of_different_sizes(
        bAso_l.to_tensor(print_max_occupation_number=False).eval().array,
        bAso_r.to_tensor(print_max_occupation_number=False).eval().array,
    )


def test_bBa():
    bBa_l = (
        ProjectionMap(2)
        >> W(2) @ W(2)
        >> Id(1) @ Swap() @ Id(1)
        >> W(2).dagger() @ W(2).dagger()
    )
    bBa_r = W(2).dagger() >> W(2)

    assert test_arrays_of_different_sizes(
        bBa_l.to_tensor(print_max_occupation_number=False).eval().array,
        bBa_r.to_tensor(print_max_occupation_number=False).eval().array,
    )


def test_bId():
    bId_l = W(2) >> Select(0) @ Id(1)
    bId_r = Id(1)

    assert test_arrays_of_different_sizes(
        bId_l.to_tensor(print_max_occupation_number=False).eval().array,
        bId_r.to_tensor(print_max_occupation_number=False).eval().array,
    )


def test_bZBA():
    from math import factorial

    N = [float(np.sqrt(factorial(i))) for i in range(5)]
    frac_N = [float(1 / np.sqrt(factorial(i))) for i in range(5)]

    bZBA_l = (
        Z(N, 1, 2) @ Z(N, 1, 2)
        >> Id(1) @ Swap() @ Id(1)
        >> W(2).dagger() @ W(2).dagger()
        >> Id(1) @ Z(frac_N, 1, 1)
    )
    bZBA_r = W(2).dagger() >> Z(lambda i: 1, 1, 2)

    assert test_arrays_of_different_sizes(
        bZBA_l.to_tensor(print_max_occupation_number=False).eval().array,
        bZBA_r.to_tensor(print_max_occupation_number=False).eval().array,
    )


def test_K0_infty():
    K0_infty_l = Create(4) >> Z(lambda i: 1, 1, 2)
    K0_infty_r = Create(4) @ Create(4)

    assert test_arrays_of_different_sizes(
        K0_infty_l.to_tensor(print_max_occupation_number=False).eval().array,
        K0_infty_r.to_tensor(print_max_occupation_number=False).eval().array,
    )


def test_scalar():
    scalar_l = Create(1) >> Z([1, 2], 1, 1) >> Select(1)
    scalar_r = Z([2], 0, 0)

    assert test_arrays_of_different_sizes(
        scalar_l.to_tensor(print_max_occupation_number=False).eval().array,
        scalar_r.to_tensor(print_max_occupation_number=False).eval().array,
    )


def test_bone():
    bone_l = Create(1) >> Select(0)
    bone_r = Create(0) >> Select(1)

    assert test_arrays_of_different_sizes(
        bone_l.to_tensor(print_max_occupation_number=False).eval().array, 0
    )
    assert test_arrays_of_different_sizes(
        0, bone_r.to_tensor(print_max_occupation_number=False).eval().array
    )


def test_branching():
    branching_l = Create(1) >> W(2)
    branching_r = Create(1) @ Create(0) + Create(0) @ Create(1)

    assert test_arrays_of_different_sizes(
        branching_l.to_tensor(print_max_occupation_number=False).eval().array,
        branching_r.to_tensor(print_max_occupation_number=False).eval().array,
    )


@pytest.mark.parametrize("k", [1, 2, 3, 4, 10])
def test_normalisation(k: int):
    from math import factorial

    normalisation_l = Create(k) @ Z([np.sqrt(factorial(k))], 0, 0)

    normalisation_r = Id()
    for _ in range(k):
        normalisation_r = normalisation_r @ Create(1)

    normalisation_r = normalisation_r >> W(k).dagger()

    assert test_arrays_of_different_sizes(
        normalisation_l.to_tensor(print_max_occupation_number=False)
        .eval()
        .array,
        normalisation_r.to_tensor(print_max_occupation_number=False)
        .eval()
        .array,
    )


# Some lemmas


@pytest.mark.parametrize("k", [1, 2, 3, 4, 10])
def test_lemma_B6(k: int):

    lemma_B6_l = Create(k) >> Z(lambda i: i + 1, 1, 1)
    lemma_B6_r = Create(k) @ Z([k + 1], 0, 0)

    assert test_arrays_of_different_sizes(
        lemma_B6_l.to_tensor(print_max_occupation_number=False).eval().array,
        lemma_B6_r.to_tensor(print_max_occupation_number=False).eval().array,
    )


def test_lemma_B8():
    lemma_B8_l = (
        Create(1)
        >> Z([1, 1], 1, 2)
        >> W(2) @ W(2)
        >> Id(1) @ Z([1, 1], 2, 0) @ Id(1)
    )

    lemma_B8_r = Create(1) >> W(2) >> Z([1, 1], 1, 2) @ Z([1, 1], 1, 0)

    assert test_arrays_of_different_sizes(
        lemma_B8_l.to_tensor(print_max_occupation_number=False).eval().array,
        lemma_B8_r.to_tensor(print_max_occupation_number=False).eval().array,
    )


def test_lemma_B7():
    lemma_B7_l = Id(1) @ W(2).dagger() >> Z(lambda i: 1, 2, 0)

    lemma_B7_r = (
        W(2) @ Id(2)
        >> Id(1) @ Id(1) @ Swap()
        >> Id(1) @ Swap() @ Id(1)
        >> Z(lambda i: 1, 2, 0) @ Z(lambda i: 1, 2, 0)
    )

    assert test_arrays_of_different_sizes(
        lemma_B7_l.to_tensor(print_max_occupation_number=False).eval().array,
        lemma_B7_r.to_tensor(print_max_occupation_number=False).eval().array,
    )


def test_prop_54():
    prop_54_l = (
        Create(1) @ Id(1)
        >> Z(lambda i: 1, 1, 2) @ Id(1)
        >> Id(1) @ W(2).dagger()
        >> Id(1) @ W(2)
        >> Z(lambda i: 1, 2, 0) @ Id(1)
    )

    prop_54_r = (
        Create(1) @ Id(1)
        >> W(2) @ W(2)
        >> Id(1) @ Z(lambda i: 1, 2, 1) @ Id(1)
        >> Z(lambda i: 1, 1, 0) @ W(2).dagger()
    )

    assert test_arrays_of_different_sizes(
        prop_54_l.to_tensor(print_max_occupation_number=False).eval().array,
        prop_54_r.to_tensor(print_max_occupation_number=False).eval().array,
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
def test_bBa(postselect_and_prob: list):
    select_1 = postselect_and_prob[0]
    select_2 = postselect_and_prob[1]
    prob = postselect_and_prob[2]

    Zb_i = Z(np.array([1, 1j / (np.sqrt(2))]), 1, 1)
    Zb_1 = Z(np.array([1, 1 / (np.sqrt(2))]), 1, 1)

    beam_splitter = (
        W(2) @ W(2)
        >> Zb_i @ Zb_1 @ Zb_1 @ Zb_i
        >> Id(1) @ Swap() @ Id(1)
        >> W(2).dagger() @ W(2).dagger()
    )

    Hong_Ou_Mandel = (
        Create(1) @ Create(1)
        >> beam_splitter
        >> Select(select_1) @ Select(select_2)
    )

    assert test_arrays_of_different_sizes(
        Hong_Ou_Mandel.to_tensor(print_max_occupation_number=False)
        .eval()
        .array,
        prob,
    )
