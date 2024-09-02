# %%
import math

from optyx.zw import *
import itertools
import tqdm
import gc
import pytest

# %%
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

# %% [markdown]
# ## Axioms

legs_a_in = range(1, 3)
legs_b_in = range(1, 3)
legs_a_out = range(1, 3)
legs_b_out = range(1, 3)
legs_between = range(1, 3)

# a set of random functions of i
fs = [lambda i: i,
      lambda i: math.factorial(i),
      lambda i: np.exp(i)]

# get all combinations of legs
fs_legs_combinations = list(itertools.product(fs, legs_a_in, legs_b_in, legs_a_out, legs_b_out, legs_between))
for max_occupation_num in [1, 2, 3]:
      for legs in tqdm.tqdm(fs_legs_combinations):
            fn = legs[0]

            S1_infty_l = ZBox(fn, legs[1], legs[3] + legs[5]) @ Id(legs[2])

            S1_infty_l = S1_infty_l >> Id(legs[3] + legs[5] + legs[2])

            S1_infty_l = S1_infty_l >> Id(legs[3]) @ ZBox(fn, legs[2] + legs[5], legs[4]) 

            fn_mult = lambda i: fn(i) * fn(i)

            S1_infty_r = ZBox(fn_mult, legs[1] + legs[2], legs[3] + legs[4])

            assert test_arrays_of_different_sizes(S1_infty_l.to_tensor(max_occupation_num=max_occupation_num,
                                                                  print_max_occupation_number=False).eval().array, 
                                                      S1_infty_r.to_tensor(max_occupation_num=max_occupation_num,
                                                                        print_max_occupation_number=False).eval().array), \
                  f"Failed for {legs} and max_occupation_num={max_occupation_num}" 
            
            del S1_infty_l, S1_infty_r, fn_mult, fn, legs

            gc.collect()



# %%
bSym_l = W(2)
bSym_r = W(2) >> Swap()

assert test_arrays_of_different_sizes(bSym_l.to_tensor(print_max_occupation_number=False).eval().array, bSym_r.to_tensor(print_max_occupation_number=False).eval().array) 

# %%
bAso_l = W(2) >> W(2) @ Id(1)
bAso_r = W(2) >> Id(1) @ W(2)

assert test_arrays_of_different_sizes(bAso_l.to_tensor(print_max_occupation_number=False).eval().array, bAso_r.to_tensor(print_max_occupation_number=False).eval().array)

# %%
bBa_l = Truncation(2) >> W(2) @ W(2) >> Id(1) @ Swap() @ Id(1) >> W(2).dagger() @ W(2).dagger()
bBa_r = W(2).dagger() >> W(2)

assert test_arrays_of_different_sizes(bBa_l.to_tensor(print_max_occupation_number=False).eval().array, bBa_r.to_tensor(print_max_occupation_number=False).eval().array)

# %%
bId_l = W(2) >> Select(0) @ Id(1)
bId_r = Id(1)

assert test_arrays_of_different_sizes(bId_l.to_tensor(print_max_occupation_number=False).eval().array, bId_r.to_tensor(print_max_occupation_number=False).eval().array)

# %%
from math import factorial

N = [float(np.sqrt(factorial(i))) for i in range(5)]
frac_N = [float(1/np.sqrt(factorial(i))) for i in range(5)]

bZBA_l = ZBox(N, 1, 2) @ ZBox(N, 1, 2) >> Id(1) @ Swap() @ Id(1) >> W(2).dagger() @ W(2).dagger() >> Id(1) @ ZBox(frac_N, 1, 1) 
bZBA_r = W(2).dagger() >> ZBox([1, 1, 1, 1, 1], 1, 2) 

assert test_arrays_of_different_sizes(bZBA_l.to_tensor(print_max_occupation_number=False).eval().array, bZBA_r.to_tensor(print_max_occupation_number=False).eval().array)

# %%
K0_infty_l = Create(4) >> ZBox([1, 1, 1, 1, 1], 1, 2)
K0_infty_r = Create(4) @ Create(4) 

assert test_arrays_of_different_sizes(K0_infty_l.to_tensor(print_max_occupation_number=False).eval().array, K0_infty_r.to_tensor(print_max_occupation_number=False).eval().array)

# %%
scalar_l = Create(1) >> ZBox([1, 2], 1, 1) >> Select(1)
scalar_r = ZBox([2], 0, 0)

assert test_arrays_of_different_sizes(scalar_l.to_tensor(print_max_occupation_number=False).eval().array, scalar_r.to_tensor(print_max_occupation_number=False).eval().array)

# %%
bone_l = Create(1) >> Select(0)
bone_r = Create(0) >> Select(1)

assert test_arrays_of_different_sizes(bone_l.to_tensor(print_max_occupation_number=False).eval().array, 0)
assert test_arrays_of_different_sizes(0, bone_r.to_tensor(print_max_occupation_number=False).eval().array)

# %%
branching_l = Create(1) >> W(2)
branching_r = Create(1) @ Create(0) + Create(0) @ Create(1)

assert test_arrays_of_different_sizes(branching_l.to_tensor(print_max_occupation_number=False).eval().array, branching_r.to_tensor(print_max_occupation_number=False).eval().array)

# %%
from math import factorial

k = 6

normalisation_l = Create(k) @ ZBox([np.sqrt(factorial(k))], 0, 0)
normalisation_r = Create(1) @ Create(1) @ Create(1) @ Create(1) @ Create(1) @ Create(1) >> W(6).dagger()

assert test_arrays_of_different_sizes(normalisation_l.to_tensor(print_max_occupation_number=False).eval().array, normalisation_r.to_tensor(print_max_occupation_number=False).eval().array) 

# %% [markdown]
# # Lemmas

# %%
k = 5

lemma_B6_l = Create(k) >> ZBox([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 1, 1)
lemma_B6_r = Create(k) @ ZBox([k+1], 0, 0)

assert test_arrays_of_different_sizes(lemma_B6_l.to_tensor(print_max_occupation_number=False).eval().array, lemma_B6_r.to_tensor(print_max_occupation_number=False).eval().array)


# %%
lemma_B8_l = Create(1) >> \
            ZBox([1, 1], 1, 2) >>\
            W(2) @ W(2) >>\
            Id(1) @ ZBox([1, 1], 2, 0) @ Id(1)

lemma_B8_r = Create(1) >> \
            W(2) >> \
            ZBox([1, 1], 1, 2) @ ZBox([1, 1], 1, 0)

assert test_arrays_of_different_sizes(lemma_B8_l.to_tensor(print_max_occupation_number=False).eval().array, lemma_B8_r.to_tensor(print_max_occupation_number=False).eval().array)

# %%
lemma_B7_l = Id(1) @ W(2).dagger() >> \
             ZBox(lambda i: 1, 2, 0)

lemma_B7_r = W(2) @ Id(2) >>\
             Id(1) @ Id(1) @ Swap() >>\
             Id(1) @ Swap() @ Id(1) >>\
             ZBox(lambda i: 1, 2, 0) @ ZBox(lambda i: 1, 2, 0)

assert test_arrays_of_different_sizes(lemma_B7_l.to_tensor(print_max_occupation_number=False).eval().array, lemma_B7_r.to_tensor(print_max_occupation_number=False).eval().array)

# %% [markdown]
# **The lemma below holds only up to Truncation:**
# 
# Maps which are equal by the triangle bialgebra in $ZW_\infty$, are only equal in $ZW$ up to Truncation

# %%
prop_54_l = Create(1) @ Id(1) >>\
            ZBox(lambda i: 1, 1, 2) @ Id(1) >>\
            Id(1) @ W(2).dagger() >>\
            Id(1) @ W(2) >>\
            ZBox(lambda i: 1, 2, 0) @ Id(1)

prop_54_r = Create(1) @ Id(1) >>\
            W(2) @ W(2) >>\
            Id(1) @ ZBox(lambda i: 1, 2, 1) @ Id(1) >>\
            ZBox(lambda i: 1, 1, 0) @ W(2).dagger()

assert test_arrays_of_different_sizes(prop_54_l.to_tensor(print_max_occupation_number=False).eval().array, prop_54_r.to_tensor(print_max_occupation_number=False).eval().array)
# %%


# %% [markdown]
# # Examples

# %% [markdown]
# ## Hong-Ou-Mandel

# %%
Zb_i = ZBox(np.array([1, 1j/(np.sqrt(2))]), 1, 1)
Zb_1 = ZBox(np.array([1, 1/(np.sqrt(2))]), 1, 1)

beam_splitter = W(2) @ W(2) >> \
               Zb_i @ Zb_1 @ Zb_1 @ Zb_i >> \
               Id(1) @ Swap() @ Id(1) >> \
               W(2).dagger() @ W(2).dagger()

# %%
Hong_Ou_Mandel = Create(1) @ Create(1) >> \
                beam_splitter >> \
                Select(1) @ Select(1) 

# %%
assert test_arrays_of_different_sizes(Hong_Ou_Mandel.to_tensor(print_max_occupation_number=False).eval().array, np.array([0]))

# %%
Hong_Ou_Mandel = Create(1) @ Create(1) >> \
                beam_splitter >> \
                Select(2) @ Select(0) 

# %%
Hong_Ou_Mandel.to_tensor(print_max_occupation_number=False).eval()

assert test_arrays_of_different_sizes(Hong_Ou_Mandel.to_tensor(print_max_occupation_number=False).eval().array, np.array(np.sqrt(2)*1j/2))

# %%
Hong_Ou_Mandel = Create(1) @ Create(1) >> \
                beam_splitter >> \
                Select(0) @ Select(2) 

# %%
Hong_Ou_Mandel.to_tensor(print_max_occupation_number=False).eval()

assert test_arrays_of_different_sizes(Hong_Ou_Mandel.to_tensor(print_max_occupation_number=False).eval().array, np.array(np.sqrt(2)*1j/2))

