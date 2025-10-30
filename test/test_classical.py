from optyx.classical import *
from optyx.utils.misc import compare_arrays_of_different_sizes

def test_addn():
    d_1 = Digit(2, 3) >> AddN(2)
    d_2 = Digit(5)

    assert compare_arrays_of_different_sizes(d_1.double().to_tensor().eval().array,
                       d_2.double().to_tensor().eval().array)

def test_subn():
    d_1 = Digit(2, 5) >> SubN()
    d_2 = Digit(3)

    assert compare_arrays_of_different_sizes(d_1.double().to_tensor().eval().array,
                       d_2.double().to_tensor().eval().array)

def test_multiplyn():
    d_1 = Digit(2, 3) >> MultiplyN()
    d_2 = Digit(6)

    assert compare_arrays_of_different_sizes(d_1.double().to_tensor().eval().array,
                          d_2.double().to_tensor().eval().array)

def test_dividen():
    d_1 = Digit(6, 2) >> DivideN()
    d_2 = Digit(3)

    assert compare_arrays_of_different_sizes(d_1.double().to_tensor().eval().array,
                          d_2.double().to_tensor().eval().array)

def test_mod2():
    d_1 = Digit(5) >> Mod2()
    d_3 = Bit(1)

    assert compare_arrays_of_different_sizes(d_1.double().to_tensor().eval().array,
                            d_3.double().to_tensor().eval().array)

def test_copyn():
    d_1 = Digit(2) >> CopyN(3)
    d_2 = Digit(2, 2, 2)

    assert compare_arrays_of_different_sizes(d_1.double().to_tensor().eval().array,
                            d_2.double().to_tensor().eval().array)

def test_swapn():
    d_1 = Digit(2, 3) >> SwapN()
    d_2 = Digit(3, 2)

    assert compare_arrays_of_different_sizes(d_1.double().to_tensor().eval().array,
                            d_2.double().to_tensor().eval().array)

def test_postselectbit():
    d_1 = Bit(1, 0) >> PostselectBit(1, 0)
    d_2 = Scalar(1)

    assert compare_arrays_of_different_sizes(d_1.double().to_tensor().eval().array,
                            d_2.double().to_tensor().eval().array)

def test_postselectdigit():
    d_1 = Digit(2, 3, 4) >> PostselectDigit(2, 3, 4)
    d_2 = Scalar(1)

    assert compare_arrays_of_different_sizes(d_1.double().to_tensor().eval().array,
                            d_2.double().to_tensor().eval().array)

def test_notbit():
    d_1 = Bit(1) >> NotBit()
    d_2 = Bit(0)

    assert compare_arrays_of_different_sizes(d_1.double().to_tensor().eval().array,
                            d_2.double().to_tensor().eval().array)

def test_xorbit():
    d_1 = Bit(1, 0) >> XorBit()
    d_2 = Bit(1)

    assert compare_arrays_of_different_sizes(d_1.double().to_tensor().eval().array,
                            d_2.double().to_tensor().eval().array)

def test_andbit():
    d_1 = Bit(1, 0) >> AndBit()
    d_2 = Bit(0)

    assert compare_arrays_of_different_sizes(d_1.double().to_tensor().eval().array,
                            d_2.double().to_tensor().eval().array)

def test_copybit():
    d_1 = Bit(1) >> CopyBit(3)
    d_2 = Bit(1, 1, 1)

    assert compare_arrays_of_different_sizes(d_1.double().to_tensor().eval().array,
                            d_2.double().to_tensor().eval().array)

def test_swapbit():
    d_1 = Bit(1, 0) >> SwapBit()
    d_2 = Bit(0, 1)

    assert compare_arrays_of_different_sizes(d_1.double().to_tensor().eval().array,
                            d_2.double().to_tensor().eval().array)

def test_orbit():
    d_1 = Bit(1, 0) >> OrBit()
    d_2 = Bit(1)

    assert compare_arrays_of_different_sizes(d_1.double().to_tensor().eval().array,
                            d_2.double().to_tensor().eval().array)

def test_zxh():
    d_1 = X(0, 1) >> H()
    d_2 = Z(0, 1)

    assert compare_arrays_of_different_sizes(d_1.double().to_tensor().eval().array,
                            d_2.double().to_tensor().eval().array)