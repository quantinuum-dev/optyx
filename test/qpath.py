from optyx.qpath import *


def test_num_op():
    num_op = Split() >> Select() @ Id(1) >> Create() @ Id(1) >> Merge()
    num_op2 = Split() @ Create() >> Id(1) @ SWAP >> Merge() @ Select()
    assert (num_op @ Id(1)).eval(2) == (num_op2 @ Id(1)).eval(2)
    assert (num_op @ Id(1)).eval(3) == (num_op2 @ Id(1)).eval(3)
    assert (Id(1) @ Create(1) >> num_op @ Id(1) >> Id(1) @ Select(1)).eval(
        3
    ) == num_op.eval(3)
    assert (num_op @ (Create(1) >> Select(1))).eval(3) == num_op.eval(3)
    assert (Create(1) @ Id(1) >> Id(1) @ Split() >> Select(1) @ Id(2)).eval(
        3
    ) == Split().eval(3)

def test_dilate():
    matrices = [Matrix(np.random.random((n + 2, m + 1)), dom = n, cod = m, 
                       creations = (1, 1, ), selections = (2,),)
                for n in range(1, 5) for m in range(1, 5)]
    for matrix in matrices:
        unitary = matrix.dilate()
        assert np.allclose(
            (unitary.umatrix >> unitary.umatrix.dagger()).array,
            np.eye(unitary.umatrix.dom))
        assert np.allclose(unitary.eval(3).array, matrix.eval(3).array)