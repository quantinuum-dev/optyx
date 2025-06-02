from optyx.core.zw import *
from optyx.core.diagram import Diagram, Mode, Id, Scalar


def test_num_op():
    num_op = Split(2) >> Select() @ Id(Mode(1)) >> Create() @ Id(Mode(1)) >> Merge(2)
    num_op2 = Split(2) @ Create() >> Id(Mode(1)) @ SWAP >> Merge(2) @ Select()
    assert (num_op @ Id(Mode(1))).to_path().eval(2) == (num_op2 @ Id(Mode(1))).to_path().eval(2)
    assert (num_op @ Id(Mode(1))).to_path().eval(3) == (num_op2 @ Id(Mode(1))).to_path().eval(3)
    assert (Id(Mode(1)) @ Create(1) >> num_op @ Id(Mode(1)) >> Id(Mode(1)) @ Select(1)).to_path().eval(
        3
    ) == num_op.to_path().eval(3)
    assert (num_op @ (Create(1) >> Select(1))).to_path().eval(3) == num_op.to_path().eval(3)
    assert (Create(1) @ Id(Mode(1)) >> Id(Mode(1)) @ Split(2) >> Select(1) @ Id(Mode(2))).to_path().eval(
        3
    ) == Split(2).to_path().eval(3)

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


def test_bosonic_operator():
    d1 = Diagram.from_bosonic_operator(
        n_modes= 2,
        operators=((0, False), (1, False), (0, True)),
        scalar=2.1
    )

    annil = Split(2) >> Select(1) @ Id(Mode(1))
    create = annil.dagger()

    d2 = Scalar(2.1) @ annil @ Id(Mode(1)) >> Id(Mode(1)) @ annil >> create @ Id(Mode(1))

    assert d1 == d2