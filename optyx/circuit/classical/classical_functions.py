from optyx.diagram.classical_functions import (
    ClassicalFunctionBox,
    BinaryMatrixBox,
    ControlChannel
)

class ClassicalFunction(ControlChannel):
    def __new__(cls, function, dom, cod):
        box = ClassicalFunctionBox(
            function,
            dom,
            cod
        )
        return super().__new__(cls, box)

class BinaryMatrix(ControlChannel):
    def __new__(cls, matrix):
        box = BinaryMatrixBox(matrix)
        return super().__new__(cls, box)
