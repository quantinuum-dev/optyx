"""
This module defines classes for classical
functions and binary matrix operations
in the context of classical circuits.
"""

from optyx.diagram.classical_functions import (
    ClassicalFunctionBox,
    BinaryMatrixBox,
    ControlChannel
)

class ClassicalFunction(ControlChannel):
    """
    Represents a classical function as a control channel. It wraps a
    `ClassicalFunctionBox` with the specified function, domain, and codomain.
    """
    def __new__(cls, function, dom, cod):
        box = ClassicalFunctionBox(
            function,
            dom,
            cod
        )
        return super().__new__(cls, box)

class BinaryMatrix(ControlChannel):
    """
    Represents a binary matrix as a control channel. It wraps a
    `BinaryMatrixBox` with the specified matrix.
    """
    def __new__(cls, matrix):
        box = BinaryMatrixBox(matrix)
        return super().__new__(cls, box)
