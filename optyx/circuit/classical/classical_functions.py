"""
This module defines classes for classical
functions and binary matrix operations
in the context of classical circuits.
"""

from optyx.diagram.classical_functions import (
    ClassicalFunctionBox,
    BinaryMatrixBox
)
from optyx.circuit.classical.classical_circuit import ClassicalBox
from optyx.diagram.channel import Ty, Ob


class ControlChannel(ClassicalBox):
    """
    Syntactic sugar.
    Converts a classical circuit (Diagram or Box)
    into a CQMap, allowing
    it to be used as a control channel in hybrid quantum-classical systems.
    """
    pass


class ClassicalFunction(ControlChannel):
    """
    Represents a classical function as a control channel. It wraps a
    `ClassicalFunctionBox` with the specified function, domain, and codomain.
    """
    def __init__(self, function, dom, cod):
        box = ClassicalFunctionBox(
            function,
            dom,
            cod
        )
        return super().__init__(
            box.name,
            box,
            Ty(*[Ob._classical[ob.name] for ob in box.dom.inside]),
            Ty(*[Ob._classical[ob.name] for ob in box.cod.inside]),
        )


class BinaryMatrix(ControlChannel):
    """
    Represents a binary matrix as a control channel. It wraps a
    `BinaryMatrixBox` with the specified matrix.
    """
    def __init__(self, matrix):
        box = BinaryMatrixBox(self, matrix)
        return super().__new__(
            box.name,
            box,
            Ty(*[Ob._classical[ob.name] for ob in box.dom.inside]),
            Ty(*[Ob._classical[ob.name] for ob in box.cod.inside]),
        )
