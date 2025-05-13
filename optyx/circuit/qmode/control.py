"""
This module defines classes for bit-controlled quantum gates and phase shifts.
"""

from optyx.diagram.control import (
    BitControlledBox,
    ControlledPhaseShift as ControlledPhaseShiftBox,
)
from optyx.diagram.channel import (
    Channel,
    bit,
    mode,
    qmode,
    Circuit
)
from typing import Callable, List

class BitControlledGate(Channel):
    """
    Represents a gate that is
    controlled by a classical bit.
    It uses a `BitControlledBox` to define
    the Kraus operators for the gate.
    """
    def __init__(self,
                 control_gate,
                 default_gate=None):
        if isinstance(control_gate, (Circuit, Channel)):
            assert control_gate.is_pure, \
                 "The input gates must be pure quantum channels"
            control_gate = control_gate.get_kraus()
        if isinstance(default_gate, (Circuit, Channel)):
            assert default_gate.is_pure, \
                 "The input gates must be pure quantum channels"
            default_gate = default_gate.get_kraus()
        kraus = BitControlledBox(control_gate, default_gate)
        super().__init__(
            f"BitControlledGate({control_gate}, {default_gate})",
            kraus,
            bit @ control_gate.dom,
            control_gate.cod
        )


class BitControlledPhaseShift(Channel):
    """
    Represents a phase shift operation
    that is controlled by classical bits.
    It uses a `ControlledPhaseShiftBox` to
    define the Kraus operators for the phase shift.
    """
    def __init__(self,
                 function: Callable[[List[int]], List[int]],
                 n_modes: int = 1,
                 n_control_modes: int = 1):
        kraus = ControlledPhaseShiftBox(function, n_modes, n_control_modes)
        super().__init__(
            "BitControlledPhaseShift",
            kraus,
            mode**n_control_modes @ qmode**n_modes,
            mode**n_modes,
        )