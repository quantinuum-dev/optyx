"""
Overview
--------

Measurement primitives for photonic quantum computing.

This module defines standard quantum measurements in the Fock basis.
Supported types include threshold measurements and number-resolving
measurements, mapping quantum modes into classical bits or modes.

Classes
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    PhotonThresholdMeasurement
    NumberResolvingMeasurement

Examples
--------

>>> from optyx.optyx import Bit, Mode
>>> box = PhotonThresholdMeasurement()
>>> assert box.double().cod == Bit(1)

Number-resolving measurement:

>>> assert NumberResolvingMeasurement.double().cod == Mode(1)
"""

from optyx.optyx import PhotonThresholdDetector
from optyx.channel import Channel, Measure, bit, qmode


class PhotonThresholdMeasurement(Channel):
    """
    Ideal non-photon resolving detector from qmode to bit.
    Detects whether one or more photons are present.
    """

    def __init__(self):
        super().__init__(
            "PhotonThresholdMeasurement", PhotonThresholdDetector(), cod=bit
        )


# Ideal photon number resolving detector from qmode to mode.
NumberResolvingMeasurement = Measure(qmode)
