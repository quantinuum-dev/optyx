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

from optyx.diagram.optyx import PhotonThresholdDetector
from optyx.diagram.zw import Select
from optyx.diagram.channel import (
    Channel,
    Measure,
    bit,
    qmode,
    Discard
)

class PhotonThresholdMeasurement(Channel):
    """
    Ideal photon-number non-resolving detector
    from mode to bit from qmode to bit.
    Detects whether one or more photons are present.
    """

    def __init__(self):
        super().__init__(
            "PhotonThresholdMeasurement",
            PhotonThresholdDetector(),
            cod=bit
        )


# Ideal photon number resolving detector from qmode to mode.
NumberResolvingMeasurement = Measure(qmode)

Discard = Discard(qmode)

class Select(Channel):
    def __init__(self, *n_photons: int):
        super().__init__(
            f"Select({n_photons})",
            Select(*n_photons)
        )