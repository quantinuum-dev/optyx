from optyx.optyx import PhotonThresholdDetector
from optyx.channel import Channel, Measure, bit, qmode

class PhotonThresholdMeasurement(Channel):
    """
    Ideal non-photon resolving detector from qmode to bit.
    Detects whether one or more photons are present.
    """

    def __init__(self):
        super().__init__("PhotonThresholdMeasurement",
                         PhotonThresholdDetector(), cod=bit)


#Ideal photon number resolving detector from qmode to mode.
NumberResolvingMeasurement = Measure(qmode)