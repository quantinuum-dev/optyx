from optyx.compiler.mbqc import Measurement
import math


# Returns a list of unique measurements, all with different angles.
def create_unique_measurements(n: int) -> dict[int, Measurement]:
    small_angle = 2 * math.pi / float(n)
    return {i: Measurement(i * small_angle, "XY") for i in range(n)}


def numeric_order(n: int) -> list[int]:
    return list(range(n + 1))
