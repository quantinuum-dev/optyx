from optyx.compiler import Measurement


def create_unique_measurements(n: int) -> list[Measurement]:
    return [Measurement(i) for i in range(n)]


def numeric_order(n: int) -> list[int]:
    return list(range(n + 1))
