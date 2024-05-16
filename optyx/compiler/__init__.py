"""
Toolkit for compiling open graphs in machine instructions

Together this tool kit constitutes a modular compiler for transforming open
graphs into hardware instructions.

Currently we only have a pipeline for compiling to a machine with a single
linear resource state emitter and a single measurement device. This
functionality is contained in the `optyx.compiler.semm` module.

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Measurement
    OpenGraph
    PartialOrder

    compile_to_semm
    decompile_to_semm
"""

from .mbqc import (
    Measurement,
    OpenGraph,
    PartialOrder,
)

from .semm import (
    compile_to_semm,
    decompile_from_semm,
)
