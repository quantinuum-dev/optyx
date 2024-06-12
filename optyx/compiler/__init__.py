"""
Toolkit for compiling open graphs in machine instructions

Together this tool kit constitutes a modular compiler for transforming open
graphs into hardware instructions.

The modular design allows reuse of common compilation and optimisation
routinues between different compilation pipelines which can diverge whenever
necessary.

There are three representation layers in the compiler:

1. **Open Graph** :class:`OpenGraph`. This is the MBQC pattern we wish to
implement. It is a graph state with inputs and outputs, together with
measurements on all non-output nodes.

2. **Fusion Network** :class:`ULFusionNetwork`. Contains the resource states
and fusions required to implement the MBQC pattern

3. **Hardware Instructions** :class:`Instruction`. The instructions that will
be executed on the quantum hardware.

The algorithm is given in the form of an **Open Graph** which is then compiled
into a **Fusion Network** and is later compiled into a series of **Hardware
Instructions**.

Currently we only have a pipeline for compiling to a machine with a single
linear resource state emitter and a single measurement device. This
functionality is contained in the :mod:`semm` module and an example is given
below.

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Measurement
    OpenGraph
    GFlow
    PartialOrder
    ULFusionNetwork
    Instruction

    compile_to_semm
    decompile_from_semm

----------
Submodules
----------

    .. autosummary::
        :template: module.rst
        :nosignatures:
        :toctree:

        semm
        semm_decompiler

Example
-------

This example defines an open graph and compiles it into a fusion network and
then into a series of instruction to be executed on a SEMM machine.

This procedure is captured in the :meth:`optyx.compiler.semm.compile_to_semm`
method.

>>> import networkx as nx
>>> from optyx.compiler.protocols import (
...     NextNodeOp,
...     MeasureOp,
...     NextResourceStateOp,
...     UnmeasuredPhotonOp,
... )
>>> from optyx.compiler.mbqc import (
...     OpenGraph,
...     Measurement,
...     ULFusionNetwork,
...     add_fusions_to_partial_order,
... )
>>> from optyx.compiler.semm import (
...     compute_linear_fn,
...     compile_linear_fn,
... )
>>>
>>> g = nx.Graph([(0, 1), (1, 2)])
>>> meas = {i: Measurement(i, 'XY') for i in range(2)}
>>> inputs = {0}
>>> outputs = {2}
>>> og = OpenGraph(g, meas, inputs, outputs)
>>> sfn = compute_linear_fn(og, 3)
>>> gflow_order = og.find_gflow().partial_order()
>>> order = add_fusions_to_partial_order(sfn.fusions, gflow_order)
>>> ins = compile_linear_fn(sfn, order)
>>> assert ins == [
...     NextResourceStateOp(),
...     NextNodeOp(node_id=0),
...     MeasureOp(delay=0, measurement=meas[0]),
...     NextNodeOp(node_id=1),
...     MeasureOp(delay=0, measurement=meas[1]),
...     NextNodeOp(node_id=2),
...     UnmeasuredPhotonOp(),
... ]
"""

from .mbqc import (
    Measurement,
    OpenGraph,
    GFlow,
    PartialOrder,
    ULFusionNetwork,
)

from .protocols import Instruction

from .semm import (
    compile_to_semm,
    decompile_from_semm,
)
