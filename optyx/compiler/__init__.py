"""
Toolkit for compiling open graphs in fusion networks

Together this tool kit constitutes a modular compiler for transforming open
graphs into fusion networks and then to optical proticals.

These three representation layers are correspond to the layers defined in the
paper "Fusion and flow: formal protocols to reliably build photonic graph
states" https://arxiv.org/pdf/2409.13541

1. **Open Graph** :class:`OpenGraph`. The MBQC pattern to implement. A graph
state with inputs and outputs, together with measurements on all non-output
nodes.

2. **Fusion Network** :class:`FusionNetwork`. Contains the resource states
and fusions required to implement the MBQC pattern.

3. **Optical Protocol** :class:`Instruction`. The instructions that will
be executed on the quantum hardware.

.. image:: /_static/compilation_flow.png
   :width: 600

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
...     UnmeasuredOp,
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
>>> inputs = [0]
>>> outputs = [2]
>>> og = OpenGraph(g, meas, inputs, outputs)
>>> gflow = og.find_gflow()
>>> fn = compute_linear_fn(g, gflow.layers, meas, 3)
>>> order = add_fusions_to_partial_order(fn.fusions, gflow.partial_order())
>>> ins = compile_linear_fn(fn, order)
>>> assert ins == [
...     NextResourceStateOp(),
...     NextNodeOp(node_id=0),
...     MeasureOp(delay=0, measurement=meas[0]),
...     NextNodeOp(node_id=1),
...     MeasureOp(delay=0, measurement=meas[1]),
...     NextNodeOp(node_id=2),
...     UnmeasuredOp(),
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
