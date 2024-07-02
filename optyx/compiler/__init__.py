"""
Toolkit for compiling open graphs in machine instructions

There are three representation layers in the compiler:

1. :class:`OpenGraph`: This is the MBQC pattern we wish to implement. It is a
graph state with inputs and outputs, together with measurements on all
non-output nodes.

2. :class:`FusionNetwork`: Contains the resource states and fusions required to
implement the MBQC pattern

3. **Hardware Instructions**: The instructions that will be executed on the
quantum hardware.

The toolkit provides several algorithms for converting an :class:`OpenGraph`
into a :class:`FusionNetwork` which aim to be hardware agnostic.

Currently we only have one pipeline for compiling a :class:`FusionNetwork` to
hardware instructions on machines which contain a single linear resource state
emitter and a single measurement device. This functionality is contained in the
:mod:`semm` module and an example is given below.

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Measurement
    OpenGraph
    GFlow
    PartialOrder
    FusionNetwork
    Instruction

    compile_to_semm
    decompile_from_semm

    pattern_satisfies_order

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
>>> from optyx.compiler.protocols import NextNodeOp, MeasureOp
>>> from optyx.compiler.mbqc import (
...     OpenGraph,
...     Measurement,
...     FusionNetwork,
...     add_fusion_order,
... )
>>> from optyx.compiler.semm import (
...     compile_to_fusion_network,
...     fn_to_semm,
... )
>>>
>>> g = nx.Graph([(0, 1), (1, 2)])
>>> meas = {i: Measurement(i, 'XY') for i in range(3)}
>>> inputs = {0}
>>> outputs = {2}
>>> og = OpenGraph(g, meas, inputs, outputs)
>>> sfn = compile_to_fusion_network(og)
>>> gflow_order = og.find_gflow().partial_order()
>>> order = add_fusion_order(sfn.fusions, gflow_order)
>>> ins = fn_to_semm(sfn, order)
>>> assert ins == [
...     NextNodeOp(node_id=0),
...     MeasureOp(delay=0, measurement=meas[0]),
...     NextNodeOp(node_id=1),
...     MeasureOp(delay=0, measurement=meas[1]),
...     NextNodeOp(node_id=2),
...     MeasureOp(delay=0, measurement=meas[2]),
... ]
"""

from .mbqc import (
    Measurement,
    OpenGraph,
    GFlow,
    PartialOrder,
    FusionNetwork,
    pattern_satisfies_order,
)

from .protocols import Instruction

from .semm import (
    compile_to_semm,
    decompile_from_semm,
)
