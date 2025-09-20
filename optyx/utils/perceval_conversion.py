from optyx.utils.utils import matrix_to_zw, invert_perm
from optyx import Channel, mode, qmode, photonic, bit
from optyx.classical import ClassicalFunction, BitControlledGate
from perceval.components.detector import DetectionType
from optyx.core.channel import Spider, Diagram, Measure
from optyx.classical import Select
from optyx.photonic import Create
from optyx.utils.postselect_parser import compile_postselect
from optyx.core.zw import Endo
import numpy as np
import perceval as pcvl


def _default_action(component):
    return matrix_to_zw(
        np.array(component.default_circuit.U.T, dtype=np.complex128)
    )


def _state_predicate(state, ty):
    """Return ClassicalFunction that outputs 1 iff input equals 'state'."""
    def f(x):
        return [1 if all(s == a for s, a in zip(state, x)) else 0]
    return ClassicalFunction(f, ty**len(state), bit)


def _rewire_and_context(*, component, wires, circuit,
                        n_classical, n_action, n_offset,
                        perm_dom_neg, perm_dom_pos):
    """Compute (input_perm, output_perm, left, right)
    based on component._offset sign."""
    if component._offset < 0:
        # classical wires after action+offset
        p_input = (
            list(range(n_action + n_offset,
                       n_action + n_offset + n_classical)) +
            list(range(0, n_action)) +
            list(range(n_action, n_action + n_offset))
        )
        input_perm = Diagram.permutation(p_input, perm_dom_neg)
        output_perm = Diagram.permutation(invert_perm(p_input), input_perm.cod)
        left = circuit.cod[:min(wires) - n_offset - n_action]
        right = circuit.cod[max(wires) + 1:]
    else:
        # classical wires before offset+action
        p_input = (
            list(range(n_classical, n_classical + n_offset)) +
            list(range(0, n_classical)) +
            list(range(n_classical + n_offset,
                       n_classical + n_offset + n_action))
        )
        input_perm = Diagram.permutation(p_input, perm_dom_pos)
        output_perm = Diagram.permutation(invert_perm(p_input), input_perm.cod)
        left = circuit.cod[:min(wires)]
        right = circuit.cod[max(wires) + n_offset + n_action + 1:]
    return input_perm, output_perm, left, right


def _assemble_controlled_box(
        map_items, *,
        input_wires,
        default_action,):
    """
    Build the feed-forward controlled box from (state, action_box) pairs.
    action_box must already be a ZW diagram (matrix_to_zw applied).
    """
    box = None
    n_action = len(default_action.dom)
    for i, (state, action) in enumerate(map_items):
        if i == 0:
            box = input_wires[:len(state)] @ photonic.Id(n_action)
        # duplicate classical control and compute predicate
        copy = Spider(1, 2, input_wires[0])**len(state)
        permutation = Diagram.permutation(
            list(range(0, 2*len(state), 2)) + list(range(1, 2*len(state), 2)),
            input_wires[0]**(2*len(state))
        )
        func = _state_predicate(list(state), input_wires[0])

        ctrl_box = BitControlledGate(action, default_action)
        q_wires = qmode**n_action

        box >>= (
            copy @ q_wires >>
            permutation @ q_wires >>
            Diagram.id(input_wires[:len(state)]) @ func @ q_wires >>
            Diagram.id(input_wires[:len(state)]) @ ctrl_box
        )
    return box


def _feedforward_common(*, component, wires, circuit,
                        map_iter, action_from_item,
                        use_provider_dom: bool):
    """
    Shared implementation for both FFCircuitProvider and FFConfigurator.
    """
    default_action = _default_action(component)

    n_classical = len(next(iter(map_iter.keys())))
    n_action = len(default_action.dom)
    n_offset = abs(component._offset)

    # build permutation domains for input permutation
    # if use_provider_dom:
    #    perm_dom_neg = qmode**(n_action + n_offset) @ mode**n_classical
    #    perm_dom_pos = mode**n_classical @ qmode**(n_action + n_offset)
    # else:
    # configurator: the dom is a slice of the current circuit wires
    perm_dom_neg = circuit.cod[min(wires) - n_offset -
                               n_action: max(wires) + 1]
    perm_dom_pos = circuit.cod[min(wires): max(wires) +
                               n_offset + n_action + 1]

    input_perm, output_perm, left, right = _rewire_and_context(
        component=component, wires=wires, circuit=circuit,
        n_classical=n_classical, n_action=n_action, n_offset=n_offset,
        perm_dom_neg=perm_dom_neg, perm_dom_pos=perm_dom_pos
    )

    # prepare (state, action_box) pairs
    action_pairs = []
    for state, payload in map_iter.items():
        action_pairs.append((state, action_from_item(payload)))

    # figure out the "offset wires" identity block to thread through
    if component._offset < 0:
        offset_wires = circuit.cod[
            len(left) + n_action: len(left) + n_action + n_offset
        ]
    else:
        offset_wires = circuit.cod[
            len(left) + n_classical: len(left) + n_classical + n_offset
        ]

    box = _assemble_controlled_box(
            action_pairs,
            input_wires=input_perm.cod[len(offset_wires):],
            default_action=default_action
        )

    return (input_perm >> offset_wires @ box >> output_perm), left, right


def ff_circuit_provider(component, wires, circuit):
    def action_from_item(action_circuit):
        return matrix_to_zw(np.array(action_circuit.U.T, dtype=np.complex128))

    box, left, right = _feedforward_common(
        component=component,
        wires=wires,
        circuit=circuit,
        map_iter=component._map,
        action_from_item=action_from_item,
        use_provider_dom=True,
    )
    return box, left, right


def ff_configurator(component, wires, circuit):
    free_symbols = component._controlled.U.free_symbols

    def action_from_item(symbol_values):
        # substitute symbol values and convert to ZW
        subs_map = {s: v for s, v in zip(free_symbols, symbol_values.values())}
        action_U = np.array(
            component._controlled.U.subs(subs_map).evalf().T,
            dtype=np.complex128
        )
        return matrix_to_zw(action_U)

    box, left, right = _feedforward_common(
        component=component,
        wires=wires,
        circuit=circuit,
        map_iter=component._configs,
        action_from_item=action_from_item,
        use_provider_dom=False,
    )
    return box, left, right


def unitary(component, wires):
    if component.U.is_symbolic and not isinstance(component.U, pcvl.MatrixN):
        if len(component.U.free_symbols) != 0:
            raise TypeError("Symbolic circuits are not currently supported")
    U = np.array(component.U, dtype=np.complex128)
    if U.shape[0] != len(wires):
        raise ValueError("A component acting on polarisation modes.")
    return Channel(name=component.name, kraus=matrix_to_zw(U.T))


def heralds_diagram(heralds, n_modes, circuit, in_out):
    layer = photonic.Id(0)
    create_select = (
        Create if in_out == "in" else
        Select if in_out == "out" else None
    )
    if create_select is None:
        raise ValueError("in_out must be either 'in' or 'out'")

    if heralds is not None:
        for m in range(n_modes):
            if m in heralds:
                layer @= create_select(heralds[m])
            else:
                layer @= circuit.cod[m] if in_out == "out" else photonic.Id(1)
    return layer


def detector(component, wires):
    if component.type == DetectionType.PNR:
        return photonic.NumberResolvingMeasurement(len(wires))
    if component.type == DetectionType.Threshold:
        return photonic.PhotonThresholdMeasurement(len(wires))
    raise ValueError(f"Unsupported perceval detector type: {component.type}")


def postselection(circuit, p):
    measure = Measure(circuit.cod)
    n_post = len(circuit.cod)

    copy = Spider(1, 2, mode)**n_post
    permutation = Diagram.permutation(
        list(range(0, 2*n_post, 2)) + list(range(1, 2*n_post, 2)),
        mode**(2*n_post)
    )
    measure >>= copy >> permutation

    postselect_f = ClassicalFunction(
        compile_postselect(str(p.post_select_fn)), mode**n_post, bit
    )
    measure >>= postselect_f @ mode**n_post

    postselection = BitControlledGate(
        Diagram.id(mode**n_post),
        Channel("PostSelection", Endo(0)**n_post, mode**n_post, mode**n_post),
        classical=True
    )
    measure >>= postselection

    return measure
