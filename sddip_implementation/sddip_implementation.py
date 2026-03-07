"""MSIP construction utilities for the hydrogen study case.

This module builds an MSPPy ``MSIP`` model with the same stage structure as
the attached Pyomo stochastic model:
stage 1 booking, stage 2 booking, and stage 3 operations + booking.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


def _incoming_arcs(nodes: Iterable[str], arcs: Sequence[Tuple[str, str]]) -> Dict[str, List[Tuple[str, str]]]:
    result = {n: [] for n in nodes}
    for a in arcs:
        result[a[1]].append(a)
    return result


def _outgoing_arcs(nodes: Iterable[str], arcs: Sequence[Tuple[str, str]]) -> Dict[str, List[Tuple[str, str]]]:
    result = {n: [] for n in nodes}
    for a in arcs:
        result[a[0]].append(a)
    return result


def _scenario_booking_costs(stage_scenarios: Sequence[object], node: str) -> List[float]:
    return [float(s.G.nodes[node].get("booking_cost", 0.0)) for s in stage_scenarios]


def _market_demand_rhs(stage3_scenarios: Sequence[object], market_node: str) -> List[float]:
    rhs = []
    for s in stage3_scenarios:
        demand_dict = s.G.nodes[market_node].get("demand", {})
        rhs.append(float(sum(demand_dict.values())))
    return rhs


def _supplier_rhs(stage3_scenarios: Sequence[object], market_nodes: Sequence[str], supplier: str) -> List[float]:
    rhs = []
    for s in stage3_scenarios:
        total = 0.0
        for n in market_nodes:
            total += float(s.G.nodes[n].get("demand", {}).get(supplier, 0.0))
        rhs.append(total)
    return rhs


def construct_study_case_msip(
    network,
    scenarios: Mapping[int, Sequence[object]],
    bound: float = 1.0e10,
):
    """Create an MSPPy MSIP model from the study-case Pyomo data objects.

    Parameters
    ----------
    network:
        Directed graph (same object type as used by the Pyomo model).
    scenarios:
        Scenario dictionary keyed by stage index ``{1, 2, 3}``, where each
        entry is a list of scenario objects with attributes ``G``, ``index``,
        and ``probability``.
    bound:
        Global model bound passed into ``MSIP``.

    Returns
    -------
    MSIP
        A 3-stage mixed-integer stochastic programming object.

    Notes
    -----
    This reproduces the stochastic booking and core flow constraints. The
    pressure/Weymouth/quality blocks from the Pyomo model are not included in
    this first MSIP mapping.
    """
    from msppy.msp import MSIP

    components = ("NG", "CO2", "H2")
    gcv = {"NG": 39.8 / 3.6 * 1000.0, "H2": 12.7 / 3.6 * 1000.0, "CO2": 0.0}

    nodes = list(network.nodes)
    arcs = list(network.edges)
    incoming = _incoming_arcs(nodes, arcs)
    outgoing = _outgoing_arcs(nodes, arcs)

    supply_nodes = [n for n, d in network.nodes(data=True) if d.get("supply_capacity", None) is not None]
    market_nodes = [n for n, d in network.nodes(data=True) if d.get("max_fractions", None) is not None]
    compression_nodes = [n for n, d in network.nodes(data=True) if d.get("compression_increase", None) is not None]
    intermediate_nodes = [n for n in nodes if n not in supply_nodes and n not in market_nodes and n not in compression_nodes]

    stage1 = scenarios[1]
    stage2 = scenarios[2]
    stage3 = scenarios[3]

    suppliers = sorted(
        {
            supplier
            for n in market_nodes
            for supplier in stage3[0].G.nodes[n].get("demand", {}).keys()
        }
    )
    supplier_to_supply_nodes = {
        h: [n for n in supply_nodes if network.nodes[n].get("supplier", "Unknown") == h] for h in suppliers
    }

    msp = MSIP(T=3, sense=-1, bound=bound)

    for t in range(3):
        m = msp[t]

        book_entry_now = {}
        book_exit_now = {}
        x_entry_new = {}
        x_exit_new = {}

        # Stage-wise booking decisions and state transitions.
        for n in nodes:
            entry_now, entry_past = m.addStateVar(name=f"book_entry[{n}]", lb=0.0)
            exit_now, exit_past = m.addStateVar(name=f"book_exit[{n}]", lb=0.0)
            book_entry_now[n] = entry_now
            book_exit_now[n] = exit_now

            if t == 0:
                cost = _scenario_booking_costs(stage1, n)[0]
                x_entry_new[n] = m.addVar(name=f"x_entry_new[{n}]", lb=0.0, obj=-cost)
                x_exit_new[n] = m.addVar(name=f"x_exit_new[{n}]", lb=0.0, obj=-cost)
            elif t == 1:
                costs = _scenario_booking_costs(stage2, n)
                x_entry_new[n] = m.addVar(name=f"x_entry_new[{n}]", lb=0.0, uncertainty={"obj": [-c for c in costs]})
                x_exit_new[n] = m.addVar(name=f"x_exit_new[{n}]", lb=0.0, uncertainty={"obj": [-c for c in costs]})
            else:
                costs = _scenario_booking_costs(stage3, n)
                x_entry_new[n] = m.addVar(name=f"x_entry_new[{n}]", lb=0.0, uncertainty={"obj": [-c for c in costs]})
                x_exit_new[n] = m.addVar(name=f"x_exit_new[{n}]", lb=0.0, uncertainty={"obj": [-c for c in costs]})

            m.addConstr(entry_now == entry_past + x_entry_new[n])
            m.addConstr(exit_now == exit_past + x_exit_new[n])

        # Booking balance per stage.
        m.addConstr(sum(x_entry_new[n] for n in nodes) == sum(x_exit_new[n] for n in nodes))

        # Stage-3 operational model with stochastic demand/price/cost.
        if t == 2:
            flow = {}
            for a in arcs:
                i, j = a
                for c in components:
                    coeffs = []
                    for s in stage3:
                        price = float(s.G.nodes[j].get("price", 0.0)) if j in market_nodes else 0.0
                        gen_cost = float(s.G.nodes[i].get("generation_cost", 0.0)) if i in supply_nodes else 0.0
                        coeffs.append(gcv[c] * price - gen_cost)
                    flow[a, c] = m.addVar(
                        name=f"f[{i},{j},{c}]",
                        lb=0.0,
                        uncertainty={"obj": coeffs},
                    )

            # Supply production capacity.
            for n in supply_nodes:
                cap = float(network.nodes[n]["supply_capacity"])
                m.addConstr(
                    sum(flow[a, c] for a in outgoing[n] for c in components)
                    - sum(flow[a, c] for a in incoming[n] for c in components)
                    <= cap
                )

            # Supply component ratios.
            for n in supply_nodes:
                alpha = network.nodes[n].get("component_ratio", {})
                for c in components:
                    ratio = float(alpha.get(c, 0.0))
                    out_total = sum(flow[a, cp] for a in outgoing[n] for cp in components)
                    in_total = sum(flow[a, cp] for a in incoming[n] for cp in components)
                    m.addConstr(
                        sum(flow[a, c] for a in outgoing[n]) - sum(flow[a, c] for a in incoming[n])
                        == ratio * (out_total - in_total)
                    )

            # Market demand satisfaction (stochastic RHS).
            for n in market_nodes:
                demand_rhs = _market_demand_rhs(stage3, n)
                lhs = sum(gcv[c] * flow[a, c] for a in incoming[n] for c in components)
                m.addConstr(lhs >= 0.0, uncertainty={"rhs": demand_rhs})

            # Supplier demand satisfaction (stochastic RHS).
            for h in suppliers:
                source_nodes = supplier_to_supply_nodes[h]
                lhs = sum(
                    gcv[c] * (sum(flow[a, c] for a in outgoing[n]) - sum(flow[a, c] for a in incoming[n]))
                    for n in source_nodes
                    for c in components
                )
                m.addConstr(lhs >= 0.0, uncertainty={"rhs": _supplier_rhs(stage3, market_nodes, h)})

            # Flow balance for intermediate nodes and compression nodes.
            for n in intermediate_nodes + compression_nodes:
                for c in components:
                    m.addConstr(sum(flow[a, c] for a in incoming[n]) == sum(flow[a, c] for a in outgoing[n]))

            # Booking-flow consistency.
            for n in nodes:
                net_in_minus_out = (
                    sum(flow[a, c] for a in incoming[n] for c in components)
                    - sum(flow[a, c] for a in outgoing[n] for c in components)
                )
                m.addConstr(book_entry_now[n] >= net_in_minus_out)
                m.addConstr(book_exit_now[n] >= -net_in_minus_out)

    return msp


def construct_nvidi():
    """Stage-wise independent finite discrete integer example."""
    from msppy.msp import MSIP

    nvidi = MSIP(T=2, sense=-1, bound=100)
    for t in range(2):
        m = nvidi[t]
        buy_now, buy_past = m.addStateVar(name="bought", obj=-1.0, vtype="I")
        if t == 1:
            sold = m.addVar(name="sold", obj=2, vtype="I")
            unsatisfied = m.addVar(name="unsatisfied", vtype="I")
            recycled = m.addVar(name="recycled", obj=0.5, vtype="I")
            m.addConstr(sold + unsatisfied == 0, uncertainty={"rhs": range(11)})
            m.addConstr(sold + recycled == buy_past)
    return nvidi
