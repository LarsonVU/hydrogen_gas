import pyomo.environ as pyo
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import matplotlib.colors as mcolors
import re
from pathlib import Path

# Soft pastel palette
PASTEL_COLORS = [
    "#82C9FF",  # pastel blue
    "#FF8692",  # pastel red
    "#4BDA6A",  # pastel green
    "#DB97E3",  # pastel purple
    "#FFFF82",  # pastel yellow
    "#FFC085",
    "#7EDCD5"
]

def safe_filename(s):
    return re.sub(r'[\\/*?:"<>|]', "_", str(s))

def adjust_color(color, factor=0.7):
    """
    Darken (<1) or lighten (>1) a color.
    """
    rgb = np.array(mcolors.to_rgb(color))
    if factor < 1:
        return tuple(rgb * factor)  # darken
    else:
        return tuple(1 - (1 - rgb) / factor)  # lighten

def plot_average_flows(model, folder="figures/", show=False):
    """
    Plot average flows per arc and per component with standard deviation across scenarios.
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    arcs = list(model.A)
    components = list(model.C)

    avg_flows_per_component = {c: [] for c in components}
    std_flows_per_component = {c: [] for c in components}

    for c in components:
        for a in arcs:
            flows = [pyo.value(model.f[a, c, m_3]) for m_3 in model.M[3]]
            probs = [pyo.value(model.sp[3, m_3]) for m_3 in model.M[3]]

            # weighted average
            avg_flow = sum(f * p for f, p in zip(flows, probs))

            avg_flows_per_component[c].append(avg_flow)
            std_flows_per_component[c].append(np.std(flows))

    arc_labels = [f"{a[0]}-{a[1]}" for a in arcs]
    x_pos = list(range(len(arcs)))
    bar_width = 0.25

    for i, c in enumerate(components):
        offset = (i - len(components) / 2 + 0.5) * bar_width

        ax.bar(
            [x + offset for x in x_pos],
            avg_flows_per_component[c],
            yerr=std_flows_per_component[c],
            capsize=5,
            alpha=0.7,
            label=c,
            color=PASTEL_COLORS[i % len(PASTEL_COLORS)],
            width=bar_width,
            error_kw={"elinewidth": 2},
        )

    ax.set_xlabel("Arc", fontsize=12)
    ax.set_ylabel("Flow (units)", fontsize=12)
    ax.set_title("Average Flow per Arc per Component with Scenario Std Dev", fontsize=14)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(arc_labels, rotation=45, ha="right")

    ax.legend(title="Component")

    plt.tight_layout()
    plt.savefig(folder / "average_flows.png")

    if show:
        plt.show()

    plt.close(fig)

def plot_component_flows_stacked(model, folder="figures/",  show = False):
    """
    Plot total flows per scenario as a stacked bar per component across all market nodes.
    """

    components = list(model.C)
    scenarios = list(model.M[3])
    n_scenarios = len(scenarios)

    # Compute total flow per component per scenario
    flow_matrix = np.zeros((len(components), n_scenarios))  # rows: components, cols: scenarios

    for j, m_3 in enumerate(scenarios):
        total_flow_all_components = 0
        for i, c in enumerate(components):
            flow_c = sum(
                pyo.value(model.f[a, c, m_3])
                for n in model.N_m
                for a in model.A_n_plus[n]
            )
            flow_matrix[i, j] = flow_c
            total_flow_all_components += flow_c

    # Plot stacked bar
    fig, ax = plt.subplots(figsize=(14, 6))
    bottom = np.zeros(n_scenarios)

    for i, c in enumerate(components):
        ax.bar(range(n_scenarios), flow_matrix[i], bottom=bottom, label=c, color=PASTEL_COLORS[i % len(PASTEL_COLORS)])
        bottom += flow_matrix[i]

    ax.set_xlabel('Scenario Index', fontsize=12)
    ax.set_ylabel('Total Flow', fontsize=12)
    ax.set_title('Total Flow per Scenario Stacked by Component', fontsize=14)
    ax.legend(title='Component')
    ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(folder / "scenario_component_flow_stacked.png")
    if show:
        plt.show()
    plt.close(fig)

def plot_inlet_outlet_pressures(model, folder, show = False):
    arcs = list(model.A)
    arc_labels = [f"{a[0]}-{a[1]}" for a in arcs]
    x_pos = range(len(arcs))
    width = 0.35

    # Compute average, min, max pressures per arc across scenarios
    avg_p_in = []
    avg_p_out = []
    min_p_in = []
    max_p_in = []
    min_p_out = []
    max_p_out = []

    for a in arcs:
        p_in_vals = [pyo.value(model.p_in[a, m_3]) for m_3 in model.M[3]]
        p_out_vals = [pyo.value(model.p_out[a, m_3]) for m_3 in model.M[3]]

        avg_p_in.append(np.mean(p_in_vals))
        min_p_in.append(min(p_in_vals))
        max_p_in.append(max(p_in_vals))

        avg_p_out.append(np.mean(p_out_vals))
        min_p_out.append(min(p_out_vals))
        max_p_out.append(max(p_out_vals))

    # Calculate error bars
    yerr_in = [np.array(avg_p_in) - np.array(min_p_in), np.array(max_p_in) - np.array(avg_p_in)]
    yerr_out = [np.array(avg_p_out) - np.array(min_p_out), np.array(max_p_out) - np.array(avg_p_out)]

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar([x - width/2 for x in x_pos], avg_p_in, width, 
           yerr=yerr_in, capsize=5, label='Inlet Pressure', color=PASTEL_COLORS[0], alpha=0.7)
    ax.bar([x + width/2 for x in x_pos], avg_p_out, width, 
           yerr=yerr_out, capsize=5, label='Outlet Pressure', color=PASTEL_COLORS[1], alpha=0.7)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(arc_labels, rotation=45, ha='right')
    ax.set_xlabel("Pipeline")
    ax.set_yticks(np.arange(0, 151, 30))
    ax.set_ylim(0, 150)
    ax.set_ylabel("Pressure (bar)")
    ax.set_title("Average Inlet and Outlet Pressure per Arc (with min/max error bars)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(folder / "inlet_outlet_pressure_plot.png")
    if show:
        plt.show()
    plt.close(fig)

def plot_inlet_outlet_pressure_violins_example(model, folder, show=False):
    selected_arcs = [("ZEEPIPE-SCP", "ZEEBRUGGE"), ("B-11", "H-7 BP"), ("VISUND", "VISUND T")]
    os.makedirs(folder, exist_ok=True)

    arcs = list(model.A)

    # =========================
    # Collect inlet/outlet + variation
    # =========================
    arc_data = []

    for a in arcs:
        p_in_vals = np.array([pyo.value(model.p_in[a, m_3]) for m_3 in model.M[3]])
        p_out_vals = np.array([pyo.value(model.p_out[a, m_3]) for m_3 in model.M[3]])

        if len(p_in_vals) == 0:
            continue

        variation = np.max(p_in_vals) - np.min(p_in_vals)

        arc_data.append({
            "arc": a,
            "label": f"{a[0]}- \n {a[1]}",
            "p_in": p_in_vals,
            "p_out": p_out_vals,
            "variation": variation
        })

    if len(arc_data) == 0:
        print("No arc data found.")
        return

    # =========================
    # Select arcs
    # =========================
    if selected_arcs is not None:
        # Filter manually selected arcs
        arc_data_selected = [d for d in arc_data if d["arc"] in selected_arcs]

        if len(arc_data_selected) == 0:
            raise ValueError("None of the selected arcs were found in model.A")

    else:
        # Fallback: min / median / max variation
        if len(arc_data) < 3:
            print("Not enough arcs to select min/median/max variation.")
            return

        arc_data_sorted = sorted(arc_data, key=lambda x: x["variation"])

        arc_data_selected = [
            arc_data_sorted[0],
            arc_data_sorted[len(arc_data_sorted) // 2],
            arc_data_sorted[-1]
        ]

    # =========================
    # Prepare plotting data
    # =========================
    data = []
    labels = []
    positions = []

    pos = 1
    spacing = 1.5

    for d in arc_data_selected:
        # inlet
        data.append(d["p_in"])
        labels.append(f"{d['label']} \n (Inlet)")
        positions.append(pos)

        # outlet
        data.append(d["p_out"])
        labels.append(f"{d['label']} \n (Outlet)")
        positions.append(pos+ 0.6)

        pos += spacing

    # =========================
    # Plot vertical violins (swapped from horizontal)
    # =========================
    fig, ax = plt.subplots(figsize=(10, 6))

    parts = ax.violinplot(
        data,
        positions=positions,
        vert=True,
        showmeans=False,
        showmedians=False,
        showextrema=False
    )

    # Styling
    for i, pc in enumerate(parts['bodies']):
        color_idx = i % 2
        pc.set_facecolor(PASTEL_COLORS[color_idx % len(PASTEL_COLORS)])
        pc.set_edgecolor(PASTEL_COLORS[color_idx % len(PASTEL_COLORS)])
        pc.set_alpha(0.7)

    # =========================
    # Overlay TRUE data points (color-matched)
    # =========================
    for i, vals in enumerate(data):
        vals = np.array(vals)

        sorted_vals = np.sort(vals)
        unique_vals, counts = np.unique(sorted_vals, return_counts=True)

        x_positions = []
        y_positions = []

        for val, count in zip(unique_vals, counts):
            offsets = [0] if count == 1 else np.linspace(-0.08, 0.08, count)
            for offset in offsets:
                x_positions.append(positions[i] + offset)
                y_positions.append(val)

        # Match violin color, but slightly darker for contrast
        base_color = PASTEL_COLORS[i % 2]
        point_color = adjust_color(base_color, factor=0.9)

        ax.scatter(
            x_positions,
            y_positions,
            s=14,
            alpha=1,
            color=point_color,
            edgecolors='none'
        )

    # =========================
    # Labels
    # =========================
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.grid(alpha=0.3, axis="y")

    ax.set_ylabel("Pressure (bar)")
    ax.set_title(
        "Inlet and Outlet Pressure Distributions"
    )
    ax.set_ylim(bottom=40, top=150)
    ax.set_yticks(np.append(np.arange(40, 151, 20), 150))

    plt.tight_layout()
    plt.savefig(folder / "inlet_outlet_pressure_violin_example.png")

    if show:
        plt.show()

    plt.close(fig)

def plot_total_vs_weymouth_histogram(model, folder="figures/", show = False):
    """
    Plot a histogram of deviations between total_flow and weymouth_flow
    across all arcs and scenarios.
    """
    deviations = []

    # Collect deviations for all arcs and all scenarios
    for a in model.A:
        for m_3 in model.M[3]:
            total = float(pyo.value(model.total_flow[a, m_3]))
            weymouth = float(pyo.value(model.weymouth_flow[a, m_3]))
            if total > 0:
                deviation = ((total - weymouth) / total)
            else:
                deviation = 0
            deviations.append(deviation *100) #(convert to percentage)

    deviations = np.array(deviations)

    # Print summary stats
    # print("Deviation statistics:")
    # print("Min:", deviations.min())
    # print("Max:", deviations.max())
    # print("Mean:", deviations.mean())
    # print("Std:", deviations.std())

    # Plot histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(deviations, bins=30, color=PASTEL_COLORS[2], edgecolor='black', alpha=0.7)

    ax.set_xlabel("Deviation (%)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("Histogram of Deviations Across All Arcs and Scenarios", fontsize=14)
    ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(folder / "total_vs_weymouth_deviation_histogram.png")
    if show:
        plt.show()
    plt.close(fig)

def plot_market_node_component_flow_with_std(model, folder="figures/", show = False):
    """
    Plot mean inflow per market node per component.
    Bars are grouped (not stacked).
    Error bars show std across scenarios.
    """

    market_nodes = list(model.N_m)
    components = list(model.C)
    scenarios = list(model.M[3])

    n_nodes = len(market_nodes)
    n_components = len(components)

    means = np.zeros((n_nodes, n_components))
    stds = np.zeros((n_nodes, n_components))

    # Compute statistics
    for i, n in enumerate(market_nodes):
        for j, c in enumerate(components):

            flows_per_scenario = []

            for m_3 in scenarios:
                total_flow = sum(
                    pyo.value(model.f[a, c, m_3])
                    for a in model.A_n_plus[n]
                )
                flows_per_scenario.append(total_flow)

            flows_per_scenario = np.array(flows_per_scenario)
            means[i, j] = flows_per_scenario.mean()
            stds[i, j] = flows_per_scenario.std()

    # Plot
    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(n_nodes)
    width = 0.8 / n_components  # adaptive width

    for j, c in enumerate(components):
        ax.bar(
            x + j * width - (n_components - 1) * width / 2,
            means[:, j],
            width,
            yerr=stds[:, j],
            capsize=4,
            label=c,
            color=PASTEL_COLORS[i % len(PASTEL_COLORS)]
        )

    ax.set_xticks(x)
    ax.set_xticklabels(market_nodes, rotation=45, ha="right")
    ax.set_xlabel("Market Node", fontsize=12)
    ax.set_ylabel("Mean Inflow", fontsize=12)
    ax.set_title("Mean Inflow per Market Node by Component (Std Across Scenarios)", fontsize=14)
    ax.legend(title="Component")
    ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(folder / "market_node_component_flow_mean_std.png")
    if show:
        plt.show()
    plt.close(fig)

def plot_supplier_limit_vs_produced(model, folder="figures/", show = False):
    """
    For each supplier node (model.N_hg):
    Plot total required demand vs actual produced amount per scenario.
    """
    os.makedirs(folder, exist_ok=True)

    suppliers = list(model.N_hg)
    scenarios = list(model.M[3])
    components = list(model.C)

    for n in suppliers:

        produced = []
        required = []

        for m_3 in scenarios:

            # --- Produced: net outflow ---
            outflow = sum(
                pyo.value(model.f[a, c, m_3])
                for a in model.A_n_minus[n]
                for c in components
            )

            inflow = sum(
                pyo.value(model.f[a, c, m_3])
                for a in model.A_n_plus[n]
                for c in components
            )

            produced.append(outflow - inflow)

            # --- Required ---
            # Replace this if your demand definition differs
            if hasattr(model, "G"):
                required.append(pyo.value(model.G[n]))
            else:
                required.append(0)

        # ---- Plot ----
        x = np.arange(len(scenarios))
        width = 0.35

        fig, ax = plt.subplots(figsize=(14, 6))

        ax.bar(x + width/2, produced, width, label="Produced", color = PASTEL_COLORS[0])
        ax.bar(x - width/2, required, width, label="Limit", color = PASTEL_COLORS[1])

        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s in scenarios], rotation=45)
        ax.set_xlabel("Scenario")
        ax.set_ylabel("Total Flow")
        ax.set_title(f"Limit vs Produced – Supplier {n}")
        ax.legend()
        ax.grid(alpha=0.3, axis='y')

        plt.tight_layout()
        safe_n = safe_filename(n)
        plt.savefig(folder / f"supplier_{safe_n}_limit_vs_produced.png")
        if show:
            plt.show()
        plt.close(fig)


def plot_supplier_total_prod_vs_total_demand(model, folder="figures/", show = False):
    """
    For each supplier s:
        total_prod(s, scenario) = sum net production of all gen nodes owned by s
        total_demand(scenario) = total system demand (same for all suppliers)

    Plots both across scenarios.
    """
    os.makedirs(folder, exist_ok=True)

    suppliers = list(model.H) 
    gen_nodes = list(model.N_hg)
    market_nodes = list(model.N_m)
    scenarios = list(model.M[3])
    components = list(model.C)

    for h in suppliers:

        prod_per_scenario = []
        demand_per_scenario = []

        for m_3 in scenarios:

            # ---------------------------
            # Total production of supplier s
            # ---------------------------
            total_prod = 0

            for g in gen_nodes:
                if model.supplier[g] != h:
                    continue

                inflow = sum(
                    pyo.value(model.f[a, c, m_3])  * model.gcv_c[c]
                    for a in model.A_n_plus[g]
                    for c in components
                )

                outflow = sum(
                    pyo.value(model.f[a, c, m_3]) * model.gcv_c[c]
                    for a in model.A_n_minus[g]
                    for c in components
                )

                total_prod += (outflow - inflow)

            prod_per_scenario.append(total_prod)

            # ---------------------------
            # Total demand (system-wide)
            # ---------------------------
            total_demand = sum(
                pyo.value(model.D[h, n, m_3])
                for n in market_nodes
            )

            demand_per_scenario.append(total_demand)

        # ---------------------------
        # Plot
        # ---------------------------
        x = np.arange(len(scenarios))

        fig, ax = plt.subplots(figsize=(14, 6))

        ax.plot(x, prod_per_scenario, marker="o", label=f"Production ({h})", color=PASTEL_COLORS[0])
        ax.plot(x, demand_per_scenario, linestyle="--", label="Total Demand", color=PASTEL_COLORS[1])

        ax.set_xticks(x)
        ax.set_xticklabels([str(sc) for sc in scenarios], rotation=45)
        ax.set_xlabel("Scenario")
        ax.set_ylabel("Flow")
        ax.set_ylim(0)
        ax.set_title(f"Supplier {h}: Production vs Total Demand")
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(folder / f"supplier_{h}_prod_vs_total_demand.png")
        if show:
            plt.show()
        plt.close(fig)

def plot_scenario_objectives(model, folder="figures/", show = False):
    """
    Plot scenario objective values as a bar chart.
    """

    scenarios = list(model.M[3])
    n_scenarios = len(scenarios)

    # Collect objective values per scenario
    obj_values = np.zeros(n_scenarios)

    for j, m_3 in enumerate(scenarios):
        obj_values[j] = pyo.value(model.scenario_objective[m_3])

    # Plot bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(range(n_scenarios), obj_values, color = PASTEL_COLORS[0])

    ax.set_xlabel('Scenario Index', fontsize=12)
    ax.set_ylabel('Objective Value', fontsize=12)
    ax.set_title('Scenario Objective Value per Scenario', fontsize=14)
    ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(folder / "scenario_objectives_bar.png")
    if show:
        plt.show()
    plt.close(fig)

def plot_scenario_revenue_costs(model, folder="figures/", show=False):
    """
    Plot revenue (objective) on primary y-axis and costs on secondary y-axis.
    Revenue gets its own subplot/axis, costs share a secondary axis.
    """

    os.makedirs(folder, exist_ok=True)

    scenarios = list(model.M[3])

    revenue = []
    generation_cost = []
    pressure_cost = []
    booking_cost = []

    # Collect values per scenario
    for m_3 in scenarios:
        revenue.append(pyo.value(model.revenue_scenario[m_3]))
        generation_cost.append(pyo.value(model.generation_scenario[m_3]))
        pressure_cost.append(pyo.value(model.pressure_scenario[m_3]))
        booking_cost.append(pyo.value(model.booking_scenario[m_3]))

    fig = plt.figure(figsize=(16, 6))
    
    # Revenue subplot: 1/4 of the width
    ax1 = fig.add_subplot(1, 5, 1)
    
    # Costs subplot: 3/4 of the width
    ax2 = fig.add_subplot(1, 5, (2, 5))

    # ============================================
    # LEFT SUBPLOT: Revenue (Primary)
    # ============================================
    positions_revenue = [1]
    revenue_data = [np.array(revenue) / 1000000]
    
    parts1 = ax1.violinplot(
        revenue_data,
        positions=positions_revenue,
        vert=True,
        showmeans=False,
        showmedians=False,
        showextrema=False
    )

    # Color revenue violin
    for pc in parts1['bodies']:
        pc.set_facecolor(PASTEL_COLORS[0])
        pc.set_edgecolor(PASTEL_COLORS[0])
        pc.set_alpha(0.7)

    # Overlay revenue data points
    revenue_vals = np.array(revenue) / 1000000
    sorted_vals = np.sort(revenue_vals)
    unique_vals, counts = np.unique(sorted_vals, return_counts=True)

    x_positions = []
    y_positions = []

    for val, count in zip(unique_vals, counts):
        offsets = [0] if count == 1 else np.linspace(-0.08, 0.08, count)
        for offset in offsets:
            x_positions.append(positions_revenue[0] + offset)
            y_positions.append(val)

    point_color = adjust_color(PASTEL_COLORS[0], factor=0.9)
    ax1.scatter(
        x_positions,
        y_positions,
        s=14,
        alpha=0.6,
        color=point_color,
        edgecolors='none'
    )

    ax1.set_ylabel("Revenue (Million Euro)", fontsize=12, color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.set_xticks(positions_revenue)
    ax1.set_xticklabels(["Revenue"])
    ax1.set_title("Revenue Distribution", fontsize=14)
    ax1.set_ylim(bottom=0, top = max(25, 1.01 * max(y_positions)))
    ax1.grid(alpha=0.3, axis="y")

    # ============================================
    # RIGHT SUBPLOT: Costs
    # ============================================
    positions_costs = [1, 2, 3]
    costs_data = [np.array(generation_cost) / 1000000, np.array(pressure_cost) / 1000000, np.array(booking_cost) / 1000000]
    cost_labels = ["Generation Cost", "Pressure Cost", "Booking Cost"]
    cost_colors = PASTEL_COLORS[1:4]

    parts2 = ax2.violinplot(
        costs_data,
        positions=positions_costs,
        vert=True,
        showmeans=False,
        showmedians=False,
        showextrema=False
    )

    # Color each cost violin
    for i, pc in enumerate(parts2['bodies']):
        pc.set_facecolor(cost_colors[i])
        pc.set_edgecolor(cost_colors[i])
        pc.set_alpha(0.7)

    # Overlay cost data points
    for i, vals in enumerate(costs_data):
        vals = np.array(vals)       

        sorted_vals = np.sort(vals)
        unique_vals, counts = np.unique(sorted_vals, return_counts=True)

        x_positions = []
        y_positions = []

        for val, count in zip(unique_vals, counts):
            offsets = [0] if count == 1 else np.linspace(-0.08, 0.08, count)
            
            for offset in offsets:
                x_positions.append(positions_costs[i] + offset)
                y_positions.append(val)

            # Match violin color, but slightly darker for contrast
            base_color = cost_colors[i]
            point_color = adjust_color(base_color, factor=0.9)

            ax2.scatter(
            x_positions,
            y_positions,
            s=14,
            alpha=0.6,
            color=point_color,
            edgecolors='none'
            )

    ax2.set_ylabel("Costs (Million Euro)", fontsize=12, color='black')
    ax2.tick_params(axis='y', labelcolor='black')
    ax2.set_xticks(positions_costs)
    ax2.set_xticklabels(cost_labels)
    ax2.set_title("Cost Distribution", fontsize=14)
    ax2.set_ylim(bottom=0, top =2)
    ax2.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(folder / "scenario_revenue_costs_violins.png")

    if show:
        plt.show()

    plt.close(fig)

def plot_entry_exit_capacity(model, folder="figures/", show=False):
    """
    Plot entry and exit capacity bought per node.

    - Separate plots for entry and exit
    - Bars are stacked by stage (capacity bought)
    - Line overlay = actual flow
    - Nodes included only if respective capacity > 0
    - Nodes sorted by total capacity (descending)
    """
    os.makedirs(folder, exist_ok=True)

    data = []

    # =========================
    # Collect capacity data
    # =========================
    for n in model.N:
        for k in model.K:
            values = [
                pyo.value(model.x_entry[n, k, m_k])
                for m_k in model.M[k]
                ]
            entry = np.mean(values)
            values = [
                pyo.value(model.x_exit[n, k, m_k])
                for m_k in model.M[k]
                ]
            exit_ = np.mean(values)

            data.append({
                "node": n,
                "stage": k,
                "entry": entry,
                "exit": exit_
            })

    df = pd.DataFrame(data)

    # =========================
    # Collect FLOW data
    # =========================
    flow_dif = {}

    for n in model.N:
        values = [sum(pyo.value(model.f[a, c, m_3]) for a in model.A_n_minus[n] for c in model.C) - sum(pyo.value(model.f[a, c, m_3]) for a in model.A_n_plus[n] for c in model.C) for m_3 in model.M[3]]
        flow_dif[n] = np.mean(values)

    # =========================
    # ENTRY CAPACITY PLOT
    # =========================
    entry_totals = df.groupby("node")["entry"].sum()
    entry_nodes = entry_totals[entry_totals > 0].sort_values(ascending=False).index
    entry_nodes = [n for n in entry_nodes if any(flow_dif[n] > 0 for n in [n])]

    if len(entry_nodes) > 0:
        entry_df = df[df["node"].isin(entry_nodes)]

        entry_pivot = entry_df.pivot(index="node", columns="stage", values="entry").fillna(0)
        entry_pivot = entry_pivot.loc[entry_nodes]

        fig, ax = plt.subplots(figsize=(12, 6))

        # Stacked bars (capacity)
        entry_pivot.columns = [f"Stage {k}" for k in entry_pivot.columns]
        entry_pivot.plot(kind="bar", stacked=True, ax=ax, color=PASTEL_COLORS[:len(entry_pivot.columns)])

        ax.set_title("Entry Capacity Bought per Stage")
        ax.set_xlabel("Node")
        ax.set_ylim(0, max(26, max(entry_totals) * 1.1))
        ax.set_xticklabels(entry_nodes, rotation=0, ha="center")
        ax.set_ylabel("Entry Capacity (Mscm)")
        ax.legend()
        ax.grid(alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig(folder / "entry_capacity_per_node.png")

        if show:
            plt.show()

        plt.close(fig)

    else:
        print("No entry capacity bought at any node.")

    # =========================
    # EXIT CAPACITY PLOT
    # =========================
    exit_totals = df.groupby("node")["exit"].sum()
    exit_nodes = exit_totals[exit_totals > 0].sort_values(ascending=False).index
    exit_nodes = [n for n in exit_nodes if any(flow_dif[n] < -0.01 for n in [n])]

    if len(exit_nodes) > 0:
        exit_df = df[df["node"].isin(exit_nodes)]

        exit_pivot = exit_df.pivot(index="node", columns="stage", values="exit").fillna(0)
        exit_pivot = exit_pivot.loc[exit_nodes]

        fig, ax = plt.subplots(figsize=(12, 6))

        # Stacked bars (capacity)
        exit_pivot.columns = [f"Stage {k}" for k in exit_pivot.columns]
        exit_pivot.plot(kind="bar", stacked=True, ax=ax, color=PASTEL_COLORS[:len(exit_pivot.columns)])

        ax.set_title("Exit Capacity Bought per Stage")
        ax.set_xlabel("Node")
        ax.set_xticklabels(exit_nodes, rotation=0, ha="center")
        ax.set_ylim(0, max(21, max(exit_totals) * 1.1))
        ax.set_ylabel("Exit Capacity (Mscm)")
        ax.legend()
        ax.grid(alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig(folder / "exit_capacity_per_node.png")

        if show:
            plt.show()

        plt.close(fig)

    else:
        print("No exit capacity bought at any node.")

def plot_overdraft_violins(model, folder="figures/", show=False):
    """
    Violin plots of overdraft per node.

    - Capacity is summed over ALL stages
    - Overdraft computed per node per scenario
    - Distribution across nodes and scenarios
    """

    os.makedirs(folder, exist_ok=True)

    entry_overdraft = []
    exit_overdraft = []

    # =========================
    # Precompute TOTAL capacity per node (sum over stages)
    # =========================
    total_entry_cap = {}
    total_exit_cap = {}

    for n in model.N:
        entry_sum = 0
        exit_sum = 0

        for k in model.K:
            # average over scenarios of that stage
            entry_vals = [pyo.value(model.x_entry[n, k, m_k]) for m_k in model.M[k]]
            exit_vals = [pyo.value(model.x_exit[n, k, m_k]) for m_k in model.M[k]]

            entry_sum += np.mean(entry_vals)
            exit_sum += np.mean(exit_vals)

        total_entry_cap[n] = entry_sum
        total_exit_cap[n] = exit_sum

    # =========================
    # Compute overdraft per node per scenario
    # =========================
    # Use final stage scenarios (full uncertainty realized)
    final_stage = max(model.K)

    for n in model.N:
        for m in model.M[final_stage]:

            inflow = sum(pyo.value(model.f[a, c, m])
                         for a in model.A_n_minus[n] for c in model.C)
            outflow = sum(pyo.value(model.f[a, c, m])
                          for a in model.A_n_plus[n] for c in model.C)

            net_flow = inflow - outflow

            entry_od = max(0, net_flow - total_entry_cap[n])
            exit_od = max(0, -net_flow - total_exit_cap[n])

            entry_overdraft.append(entry_od)
            exit_overdraft.append(exit_od)

    # =========================
    # ENTRY OVERDRAFT PLOT
    # =========================
    if len(entry_overdraft) > 0:
        fig, ax = plt.subplots(figsize=(8, 6))

        ax.violinplot([entry_overdraft], showmeans=True, showextrema=True)

        ax.set_xticks([1])
        ax.set_xticklabels(["Entry"])

        ax.set_title("Distribution of Entry Overdraft (Total Capacity)")
        ax.set_ylabel("Overdraft Amount")

        ax.grid(alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig(folder / "entry_overdraft_violin.png")

        if show:
            plt.show()

        plt.close(fig)
    else:
        print("No entry overdraft data available.")

    # =========================
    # EXIT OVERDRAFT PLOT
    # =========================
    if len(exit_overdraft) > 0:
        fig, ax = plt.subplots(figsize=(8, 6))

        ax.violinplot([exit_overdraft], showmeans=True, showextrema=True)

        ax.set_xticks([1])
        ax.set_xticklabels(["Exit"])

        ax.set_title("Distribution of Exit Overdraft (Total Capacity)")
        ax.set_ylabel("Overdraft Amount")

        ax.grid(alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig(folder / "exit_overdraft_violin.png")

        if show:
            plt.show()

        plt.close(fig)
    else:
        print("No exit overdraft data available.")

def plot_flow_violins(model, folder="figures/", show=False, tol = 0.01):
    """
    Violin plots of NET flow per node.

    - Inflow plot: only nodes where inflow > outflow (net inflow)
        -> plots (inflow - outflow)
    - Outflow plot: only nodes where outflow > inflow (net outflow)
        -> plots (outflow - inflow)
    - Uses scenario-level values (no averaging)
    """

    os.makedirs(folder, exist_ok=True)

    net_data = []

    # =========================
    # Collect NET FLOW per scenario
    # =========================
    for n in model.N:
        for m_3 in model.M[3]:

            inflow = sum(
                pyo.value(model.f[a, c, m_3])
                for a in model.A_n_minus[n]
                for c in model.C
            )

            outflow = sum(
                pyo.value(model.f[a, c, m_3])
                for a in model.A_n_plus[n]
                for c in model.C
            )

            net = inflow - outflow

            net_data.append({
                "node": n,
                "net": net
            })

    df = pd.DataFrame(net_data)

    # =========================
    # Classify nodes
    # =========================
    mean_net = df.groupby("node")["net"].mean()

    inflow_nodes = mean_net[mean_net > tol].sort_values(ascending=False).index
    outflow_nodes = mean_net[mean_net < -tol].sort_values().index  # most negative first

    # =========================
    # INFLOW DOMINATED (net > 0)
    # =========================
    if len(inflow_nodes) > 0:
        fig, ax = plt.subplots(figsize=(12, 6))

        data = [
            df[df["node"] == n]["net"].values
            for n in inflow_nodes
        ]

        positions = range(1, len(inflow_nodes) + 1)
        
        parts = ax.violinplot(
            data,
            positions=positions,
            vert=True,
            showmeans=False,
            showmedians=False,
            showextrema=False
        )

        # Styling
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(PASTEL_COLORS[i % len(PASTEL_COLORS)])
            pc.set_edgecolor(PASTEL_COLORS[i % len(PASTEL_COLORS)])
            pc.set_alpha(0.7)

        # =========================
        # Overlay TRUE data points (color-matched)
        # =========================
        for i, vals in enumerate(data):
            vals = np.array(vals)

            sorted_vals = np.sort(vals)
            unique_vals, counts = np.unique(sorted_vals, return_counts=True)

            x_positions = []
            y_positions = []

            for val, count in zip(unique_vals, counts):
                offsets = [0] if count == 1 else np.linspace(-0.08, 0.08, count)
                
                for offset in offsets:
                    x_positions.append(positions[i] + offset)
                    y_positions.append(val)

            # Match violin color, but slightly darker for contrast
            base_color = PASTEL_COLORS[i % len(PASTEL_COLORS)]
            point_color = adjust_color(base_color, factor=0.9)

            ax.scatter(
            x_positions,
            y_positions,
            s=14,
            alpha=1,
            color=point_color,
            edgecolors='none'
            )

        ax.set_xticks(range(1, len(inflow_nodes) + 1))
        ax.set_xticklabels(inflow_nodes, rotation=0, ha='center')

        ax.set_title("Network Inflow per Node")
        ax.set_xlabel("Node")
        ax.set_ylim(0)
        ax.set_ylabel("Inflow (Mscm)")
        ax.grid(alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig(folder / "net_inflow_violin.png")

        if show:
            plt.show()

        plt.close(fig)

    else:
        print("No net inflow nodes.")

    # =========================
    # OUTFLOW DOMINATED (net < 0)
    # =========================
    if len(outflow_nodes) > 0:
        fig, ax = plt.subplots(figsize=(12, 6))

        data = [
            -df[df["node"] == n]["net"].values  # flip sign → positive magnitude
            for n in outflow_nodes
        ]

        positions = range(1, len(outflow_nodes) + 1)
        
        parts = ax.violinplot(
            data,
            positions=positions,
            vert=True,
            showmeans=False,
            showmedians=False,
            showextrema=False
        )

        # Styling
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(PASTEL_COLORS[i % len(PASTEL_COLORS)])
            pc.set_edgecolor(PASTEL_COLORS[i % len(PASTEL_COLORS)])
            pc.set_alpha(0.7)

        # =========================
        # Overlay TRUE data points (color-matched)
        # =========================
        for i, vals in enumerate(data):
            vals = np.array(vals)

            sorted_vals = np.sort(vals)
            unique_vals, counts = np.unique(sorted_vals, return_counts=True)

            x_positions = []
            y_positions = []

            for val, count in zip(unique_vals, counts):
                offsets = [0] if count == 1 else np.linspace(-0.08, 0.08, count)
                for offset in offsets:
                    x_positions.append(positions[i] + offset)
                    y_positions.append(val)

            # Match violin color, but slightly darker for contrast
            base_color = PASTEL_COLORS[i % len(PASTEL_COLORS)]
            point_color = adjust_color(base_color, factor=0.9)

            ax.scatter(
            x_positions,
            y_positions,
            s=14,
            alpha=1,
            color=point_color,
            edgecolors='none'
            )

        ax.set_xticks(range(1, len(outflow_nodes) + 1))
        ax.set_xticklabels(outflow_nodes, rotation=0, ha='center')

        ax.set_title("Network Outflow per Node")
        ax.set_xlabel("Node")
        ax.set_ylim(0)
        ax.set_ylabel("Outflow (Mscm)")
        ax.grid(alpha=0.3, axis="y")


        plt.tight_layout()
        plt.savefig(folder / "net_outflow_violin.png")

        if show:
            plt.show()

        plt.close(fig)

    else:
        print("No net outflow nodes.")


def plot_results(model, folder = "figures/"):
    folder = Path(folder)
    if len(model.N) == 0:
        print("Warning: No plots were made as there are no nodes in the model")
        return None
    os.makedirs(folder, exist_ok=True)
    plot_average_flows(model, folder)
    plot_flow_violins(model, folder)
    plot_component_flows_stacked(model, folder)
    plot_inlet_outlet_pressures(model, folder)
    plot_total_vs_weymouth_histogram(model, folder)
    plot_market_node_component_flow_with_std(model, folder)
    plot_supplier_limit_vs_produced(model, folder / "limit_production/")
    plot_supplier_total_prod_vs_total_demand(model, folder / "contracted_demand/")
    plot_scenario_revenue_costs(model, folder)
    plot_scenario_objectives(model, folder)
    plot_entry_exit_capacity(model, folder)
    plot_overdraft_violins(model, folder)
    plot_inlet_outlet_pressure_violins_example(model, folder)
