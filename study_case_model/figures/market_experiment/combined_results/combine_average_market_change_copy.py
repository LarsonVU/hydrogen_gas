import os
import pickle
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import folium
import geopandas as gpd
import numbers
import pandas as pd
import ast

# =========================
# SETTINGS
# =========================
EXPERIMENT = "run_30326"
LOAD_ONE_RUN = None
MINIMUM_RUNS = 1
SUBSIDY = 70.0

# =========================
# COLOR PALETTE
# =========================
PASTEL_COLORS = [
    "#82C9FF",  # blue
    "#FF8692",  # red
    "#4BDA6A",  # green
    "#DB97E3",  # purple
    "#FFC085",
    "#FFFF82",  # yellow
    "#7EDCD5"
]

# =========================
# HELPERS
# =========================
def crn_standard_error(values):
    values = np.array(values)
    return np.std(values, ddof=1) / np.sqrt(len(values))


# ---------------------------
# LOAD SNAPSHOT
# ---------------------------
def load_snapshot(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)


# ---------------------------
# COMPUTE METRIC (EDIT THIS)
# ---------------------------
def compute_h2(snapshot):
    """
    Example: compute mean H2 production from saved variables.
    Adjust key names if needed.
    """
    try:
        h2_dict = snapshot["expressions"]["h2_production"]
        values = list(h2_dict.values())
        return np.mean(values), np.std(values)
    except KeyError:
        return None
    
def compute_h2_market(snapshot, market, hydrogen_key="H2"):
    try:
        f_dict = snapshot["variables"]["f"]

        # Step 1: aggregate per scenario m
        h2_per_m = {}

        for (i, j, c, m), val in f_dict.items():

            if j == market and c == hydrogen_key:
                if val is not None and not np.isnan(val):
                    h2_per_m[m] = h2_per_m.get(m, 0.0) + val

        if not h2_per_m:
            return None

        values = list(h2_per_m.values())

        return np.mean(values), np.std(values), len(values)

    except KeyError:
        return None

def compute_objective(snapshot):
    """
    Extract objective value from snapshot.
    Assumes single objective.
    """
    try:
        return list(snapshot["objectives"].values())[0]
    except:
        return None

# ---------------------------
# PROCESS SINGLE (market, h2, subsidy) folder
# ---------------------------
def process_market_h2_sub(folder, compute_method = compute_h2, **kwargs):
    """
    Loop over all runs inside a (market, h2, sub) folder.

    Parameters
    ----------
    compute_method : function
        Function applied to each snapshot.
        Must return either:
            - scalar
            - tuple (value, ...)
    kwargs :
        Extra arguments passed to compute_method
    """
    results = []

    for run_dir in sorted(Path(folder).glob("run*/")):

        if LOAD_ONE_RUN and run_dir.name != LOAD_ONE_RUN:
            continue

        files = list(run_dir.glob("*.pkl"))
        if not files:
            continue

        snapshot = load_snapshot(files[0])

        out = compute_method(snapshot, **kwargs)

        if out is None:
            continue

        # allow flexible return types
        val = out[0] if isinstance(out, tuple) else out
        results.append(val)

    if len(results) == 0:
        return None, None, 0, []

    results = np.array(results)

    mean = np.mean(results)
    stderr = np.std(results, ddof=1) / np.sqrt(len(results))

    return mean, stderr, len(results), results

# ---------------------------
# MAIN LOOP
# ---------------------------
def analyze_experiment(base_folder, method = 0):
    summary = {}

    base = Path(base_folder)
    print("H2 Production:")
    for market_dir in sorted(base.glob("market*/")):
        market_value = str(market_dir.name.replace("market", ""))
        for max_h2_dir in sorted(market_dir.glob("maxh2_*/")):
            allowed_h2  = float(max_h2_dir.name.replace("maxh2_", ""))
            for sub_dir in sorted(max_h2_dir.glob("sub*/")):
                sub_value = float(sub_dir.name.replace("sub", ""))
                if method  == 0:
                    mean, stderr, n, results = process_market_h2_sub(sub_dir, compute_method=compute_h2_market, market=market_value, hydrogen_key="H2")
                elif method == 1:
                    mean, stderr, n, results = process_market_h2_sub(sub_dir, compute_method= compute_h2)

                if n > MINIMUM_RUNS-1:
                    summary.setdefault((market_value, allowed_h2), {
                                        "subsidy": [],
                                        "mean": [],
                                        "se": [],
                                        "values": []
                                    })

                    summary[(market_value, allowed_h2)]["subsidy"].append(sub_value)
                    summary[(market_value, allowed_h2)]["mean"].append(mean)
                    summary[(market_value, allowed_h2)]["se"].append(stderr)
                    summary[(market_value, allowed_h2)]["values"].append(results)

                    print(f"Market={market_value}, Allowed H2={allowed_h2}, sub={sub_value} | mean={mean:.4f}, stderr={stderr:.4f}, n={n}")
    return summary

def plot_h2_vs_allowed_for_subsidy(h2_dict, subsidy_value, folder, tol=1e-6):
    os.makedirs(folder, exist_ok=True)
    plt.figure(figsize=(10, 5))

    market_data = {}
    color_map = {}
    color_idx = 0

    # --- collect data ---
    for (market, allowed_h2), stats in h2_dict.items():
        for sub, values in zip(stats["subsidy"], stats["values"]):
            if abs(sub - subsidy_value) < tol:
                market_data.setdefault(market, []).append((allowed_h2, values))

    # --- baseline (CRN vector) ---
    baseline_values = None

    for (market, allowed_h2), stats in h2_dict.items():
        if market == "ZEEBRUGGE" and abs(allowed_h2) < tol:
            for sub, values in zip(stats["subsidy"], stats["values"]):
                if abs(sub - subsidy_value) < tol:
                    baseline_values = np.array(values)
                    break

    if baseline_values is None:
        raise ValueError("Baseline not found")

    # --- plot per market ---
    for market, values in sorted(market_data.items()):
        values_sorted = sorted(values, key=lambda x: x[0])

        allowed_list = []
        means = []
        ses = []

        for allowed_h2, scenario_vals in values_sorted:
            scenario_vals = np.array(scenario_vals)

            diff = scenario_vals - baseline_values

            allowed_list.append(allowed_h2)
            means.append(np.mean(diff))

            # CRN variance
            var = np.var(diff, ddof=1)
            ses.append(np.sqrt(var / len(diff)))

        means = np.array(means)
        ses = np.array(ses)

        if market not in color_map:
            color_map[market] = PASTEL_COLORS[color_idx % len(PASTEL_COLORS)]
            color_idx += 1

        color = color_map[market]

        plt.plot(allowed_list, means, '-', color=color, label=f"Market {market}")

        lower = means - ses
        upper = means + ses
        plt.fill_between(allowed_list, lower, upper, color=color, alpha=0.2)

    plt.xlabel('Allowed Hydrogen Share')
    plt.ylabel('Hydrogen Production (CRN Δ vs Zeebrugge)')
    plt.title(f'Hydrogen Production vs Allowed H2 (Subsidy = {subsidy_value})')
    plt.grid(alpha=0.3, axis="y")
    plt.legend()

    plt.savefig(os.path.join(folder, f"h2_prod_vs_allowed_sub_{subsidy_value}.png"))
    plt.close()

def plot_h2_per_market_subsidy(h2_dict, subsidy_value, folder, tol=1e-6):
    os.makedirs(folder, exist_ok=True)
    plt.figure(figsize=(10, 5))

    market_data = {}
    color_map = {}
    color_idx = 0

    # -------------------------
    # Collect scenario-level data
    # -------------------------
    for (market, allowed_h2), stats in h2_dict.items():
        for sub, values in zip(stats["subsidy"], stats["values"]):
            if abs(sub - subsidy_value) < tol:
                market_data.setdefault(market, []).append((allowed_h2, np.array(values)))

    # -------------------------
    # Baseline (CRN vector)
    # -------------------------
    baseline_values = None

    for (market, allowed_h2), stats in h2_dict.items():
        if market == "ZEEBRUGGE" and abs(allowed_h2) < tol:
            for sub, values in zip(stats["subsidy"], stats["values"]):
                if abs(sub - subsidy_value) < tol:
                    baseline_values = np.array(values)
                    break

    if baseline_values is None:
        raise ValueError("Zeebrugge baseline not found")

    # -------------------------
    # Plot per market
    # -------------------------
    for market, values in sorted(market_data.items()):
        values_sorted = sorted(values, key=lambda x: x[0])

        allowed_list = []
        means = []
        ses = []

        for allowed_h2, scenario_vals in values_sorted:
            scenario_vals = np.array(scenario_vals)

            # CRN difference
            diff = scenario_vals - baseline_values

            allowed_list.append(allowed_h2)
            means.append(np.mean(diff))

            # CRN variance
            var = np.var(diff, ddof=1)
            se = np.sqrt(var / len(diff))
            ses.append(se)

        means = np.array(means)
        ses = np.array(ses)

        # -------------------------
        # Color assignment
        # -------------------------
        if market not in color_map:
            color_map[market] = PASTEL_COLORS[color_idx % len(PASTEL_COLORS)]
            color_idx += 1

        color = color_map[market]

        # -------------------------
        # Plot
        # -------------------------
        plt.plot(allowed_list, means, '-', color=color, label=f"Market {market}")

        lower = means - ses
        upper = means + ses
        plt.fill_between(allowed_list, lower, upper, color=color, alpha=0.2)

    plt.xlabel('Allowed Hydrogen Share')
    plt.ylabel('Hydrogen Consumption (CRN Δ vs Zeebrugge)')
    plt.title(f'Hydrogen Consumption vs Allowed H2 (Subsidy = {subsidy_value})')
    plt.grid(alpha=0.3, axis="y")
    plt.legend()

    plt.savefig(os.path.join(folder, f"h2_cons_vs_allowed_sub_{subsidy_value}.png"))
    plt.close()
    
def analyze_objectives(base_folder):
    summary = {}

    base = Path(base_folder)
    print("Objective value:")
    for market_dir in sorted(base.glob("market*/")):
        market_value = str(market_dir.name.replace("market", ""))
        for max_h2_dir in sorted(market_dir.glob("maxh2_*/")):
            allowed_h2  = float(max_h2_dir.name.replace("maxh2_", ""))
            for sub_dir in sorted(max_h2_dir.glob("sub*/")):
                sub_value = float(sub_dir.name.replace("sub", ""))
                results = []
                for run_dir in sorted(sub_dir.glob("run*/")):
                    files = list(run_dir.glob("*.pkl"))
                    
                    if LOAD_ONE_RUN:
                        if run_dir.name != LOAD_ONE_RUN:
                            continue
                    if not files:
                        continue

                    with open(files[0], "rb") as f:
                        snapshot = pickle.load(f)

                    val = compute_objective(snapshot)
                    if val is not None:
                        results.append(val)

                if len(results) <= MINIMUM_RUNS-1:
                    continue

                results = np.array(results)

                mean = np.mean(results)
                se = np.std(results, ddof=1) / np.sqrt(len(results))

                # store in structured format
                summary.setdefault((market_value, allowed_h2), {
                    "subsidy": [],
                    "mean": [],
                    "se": [],
                    "values": []
                })

                summary[(market_value, allowed_h2)]["subsidy"].append(sub_value)
                summary[(market_value, allowed_h2)]["mean"].append(mean)
                summary[(market_value, allowed_h2)]["se"].append(se)
                summary[(market_value, allowed_h2)]["values"].append(results)

                print(f"Market={market_value}, Allowed H2={allowed_h2}, sub={sub_value} | mean={mean:.4f}, stderr={se:.4f}, n={len(results)}")
    return summary

def plot_objective_vs_allowed_for_subsidy(objective_dict, subsidy_value, folder, tol=1e-6):
    """
    For a fixed subsidy:
    - x-axis: allowed_h2
    - lines: markets
    - CRN applied using scenario-level values
    """

    os.makedirs(folder, exist_ok=True)
    plt.figure(figsize=(10, 5))

    market_data = {}
    color_map = {}
    color_idx = 0

    # -------------------------
    # Collect scenario-level data
    # -------------------------
    for (market, allowed_h2), stats in objective_dict.items():
        for sub, values in zip(stats["subsidy"], stats["values"]):
            if abs(sub - subsidy_value) < tol:
                market_data.setdefault(market, []).append((allowed_h2, np.array(values)))

    # -------------------------
    # Baseline (CRN vector)
    # -------------------------
    baseline_values = None

    for (market, allowed_h2), stats in objective_dict.items():
        if str(market).lower() == "zeebrugge" and abs(allowed_h2) < tol:
            for sub, values in zip(stats["subsidy"], stats["values"]):
                if abs(sub - subsidy_value) < tol:
                    baseline_values = np.array(values)
                    break

    if baseline_values is None:
        raise ValueError("Zeebrugge baseline not found")

    # -------------------------
    # Plot per market
    # -------------------------
    for market, values in sorted(market_data.items()):
        values_sorted = sorted(values, key=lambda x: x[0])

        allowed_list = []
        means = []
        ses = []

        for allowed_h2, scenario_vals in values_sorted:
            scenario_vals = np.array(scenario_vals)

            # CRN difference
            diff = scenario_vals - baseline_values

            allowed_list.append(allowed_h2)
            means.append(np.mean(diff))

            # CRN variance
            var = np.var(diff, ddof=1)
            se = np.sqrt(var / len(diff))
            ses.append(se)

        means = np.array(means)
        ses = np.array(ses)

        # -------------------------
        # Colors
        # -------------------------
        if market not in color_map:
            color_map[market] = PASTEL_COLORS[color_idx % len(PASTEL_COLORS)]
            color_idx += 1

        color = color_map[market]

        # -------------------------
        # Plot
        # -------------------------
        plt.plot(allowed_list, means, '-', color=color, label=f"Market {market}")

        lower = means - ses
        upper = means + ses
        plt.fill_between(allowed_list, lower, upper, color=color, alpha=0.2)

    plt.xlabel('Allowed Hydrogen Share')
    plt.ylabel('Objective Value (CRN Δ vs Zeebrugge)')
    plt.title(f'Objective vs Allowed H2 (Subsidy = {subsidy_value})')
    plt.grid(alpha=0.3, axis="y")
    plt.legend()

    filename = f"objective_vs_allowed_sub_{subsidy_value}.png"
    plt.savefig(os.path.join(folder, filename))
    plt.close()

def plot_net_effect_for_subsidy(
    objective_dict,
    h2_dict,
    subsidy_value,
    folder,
    co2_method="zero",
    tol=1e-6
):
    os.makedirs(folder, exist_ok=True)
    plt.figure(figsize=(10, 5))

    # ---------------------------
    # CO2 valuation
    # ---------------------------
    if co2_method == "zero":
        co2_savings_unit = 0

    elif co2_method == "energy":
        co2_cost_ng = 2.134 * 1000
        co2_cost_green_h2 = 1.080 / 11.94 * 1000

        conversion_ng = 39.8 / 3.6 * 1000
        conversion_h2 = 12.7 / 3.6 * 1000

        co2_savings_unit = 185 * (co2_cost_ng - co2_cost_green_h2) * (conversion_h2 / conversion_ng)

    elif co2_method == "volume":
        co2_cost_ng = 2.134 * 1000
        co2_cost_green_h2 = 1.080 / 11.94 * 1000

        co2_savings_unit = 185 * (co2_cost_ng - co2_cost_green_h2)

    else:
        raise ValueError("No accepted co2 saving method")

    # ---------------------------
    # Collect CRN scenario data
    # ---------------------------
    market_data = {}
    color_map = {}
    color_idx = 0

    for (market, allowed_h2), obj_stats in objective_dict.items():
        h2_stats = h2_dict[(market, allowed_h2)]

        for i, sub in enumerate(obj_stats["subsidy"]):
            if abs(sub - subsidy_value) < tol:

                market_data.setdefault(market, []).append(
                    (
                        allowed_h2,
                        np.array(obj_stats["values"][i]),   # scenario-level objective
                        np.array(h2_stats["values"][i])     # scenario-level H2
                    )
                )

    # ---------------------------
    # Baseline (CRN vector)
    # ---------------------------
    baseline_obj = None
    baseline_h2 = None

    for (market, allowed_h2), obj_stats in objective_dict.items():
        if str(market).lower() == "zeebrugge" and abs(allowed_h2) < tol:

            h2_stats = h2_dict[(market, allowed_h2)]

            for i, sub in enumerate(obj_stats["subsidy"]):
                if abs(sub - subsidy_value) < tol:
                    baseline_obj = np.array(obj_stats["values"][i])
                    baseline_h2 = np.array(h2_stats["values"][i])
                    break

    if baseline_obj is None or baseline_h2 is None:
        raise ValueError("Baseline not found")

    # ---------------------------
    # Plot per market
    # ---------------------------
    for market, values in sorted(market_data.items()):
        values_sorted = sorted(values, key=lambda x: x[0])

        allowed_list = []
        net_means = []
        net_ses = []

        for allowed_h2, obj_vals, h2_vals in values_sorted:

            # ---------------------------
            # CRN differences
            # ---------------------------
            d_obj = obj_vals - baseline_obj
            d_h2 = h2_vals - baseline_h2

            # ---------------------------
            # Net at scenario level
            # ---------------------------
            sub_cost = subsidy_value * 2.78 * 1000 * d_h2
            co2_savings = d_h2 * co2_savings_unit

            net_vals = d_obj - sub_cost + co2_savings

            # ---------------------------
            # Aggregation
            # ---------------------------
            allowed_list.append(allowed_h2)
            net_means.append(np.mean(net_vals))

            var = np.var(net_vals, ddof=1)
            net_ses.append(np.sqrt(var / len(net_vals)))

        net_means = np.array(net_means)
        net_ses = np.array(net_ses)

        # ---------------------------
        # Color assignment
        # ---------------------------
        if market not in color_map:
            color_map[market] = PASTEL_COLORS[color_idx % len(PASTEL_COLORS)]
            color_idx += 1

        color = color_map[market]

        # ---------------------------
        # Plot
        # ---------------------------
        plt.plot(allowed_list, net_means, '-', color=color, label=f"Market {market}")

        lower = net_means - net_ses
        upper = net_means + net_ses
        plt.fill_between(allowed_list, lower, upper, color=color, alpha=0.2)

    # ---------------------------
    # Labels
    # ---------------------------
    plt.xlabel('Allowed Hydrogen Share')
    plt.ylabel('Net Effect (CRN, €)')
    plt.title(f'Net Welfare Effect (Subsidy = {subsidy_value}, CO2 = {co2_method})')
    plt.grid(alpha=0.3, axis="y")
    plt.legend()

    filename = f"net_effect_vs_allowed_sub_{subsidy_value}_{co2_method}.png"
    plt.savefig(os.path.join(folder, filename))
    plt.close()


def analyze_network_flows(market, subsidy, allowed_h2, base_folder):
    """
    Extract all flow variables for a given (market, allowed_h2, subsidy) setting.

    Returns:
        dict with:
            - "flows": list of flow dictionaries per run
            - "mean": aggregated mean flow (per variable)
            - "se": standard error (per variable)
    """
    base = Path(base_folder)

    market_dir = base / f"market{market}"
    allowed_h2_dir = market_dir / f"maxh2_{allowed_h2}"
    sub_dir = allowed_h2_dir / f"sub{subsidy}"
    flows = []

    for run_dir in sorted(sub_dir.glob("run*/")):
        files = list(run_dir.glob("*.pkl"))
        if not files:
            continue

        with open(files[0], "rb") as f:
            snapshot = pickle.load(f)

        # ---- Extract flows from snapshot ----
        # This assumes your snapshot stores flows in model.f as before
        flow_dict = {}

        f_vals = snapshot["variables"]["f"]
        sp = snapshot["parameters"]["sp"]

        for (a_in, a_out, c, s), val in f_vals.items():
            prob = sp[3,s]

            if (a_in, a_out) not in flow_dict:
                flow_dict[a_in, a_out] = {}

            if c not in flow_dict[a_in, a_out]:
                flow_dict[a_in, a_out][c] = 0.0

            flow_dict[a_in, a_out][c] += prob * val
        flows.append(flow_dict)
        # print(f"Extracted flow for run {run_dir.name}: {flow_dict}")

    if len(flows) == 0:
        return None

    # ---- Aggregate (example: mean & SE per variable/index) ----
    # Convert to structured numeric arrays
    aggregated = {}

    # Collect all variable names
    var_names = flows[0].keys()

    for vname in var_names:
        # collect all runs for this variable
        values = []

        for flow in flows:
            if isinstance(flow[vname], dict):
                values.append(list(flow[vname].values()))
            else:
                values.append([flow[vname]])

        values = np.array(values)

        mean = np.mean(values, axis=0)
        se = np.std(values, axis=0, ddof=1) / np.sqrt(values.shape[0])

        aggregated[vname] = {
            "mean": mean,
            "se": se
        }

        #print(f"Aggregated flow for variable {vname}: mean={mean}, se={se}")

    return {
        "flows": flows,
        "aggregated": aggregated
    }


def find_node_by_coords(node_rows, coords, tol=1e-6):
    try:
        coords = ast.literal_eval(coords)
    except (ValueError, SyntaxError):
        return coords

    x, y = coords
    matches = node_rows[
        node_rows.geometry.apply(
            lambda g: abs(g.x - x) < tol and abs(g.y - y) < tol
        )
    ]
    if len(matches) == 0:
        raise ValueError(f"No node found near {coords}")
    if len(matches) > 1:
        raise ValueError(f"Multiple nodes found near {coords}")
    return matches.iloc[0]["location"]

def network_plot_hydrogen_production(market, allowed_h2, subsidy, base_folder, output_folder):
    # -------------------------------
    # Load flows
    # -------------------------------
    result = analyze_network_flows(market, allowed_h2, subsidy, base_folder)
    if result is None:
        print("No flow data found.")
        return None

    # ---- Average flows over runs ----
    avg_flow = result["aggregated"]

    # -------------------------------
    # Load geo data
    # -------------------------------
    FOLDER = "data/data_analysis_results/Geojson_pipelines/"
    input_path = FOLDER + "study_case_network.geojson"
    os.makedirs(output_folder, exist_ok=True)
    output_html = os.path.join(output_folder, f"network_h2_sub{subsidy}_market{market}_allowed_h2{allowed_h2}.html")

    gdf = gpd.read_file(input_path)

    nodes = gdf[gdf.geometry.type == "Point"].copy()
    edges = gdf[gdf.geometry.type.isin(["LineString", "MultiLineString"])].copy()

    # -------------------------------
    # Create map
    # -------------------------------
    center = gdf.geometry.union_all().centroid
    m = folium.Map(location=[center.y, center.x], zoom_start=7, tiles="CartoDB positron")

    # -------------------------------
    # Helpers
    # -------------------------------
    def is_nan_safe(val):
        if isinstance(val, numbers.Number):
            return np.isnan(val)
        if isinstance(val, pd.Timestamp):
            return not pd.isna(val)
        return False

    def make_tooltip(row):
        props = []
        for col in row.index:
            if col in ["pipName", "fromFacili", "toFacility", "max_flow", "location", "node_type"]:
                val = row[col]
                if val is not None and not is_nan_safe(val):
                    props.append(f"<b>{col}</b>: {val}")
        return "<br>".join(props)

    # -------------------------------
    # Determine max H2 flow (for scaling)
    # -------------------------------
    h2_values = []
    tot_flow_values = []
    for arc in avg_flow.keys():
        h2_values.append(avg_flow[arc]["mean"][2])
        tot_flow_values.append(sum(avg_flow[arc]["mean"]))

    max_flow = 4.75 #max(h2_values) if h2_values else 1.0
    max_tot_flow = max(tot_flow_values) if tot_flow_values else 1.0



    # -------------------------------
    # Add edges (WITH hydrogen coloring)
    # -------------------------------
    for _, row in edges.iterrows():
        geom = row.geometry
        lines = geom.geoms if geom.geom_type == "MultiLineString" else [geom]

        # ---- Map arc ----
        a_in = find_node_by_coords(nodes, row.get("from_node"))
        a_out = find_node_by_coords(nodes, row.get("to_node"))

        h2_flow = avg_flow[a_in, a_out]["mean"][2]
        total_flow = sum(avg_flow[a_in, a_out]["mean"])

        # Normalize
        intensity = round(h2_flow / max_flow,5) if max_flow > 0 else 0.0

        # Blue gradient (light → strong blue)
        def interpolate_color(intensity, start=(60, 60, 60), end=(0, 201, 255)):
            intensity = intensity * 5 # Max 20% accross pipelines
            r = int(start[0] + (end[0] - start[0]) * intensity)
            g = int(start[1] + (end[1] - start[1]) * intensity)
            b = int(start[2] + (end[2] - start[2]) * intensity)
            return f"rgb({r}, {g}, {b})"

        def adjust_intensity(intensity):
            return np.power(intensity, 0.8)  # Adjust exponent for better contrast

        color_intensity = adjust_intensity(intensity)
        color = interpolate_color(color_intensity)

        # Thickness scaling
        weight_intensity = total_flow / max_tot_flow if max_tot_flow > 0 else 0
        weight = 2 + 6 * np.power(weight_intensity, 0.5)
        

        for line in lines:
            coords = [(lat, lon) for lon, lat in line.coords]

            tooltip = make_tooltip(row) + f"<br><b>H2 flow</b>: {h2_flow:.2f}" + f"<br><b>Total flow</b>: {total_flow:.2f}"

            folium.PolyLine(
                coords,
                color=color,
                weight=weight,
                opacity=0.85,
                tooltip=folium.Tooltip(tooltip, sticky=True),
            ).add_to(m)

    # -------------------------------
    # Add nodes (unchanged)
    # -------------------------------
    for _, row in nodes.iterrows():
        point = row.geometry
        tooltip = make_tooltip(row)

        color_map = {
            "Generation": "#FFB3BA",
            "Processing": "#BAFFC9",
            "Market": "#BAE1FF",
            "Compression": "#FFFFBA",
            "Junction": "#E0BBE4",
        }

        node_type = row.get("node_type", "junction")
        color = color_map.get(node_type, "#CCCCCC")

        folium.CircleMarker(
            location=[point.y, point.x],
            radius=5,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.9,
            tooltip=folium.Tooltip(tooltip, sticky=True),
        ).add_to(m)

    # -------------------------------
    # Add legend (with hydrogen)
    # -------------------------------
    # Precompute example values
    flow_ticks = [0, 0.25, 0.5, 0.75, 1.0]
    flow_values = [f"{t * max_tot_flow:.1f}" for t in flow_ticks]


    legend_html = f'''
        <div style="position: fixed; 
            bottom: 50px; right: 50px; width: 260px; height: auto; 
            background-color: white; border:2px solid grey; z-index:9999; 
            font-size:14px; padding: 12px; overflow-y: auto;">

        <p style="margin: 0 0 10px 0; font-weight: bold;">Legend</p>

        <p><b>Node Types</b></p>
        <p><span style="background-color: #FFB3BA; width: 15px; height: 15px; display: inline-block; border-radius: 50%;"></span> Generation</p>
        <p><span style="background-color: #BAFFC9; width: 15px; height: 15px; display: inline-block; border-radius: 50%;"></span> Processing</p>
        <p><span style="background-color: #BAE1FF; width: 15px; height: 15px; display: inline-block; border-radius: 50%;"></span> Market</p>
        <p><span style="background-color: #FFFFBA; width: 15px; height: 15px; display: inline-block; border-radius: 50%;"></span> Compression</p>
        <p><span style="background-color: #E0BBE4; width: 15px; height: 15px; display: inline-block; border-radius: 50%;"></span> Junction</p>

        <p><b>Hydrogen (%)</b></p>
        <div style="width: 100%; height: 15px; 
            background: linear-gradient(to right, 
            {interpolate_color(adjust_intensity(0))}, 
            {interpolate_color(adjust_intensity(0.05))}, 
            {interpolate_color(adjust_intensity(0.1))}, 
            {interpolate_color(adjust_intensity(0.15))}, 
            {interpolate_color(adjust_intensity(0.2))}); 
            border: 1px solid black;">
        </div>
        <div style="display: flex; justify-content: space-between;">
            <span>0</span>
            <span>10</span>
            <span>20</span>
        </div>

        <p><b>Flow (Mscm)</b></p>

        <div style="width: 100%; height: 20px; position: relative;">

            <!-- Tapered thickness bar -->
            <div style="
                width: 100%; 
                height: 100%;
                background: linear-gradient(to right,
                    rgba(0,0,0,0.2) 0%,
                    rgba(0,0,0,0.4) 25%,
                    rgba(0,0,0,0.6) 50%,
                    rgba(0,0,0,0.8) 75%,
                    rgba(0,0,0,1.0) 100%);
                clip-path: polygon(
                    0% 50%, 
                    100% 0%, 
                    100% 100%
                );
            "></div>

        </div>

        <div style="display: flex; justify-content: space-between; font-size: 12px;">
            <span>{flow_values[0]}</span>
            <span>{flow_values[2]}</span>
            <span>{flow_values[4]}</span>
        </div>
        </div>
        '''
    m.get_root().html.add_child(folium.Element(legend_html))

    # -------------------------------
    # Save
    # -------------------------------
    m.save(output_html)
    print(f"Saved interactive map to: {output_html}")

    return m


# ---------------------------
# RUN
# ---------------------------
if __name__ == "__main__":
    base_folder = f"study_case_model/figures/market_experiment/{EXPERIMENT}/"
    
    max_h2_values = [0.0, 0.1, 0.2]
    market_values = ["DORNUM", "ZEEBRUGGE", "DUNKERQUE"]

    for max_h2_value in max_h2_values:
        for market in market_values:
            network_plot_hydrogen_production(market, SUBSIDY, max_h2_value, base_folder, f"study_case_model/figures/market_experiment/combined_results/html_networks/{EXPERIMENT}/")

    results = analyze_experiment(base_folder, method=1)
    results_2 = analyze_experiment(base_folder, method=0)

    if LOAD_ONE_RUN:
        plot_h2_vs_allowed_for_subsidy(results, subsidy_value=SUBSIDY, folder=f"study_case_model/figures/market_experiment/combined_results/{LOAD_ONE_RUN}")
        plot_h2_per_market_subsidy(results_2, subsidy_value=SUBSIDY, folder=f"study_case_model/figures/market_experiment/combined_results/{LOAD_ONE_RUN}")
    else:
        plot_h2_vs_allowed_for_subsidy(results, subsidy_value=SUBSIDY, folder=f"study_case_model/figures/market_experiment/combined_results/")
        plot_h2_per_market_subsidy(results_2, subsidy_value=SUBSIDY, folder=f"study_case_model/figures/market_experiment/combined_results/")
       

    objective_dict = analyze_objectives(base_folder)

    if LOAD_ONE_RUN:
        plot_objective_vs_allowed_for_subsidy(objective_dict, subsidy_value=SUBSIDY, folder=f"study_case_model/figures/market_experiment/combined_results/{LOAD_ONE_RUN}")
        plot_net_effect_for_subsidy(objective_dict, results, subsidy_value=SUBSIDY, folder=f"study_case_model/figures/market_experiment/combined_results/{LOAD_ONE_RUN}")
        plot_net_effect_for_subsidy(objective_dict, results, subsidy_value=SUBSIDY,  folder=f"study_case_model/figures/market_experiment/combined_results/{LOAD_ONE_RUN}", co2_method="energy")
        plot_net_effect_for_subsidy(objective_dict, results, subsidy_value=SUBSIDY, folder=f"study_case_model/figures/market_experiment/combined_results/{LOAD_ONE_RUN}", co2_method="volume")
    else:
        plot_objective_vs_allowed_for_subsidy(objective_dict, subsidy_value=SUBSIDY, folder=f"study_case_model/figures/market_experiment/combined_results/")
        plot_net_effect_for_subsidy(objective_dict, results_2, subsidy_value=SUBSIDY, folder=f"study_case_model/figures/market_experiment/combined_results/")
        plot_net_effect_for_subsidy(objective_dict, results_2, subsidy_value=SUBSIDY, folder=f"study_case_model/figures/market_experiment/combined_results/", co2_method="energy")
        plot_net_effect_for_subsidy(objective_dict, results_2, subsidy_value=SUBSIDY, folder=f"study_case_model/figures/market_experiment/combined_results/", co2_method="volume")