from pathlib import Path
import folium
import numpy as np
import pickle
import matplotlib.pyplot as plt
import ast
import os
import geopandas as gpd
import pandas as pd
import numbers
import matplotlib.colors as mcolors


EXPERIMENT = "run_14426"
LOAD_ONE_RUN = None  # "run0"
MINIMUM_RUNS = 2
HYDROGEN_MSCM_MWH = 2.78 * 1000 

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

# ---------------------------
# LOAD SNAPSHOT
# ---------------------------
def load_snapshot(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)

# ---------------------------
# COMPUTE METRICS
# ---------------------------
def compute_h2(snapshot):
    try:
        h2_dict = snapshot["expressions"]["h2_production"]
        values = list(h2_dict.values())
        return np.mean(values), np.std(values)
    except KeyError:
        return None, None

def compute_objective(snapshot):
    try:
        return list(snapshot["objectives"].values())[0]
    except:
        return None

# ---------------------------
# FAILURE CASE PROCESSING
# ---------------------------
def process_failed_sub(folder):
    results = []

    for run_dir in sorted(Path(folder).glob("run*/")):
        files = list(run_dir.glob("*.pkl"))

        if LOAD_ONE_RUN and run_dir.name != LOAD_ONE_RUN:
            continue

        if not files:
            continue

        snapshot = load_snapshot(files[0])
        val, std = compute_h2(snapshot)

        if val is not None:
            results.append(val)

    if len(results) == 0:
        return None, None, 0, []

    results = np.array(results)
    mean = np.mean(results)
    stderr = np.std(results, ddof=1) / np.sqrt(len(results))

    return mean, stderr, len(results), results

def analyze_failed_experiment(base_folder):
    summary = {}
    base = Path(base_folder)

    print("H2 Production by FAILED component:")

    for failed_dir in sorted(base.glob("maxh2_*/")):
        FAILED = failed_dir.name.replace("maxh2_", "")

        for sub_dir in sorted(failed_dir.glob("sub*/")):
            SUBSIDY = float(sub_dir.name.replace("sub", ""))

            mean, stderr, n, results = process_failed_sub(sub_dir)

            if n >= MINIMUM_RUNS:
                summary.setdefault(FAILED, {
                    "subs": [],
                    "mean": [],
                    "se": [],
                    "runs": []
                })

                summary[FAILED]["subs"].append(SUBSIDY)
                summary[FAILED]["mean"].append(mean)
                summary[FAILED]["se"].append(stderr)
                summary[FAILED]["runs"].append(results)

                print(f"FAILED={FAILED}, sub={SUBSIDY} | mean={mean:.4f}, se={stderr:.4f}, n={n}")

    return summary

# ---------------------------
# BASELINE (NO FAILURE)
# ---------------------------
def process_subsidy_folder(folder):
    results = []

    for run_dir in sorted(Path(folder).glob("run*/")):
        files = list(run_dir.glob("*.pkl"))

        if LOAD_ONE_RUN and run_dir.name != LOAD_ONE_RUN:
            continue

        if not files:
            continue

        snapshot = load_snapshot(files[0])
        val, std = compute_h2(snapshot)

        if val is not None:
            results.append(val)

    if len(results) == 0:
        return None, None, 0, []

    results = np.array(results)
    mean = np.mean(results)
    stderr = np.std(results, ddof=1) / np.sqrt(len(results))

    return mean, stderr, len(results), results

def analyze_subsidy_experiment(base_folder,  sub_values = [30.0, 45.0, 70.0]):
    summary = {}
    base = Path(base_folder)

    print("\nBaseline H2 Production (no failures):")

    for sub in sub_values:
        sub_dir = base / f"sub{sub}"
        print(f"Looking for folder: {sub_dir}")
        if not sub_dir.exists():
            print(f"Missing folder for subsidy {sub}")
            continue

        print(f"Processing subsidy {sub}...")
        mean, stderr, n, results = process_subsidy_folder(sub_dir)

        if n >= MINIMUM_RUNS:
            summary[sub] = {
                "mean": mean,
                "se": stderr,
                "n": n,
                "runs": results
            }

            print(f"sub={sub} | mean={mean:.4f}, se={stderr:.4f}, n={n}")

    return summary

def adjust_color(color, factor=0.7):
    """
    Darken (<1) or lighten (>1) a color.
    """
    rgb = np.array(mcolors.to_rgb(color))
    if factor < 1:
        return tuple(rgb * factor)  # darken
    else:
        return tuple(1 - (1 - rgb) / factor)  # lighten


# ---------------------------
# PLOTTING
# ---------------------------
def plot_remaining_h2(summary, baseline_summary=None, output_path=None, failed_elements=None, failed_labels_custom=None, type_failure = "Generation Node"):
    """
    Plot remaining H2 production for failed scenarios.
    
    Args:
        summary: dict with failure analysis results
        baseline_summary: dict with baseline results
        output_path: path to save figure
        failed_elements: list of failed element keys to include (None = all)
        failed_labels_custom: dict mapping failed element keys to custom display names
    """
    if failed_elements is None:
        failed_elements = list(summary.keys())
    
    if failed_labels_custom is None:
        failed_labels_custom = {elem: elem for elem in failed_elements}
    
    # Filter and prepare data
    filtered_data = {elem: summary[elem] for elem in failed_elements if elem in summary}
    
    all_means = []
    all_errors = []
    display_labels = []
    subsidy_values = []

    # Collect subsidies
    for data in filtered_data.values():
        subsidy_values.extend(data["subs"])
    subsidy_values = sorted(set(subsidy_values))

    # Collect data in filtered order
    for elem in failed_elements:
        if elem in filtered_data:
            data = filtered_data[elem]
            display_labels.append(failed_labels_custom[elem])
            all_means.append(data["mean"])
            all_errors.append(data["se"])
    
    x = np.arange(len(display_labels))
    width = 0.4

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    y_lim = [1, 10, 10]
    for plot_idx, subsidy in enumerate(subsidy_values[:3]):
        ax = axes[plot_idx]

        means_for_subsidy = [
            means[plot_idx] * HYDROGEN_MSCM_MWH / 1000 if plot_idx < len(means) else 0
            for means in all_means
        ]

        ses_for_subsidy = [
            ses[plot_idx] * HYDROGEN_MSCM_MWH / 1000 if plot_idx < len(ses) else 0
            for ses in all_errors
        ]

        # ---- MAIN BARS ----
        ax.bar(
            x,
            means_for_subsidy,
            width,
            color=PASTEL_COLORS[plot_idx]
        )

        # ---- ERROR BAND ----
        bottom = [m - se for m, se in zip(means_for_subsidy, ses_for_subsidy)]
        top    = [m + se for m, se in zip(means_for_subsidy, ses_for_subsidy)]

        ax.bar(
            x,
            [2 * se for se in ses_for_subsidy],
            width,
            bottom=bottom,
            color=adjust_color(PASTEL_COLORS[plot_idx], factor=1.15),
            alpha=0.2,
            edgecolor='none'
        )

        # ---- DASHED TOP & BOTTOM ----
        for xi, low, high in zip(x, bottom, top):
            ax.hlines(
                high,
                xi - width/2,
                xi + width/2,
                colors=adjust_color(PASTEL_COLORS[plot_idx], factor=0.6),
                linestyles='dashed',
                linewidth=1
            )
            ax.hlines(
                low,
                xi - width/2,
                xi + width/2,
                colors=adjust_color(PASTEL_COLORS[plot_idx], factor=0.6),
                linestyles='dashed',
                linewidth=1
            )
        
        # ---- BASELINE LINE ----
        if baseline_summary and subsidy in baseline_summary:
            baseline = baseline_summary[subsidy]["mean"] * HYDROGEN_MSCM_MWH /1000
            ax.axhline(
                baseline,
                linestyle="--",
                linewidth=2,
                color="black"
            )

        ax.set_xticks(x)
        ax.set_ylim(0, y_lim[plot_idx])
        ax.set_xticklabels(display_labels, rotation=0, ha="center")
        ax.set_ylabel("Hydrogen Production (Gwh)")
        ax.set_xlabel(type_failure)
        ax.set_title(f"Impact of Failures (Subsidy {int(subsidy)} Eur/MWh)")

    # Create single legend on the right
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, fc=PASTEL_COLORS[i], label=f"Subsidy {int(subsidy)} Eur/MWh")
        for i, subsidy in enumerate(subsidy_values[:3])
    ]
    legend_handles.append(
        plt.Line2D([0], [0], linestyle="--", linewidth=2, color="black", label="Baseline (no closure)")
    )
    
    fig.legend(handles=legend_handles, bbox_to_anchor=(1.12, 1.0))

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()

    plt.close()

def map_name(name):
    mapping = {
        "GJØA": "GJOA",
        "VISUND": "VISUND",
        "NORNE ERB": "NORNE_ERB",
        "KÅRSTØ": "KARSTO",
        "DRAUPNER S": "DRAUPNER_S",
        "DORNUM": "DORNUM",
        "DUNKERQUE": "DUNKERQUE",
        "H-7 BP": "H-7_BP",
        "EMDEN": "EMDEN"
    }
    return mapping.get(name, name)


def analyze_network_flows(subsidy, failed_element, base_folder):
    """
    Extract all flow variables for a given (subsidy, failed_element) setting.

    Returns:
        dict with:
            - "flows": list of flow dictionaries per run
            - "aggregated": mean and SE per variable
    """
    base = Path(base_folder)

    # 🔥 Apply mapping here as well (important!)
    failed_element_mapped = failed_element #map_name(failed_element)

    max_h2_dir = base / f"maxh2_{failed_element_mapped}"
    sub_dir = max_h2_dir / f"sub{subsidy}"

    flows = []

    for run_dir in sorted(sub_dir.glob("run*/")):
        files = list(run_dir.glob("*.pkl"))
        if not files:
            continue

        with open(files[0], "rb") as f:
            snapshot = pickle.load(f)

        flow_dict = {}

        f_vals = snapshot["variables"]["f"]
        sp = snapshot["parameters"]["sp"]

        for (a_in, a_out, c, s), val in f_vals.items():
            prob = sp[3, s]

            # ✅ APPLY MAPPING HERE
            a_in_mapped = map_name(a_in)
            a_out_mapped = map_name(a_out)

            key = (a_in_mapped, a_out_mapped)

            if key not in flow_dict:
                flow_dict[key] = {}

            if c not in flow_dict[key]:
                flow_dict[key][c] = 0.0

            flow_dict[key][c] += prob * val

        flows.append(flow_dict)

    if len(flows) == 0:
        return None

    # ---------------------------
    # AGGREGATION
    # ---------------------------
    aggregated = {}

    # Collect all variable names across runs (safer than flows[0])
    var_names = set()
    for flow in flows:
        var_names.update(flow.keys())

    for vname in var_names:
        values = []

        for flow in flows:
            if vname not in flow:
                continue

            if isinstance(flow[vname], dict):
                values.append(list(flow[vname].values()))
            else:
                values.append([flow[vname]])

        values = np.array(values)

        if len(values) == 0:
            continue

        mean = np.mean(values, axis=0)
        se = np.std(values, axis=0, ddof=1) / np.sqrt(values.shape[0])

        aggregated[vname] = {
            "mean": mean,
            "se": se
        }

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

def network_plot_hydrogen_production(subsidy, failed_element, base_folder, output_folder):
    # -------------------------------
    # Load flows
    # -------------------------------
    result = analyze_network_flows(subsidy, failed_element, base_folder)
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
    output_html = os.path.join(output_folder, f"network_h2_sub{subsidy}_failed{failed_element}.html")

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

    #max_flow = 5 #max(h2_values) if h2_values else 1.0
    max_tot_flow = 30 #max(tot_flow_values) if tot_flow_values else 1.0



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
        intensity = h2_flow / total_flow if total_flow > 0 else 0 # max_flow

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

        # Thickness scaling
        weight_intensity = total_flow / max_tot_flow if max_tot_flow > 0 else 0
        weight = 2 + 6 * np.power(weight_intensity, 0.3)
        
        if total_flow == 0:
            color = "#CCCCCC"
            weight = 4
        else:
            color = interpolate_color(color_intensity)

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
    
# Mapping, reverse to original names for failed elements
def map_name(name):
    mapping = {
        "GJOA": "GJØA",
        "VISUND": "VISUND",
        "NORNE_ERB": "NORNE ERB",
        "KARSTO": "KÅRSTØ",
        "DRAUPNER_S": "DRAUPNER S",
        "DORNUM": "DORNUM",
        "DUNKERQUE": "DUNKERQUE",
        "H-7_BP": "H-7 BP",
        "EMDEN": "EMDEN"
    }
    return mapping.get(name, name)


# ---------------------------
# MAIN
# ---------------------------
if __name__ == "__main__":
    base_folder_failure = f"study_case_model/figures/failure_experiment/{EXPERIMENT}/"
    base_folder_no_failure = f"study_case_model/figures/failure_experiment/{EXPERIMENT}/maxh2_None/"

    sub_values = [30.0, 45.0, 70.0]
    failed_elements = [
        "KARSTO_to_DRAUPNER_S",
        "KARSTO_to_DORNUM",
        "VISUND",
        "NORNE_ERB",
    ]
    
    # Example: Plot network for one failure case
    for sub in sub_values:
        for failed_element in failed_elements:
            print(f"\nGenerating network plot for subsidy {sub} and failed element {failed_element}...")
            network_plot_hydrogen_production(subsidy=sub, failed_element=failed_element,
                                            base_folder=base_folder_failure,
                                            output_folder="study_case_model/figures/failure_experiment/combine_results/")

    # Failure results
    summary_failure = analyze_failed_experiment(base_folder_failure)

    # Baseline results
    baseline_summary = analyze_subsidy_experiment(base_folder_no_failure,sub_values=sub_values)

    print("\nBaseline summary:")
    for sub, data in baseline_summary.items():
        print(sub, data)

    # Plot
    plot_remaining_h2(
        summary_failure,
        baseline_summary=baseline_summary,
        output_path="study_case_model/figures/failure_experiment/combine_results/remaining_h2_generation.png",
        failed_elements=[
            "KARSTO_to_DRAUPNER_S",
            "KARSTO_to_DORNUM",
            # "DRAUPNER_S_to_DUNKERQUE",
            # "H-7_BP_to_EMDEN",
            "VISUND",
            "NORNE_ERB"
            "GJOA",
        ],
        failed_labels_custom={
            "KARSTO_to_DRAUPNER_S": "KÅRSTØ → \n DRAUPNER S",
            "KARSTO_to_DORNUM": "KÅRSTØ → \n DORNUM",
            # "DRAUPNER_S_to_DUNKERQUE": "Pipeline: DRAUPNER S → DUNKERQUE",
            # "H-7_BP_to_EMDEN": "Pipeline: H-7 BP → EMDEN",
            "VISUND": "VISUND",
            "NORNE_ERB": "NORNE ERB",
            "GJOA": "GJØA",
        },
        type_failure="Pipeline / Generation Node"

    )

    # plot_remaining_h2(
    #     summary_failure,
    #     baseline_summary=baseline_summary,
    #     output_path="study_case_model/figures/failure_experiment/combine_results/remaining_h2_pipelines.png",
    #     failed_elements=[
    #         "KARSTO_to_DRAUPNER_S",
    #         "KARSTO_to_DORNUM",
    #         #"DRAUPNER_S_to_DUNKERQUE",
    #         #"H-7_BP_to_EMDEN",

    #     ],
    #     failed_labels_custom={
    #         "KARSTO_to_DRAUPNER_S": " KÅRSTØ → \n DRAUPNER S",
    #         "KARSTO_to_DORNUM": "KÅRSTØ → \n DORNUM",
    #         #"DRAUPNER_S_to_DUNKERQUE": " DRAUPNER S → \n DUNKERQUE",
    #         #"H-7_BP_to_EMDEN": " H-7 BP → \n EMDEN",
    #     },
    #     type_failure="Pipeline"

    # )
