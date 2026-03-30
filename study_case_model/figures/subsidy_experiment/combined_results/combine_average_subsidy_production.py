import os
import pickle
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import os
import folium
import geopandas as gpd
import numbers
import pandas as pd
import ast

LOAD_ONE_RUN = None #"run0"

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
# PROCESS SINGLE (dev, sub)
# ---------------------------
def process_dev_sub(folder):
    """
    Loop over all runs inside a (dev, sub) folder.
    """
    results = []

    for run_dir in sorted(Path(folder).glob("run*/")):
        files = list(run_dir.glob("*.pkl"))
        if LOAD_ONE_RUN:
            if run_dir.name != LOAD_ONE_RUN:
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
    stderr = np.std(results, ddof =1) /len(results)

    return mean, stderr, len(results), results


# ---------------------------
# MAIN LOOP
# ---------------------------
def analyze_experiment(base_folder):
    summary = {}

    base = Path(base_folder)
    print("H2 Production:")
    for dev_dir in sorted(base.glob("dev*/")):
        dev_value = float(dev_dir.name.replace("dev", ""))

        for sub_dir in sorted(dev_dir.glob("sub*/")):
            sub_value = float(sub_dir.name.replace("sub", ""))
            
            mean, stderr, n, results = process_dev_sub(sub_dir)

            if n > 0:
                summary.setdefault(dev_value, {
                                    "subsidy": [],
                                    "mean": [],
                                    "se": []
                                })

                summary[dev_value]["subsidy"].append(sub_value)
                summary[dev_value]["mean"].append(mean)
                summary[dev_value]["se"].append(stderr)

                print(f"dev={dev_value}, sub={sub_value} | mean={mean:.4f}, stderr={stderr:.4f}, n={n}")
    return summary


def plot_hydrogen_production(h2_dict, folder):
    os.makedirs(folder, exist_ok=True)

    plt.figure(figsize=(10, 5))

    for label, stats in sorted(h2_dict.items()):
        # sort by subsidy to ensure clean lines
        sorted_data = sorted(
            zip(stats["subsidy"], stats["mean"], stats["se"]),
            key=lambda x: x[0]
        )

        subs, means, ses = zip(*sorted_data)

        plt.errorbar(
            subs,
            means,
            yerr=ses,
            fmt='o-',
            capsize=5,
            label=f"Deviation {label}"
        )

    plt.xlabel('Subsidy (Euro/MWh)')
    plt.ylabel('Hydrogen Production')
    plt.title('Hydrogen Production vs Subsidy')
    plt.grid(True)
    plt.legend()

    plt.savefig(os.path.join(folder, "hydrogen_production_vs_subsidy.png"))
    plt.close()

def analyze_objectives(base_folder):
    summary = {}

    base = Path(base_folder)
    print("Objective value:")
    for dev_dir in sorted(base.glob("dev*/")):
        dev_value = float(dev_dir.name.replace("dev", ""))

        for sub_dir in sorted(dev_dir.glob("sub*/")):
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

            if len(results) == 0:
                continue

            results = np.array(results)

            mean = np.mean(results)
            se = np.std(results, ddof=1) / np.sqrt(len(results))

            # store in structured format
            summary.setdefault(dev_value, {
                "subsidy": [],
                "mean": [],
                "se": []
            })

            summary[dev_value]["subsidy"].append(sub_value)
            summary[dev_value]["mean"].append(mean)
            summary[dev_value]["se"].append(se)
            print(f"dev={dev_value}, sub={sub_value} | mean={mean:.4f}, stderr={se:.4f}, n={len(results)}")
    return summary

def plot_objective_values(objective_dict, folder):
    os.makedirs(folder, exist_ok=True)

    plt.figure(figsize=(10, 5))

    for label, stats in sorted(objective_dict.items()):
        # ensure correct ordering by subsidy
        sorted_data = sorted(
            zip(stats["subsidy"], stats["mean"], stats["se"]),
            key=lambda x: x[0]
        )

        subs, means, ses = zip(*sorted_data)

        ses  = [np.sqrt(s**2 + ses[0]**2) for s in ses]
        means = means - means[0]

        plt.errorbar(
            subs,
            means,
            yerr=ses,
            fmt='o-',
            capsize=5,
            label=f"Deviation {label}"
        )

    plt.xlabel('Subsidy (Euro/MWh)')
    plt.ylabel('Objective Value')
    plt.title('Objective Value vs Subsidy')
    plt.grid(True)
    plt.legend()

    plt.savefig(os.path.join(folder, "objective_vs_subsidy.png"))
    plt.close()

def plot_net_effect(objective_dict, h2_dict, folder, co2_method ="zero"):
    os.makedirs(folder, exist_ok=True)

    plt.figure(figsize=(10, 5))

    if co2_method == "zero":
        # No social cost on co2
        co2_savings_unit = 0
    elif co2_method == "energy":
        # Social cost savings per displaced mwh (transformed to mscm)
        # Sources:
        # Costs per kwh :   https://co2emissiefactoren.nl/ 
        # Conversion rates : https://www.engineeringtoolbox.com/fuels-higher-calorific-values-d_169.html
        # https://www-nature-com.vu-nl.idm.oclc.org/articles/s41586-022-05224-9  # social cost co2

        co2_cost_ng = 2.134 * 1000 # in tonne (metric ton) CO2e per Mscm 
        co2_cost_green_h2 = 1.080 /11.94 *1000 # in tonne (metric ton) CO2e per Mscm 
        
        conversion_ng = 39.8 /3.6 *1000 # mwh to Mscm
        conversion_h2 = 12.7 /3.6 *1000 #  mwh to Mscm

        co2_savings_unit = 185 * (co2_cost_ng-co2_cost_green_h2) * (conversion_h2/ conversion_ng) # Euro per Mscm
    elif co2_method == "volume":
        # Social cost savings per mscm
        # Sources:
        # https://h2tools.org/hyarc/calculator-tools/hydrogen-conversions-calculator # KG to scm transformation h2
        # https://co2emissiefactoren.nl/  Emissionfactors
        # https://www-nature-com.vu-nl.idm.oclc.org/articles/s41586-022-05224-9  # social cost co2 
        co2_cost_ng = 2.134 * 1000 # in tonne (metric ton) CO2e per Mscm 
        co2_cost_green_h2 = 1.080 /11.94 *1000 # in tonne (metric ton) CO2e per Mscm 
        co2_savings_unit = 185 * (co2_cost_ng -co2_cost_green_h2) # in euro per Mscm
    else:
        Exception("No accepted co2 saving method")
    print("CO2 cost per Mcsm:", co2_savings_unit)


    for label in sorted(objective_dict.keys()):
        obj_stats = objective_dict[label]
        h2_stats = h2_dict[label]

        # sort both consistently
        sorted_data = sorted(
            zip(obj_stats["subsidy"], obj_stats["mean"], obj_stats["se"],
                h2_stats["mean"], h2_stats["se"]),
            key=lambda x: x[0]
        )

        subs, obj_means, obj_ses, h2_means, h2_ses = zip(*sorted_data)
        base_obj = obj_means[0]

        net_means = []
        net_ses = []



        for s, m_obj, se_obj, m_h2, se_h2 in zip(subs, obj_means, obj_ses, h2_means, h2_ses):
            delta_obj =  m_obj - base_obj

            # correct net effect
            net = delta_obj - s *2.78 * 1000 * m_h2 + m_h2 *co2_savings_unit
            net_means.append(net)

            # error propagation (approx)
            var = se_obj**2 + obj_ses[0]**2 + (s**2) * (se_h2**2)
            net_ses.append(np.sqrt(var))

        plt.errorbar(
            subs,
            net_means,
            yerr=net_ses,
            fmt='o-',
            capsize=5,
            label=f"Deviation {label}"
        )

    plt.xlabel('Subsidy (Euro/MWh)')
    plt.ylabel('Net Effect (Euro)')
    plt.title('Net Welfare Effect of Subsidy')
    plt.grid(True)
    plt.legend()

    plt.savefig(os.path.join(folder, f"net_effect_vs_subsidy_{co2_method}.png"))
    plt.close()

def analyze_network_flows(subsidy, deviation, base_folder):
    """
    Extract all flow variables for a given (subsidy, deviation) setting.

    Returns:
        dict with:
            - "flows": list of flow dictionaries per run
            - "mean": aggregated mean flow (per variable)
            - "se": standard error (per variable)
    """
    base = Path(base_folder)

    dev_dir = base / f"dev{deviation}"
    sub_dir = dev_dir / f"sub{subsidy}"
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

def network_plot_hydrogen_production(subsidy, deviation, base_folder, output_folder):
    # -------------------------------
    # Load flows
    # -------------------------------
    result = analyze_network_flows(subsidy, deviation, base_folder)
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
    output_html = os.path.join(output_folder, f"network_h2_sub{subsidy}_dev{deviation}.html")

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

    max_flow = max(h2_values) if h2_values else 1.0
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
        intensity = h2_flow / max_flow if max_flow > 0 else 0

        # Blue gradient (light → strong blue)
        def interpolate_color(intensity, start=(60, 60, 60), end=(0, 201, 255)):
            r = int(start[0] + (end[0] - start[0]) * intensity)
            g = int(start[1] + (end[1] - start[1]) * intensity)
            b = int(start[2] + (end[2] - start[2]) * intensity)
            return f"rgb({r}, {g}, {b})"

        color_intensity = np.power(intensity, 0.2) 
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
    legend_html = f'''
    <div style="position: fixed; 
         bottom: 50px; right: 50px; width: 220px; height: 300px; 
         background-color: white; border:2px solid grey; z-index:9999; 
         font-size:14px; padding: 10px">
         <p style="margin: 0 0 10px 0; font-weight: bold;">Legend</p>

         <p><b>Node Types</b></p>
         <p><span style="background-color: #FFB3BA; width: 15px; height: 15px; display: inline-block; border-radius: 50%;"></span> Generation</p>
         <p><span style="background-color: #BAFFC9; width: 15px; height: 15px; display: inline-block; border-radius: 50%;"></span> Processing</p>
         <p><span style="background-color: #BAE1FF; width: 15px; height: 15px; display: inline-block; border-radius: 50%;"></span> Market</p>
         <p><span style="background-color: #FFFFBA; width: 15px; height: 15px; display: inline-block; border-radius: 50%;"></span> Compression</p>
         <p><span style="background-color: #E0BBE4; width: 15px; height: 15px; display: inline-block; border-radius: 50%;"></span> Junction</p>
         <p><b>Hydrogen Flow</b></p>
         <p><span style="background-color: {interpolate_color(1)}; width: 15px; height: 15px; display: inline-block; border-radius: 50%;"></span> Flow magnitude</p>
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
    base_folder = "study_case_model/figures/subsidy_experiment/run_27326/"
    
    network_plot_hydrogen_production(70.0, 1.0, base_folder,f"study_case_model/figures/subsidy_experiment/combined_results/html_networks/")
    
    results = analyze_experiment(base_folder)

    if LOAD_ONE_RUN:
        plot_hydrogen_production(results, folder=f"study_case_model/figures/subsidy_experiment/combined_results/{LOAD_ONE_RUN}")
    else:
        plot_hydrogen_production(results, folder=f"study_case_model/figures/subsidy_experiment/combined_results/")

    objective_dict = analyze_objectives(base_folder)

    if LOAD_ONE_RUN:
        plot_objective_values(objective_dict, folder=f"study_case_model/figures/subsidy_experiment/combined_results/{LOAD_ONE_RUN}")
        plot_net_effect(objective_dict, results, folder=f"study_case_model/figures/subsidy_experiment/combined_results/{LOAD_ONE_RUN}")
        plot_net_effect(objective_dict, results, folder=f"study_case_model/figures/subsidy_experiment/combined_results/{LOAD_ONE_RUN}", co2_method="energy")
        plot_net_effect(objective_dict, results, folder=f"study_case_model/figures/subsidy_experiment/combined_results/{LOAD_ONE_RUN}", co2_method="volume")
    else:
        plot_objective_values(objective_dict, folder=f"study_case_model/figures/subsidy_experiment/combined_results/")
        plot_net_effect(objective_dict, results, folder=f"study_case_model/figures/subsidy_experiment/combined_results/")
        plot_net_effect(objective_dict, results, folder=f"study_case_model/figures/subsidy_experiment/combined_results/", co2_method="energy")
        plot_net_effect(objective_dict, results, folder=f"study_case_model/figures/subsidy_experiment/combined_results/", co2_method="volume")