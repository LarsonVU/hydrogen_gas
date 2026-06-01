import os
import pickle
import numpy as np
from pathlib import Path
import folium
import geopandas as gpd
import numbers
import pandas as pd
import ast

BASE_FOLDER = Path("study_case_model/figures/force_hydrogen")
GEOJSON_FILE = Path("data/data_analysis_results/Geojson_pipelines/study_case_network.geojson")
OUTPUT_FOLDER = BASE_FOLDER / "network_maps"


def load_snapshot(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def build_flow_map(snapshot):
    f_vals = snapshot["variables"].get("f", {})
    sp = snapshot["parameters"].get("sp", {})

    flow_map = {}
    for key, val in f_vals.items():
        if len(key) != 4:
            continue
        a_in, a_out, c, s = key

        prob = sp.get((3, s), 0)
        if prob == 0:
            continue

        arc = (a_in, a_out)
        flow_map.setdefault(arc, {})
        flow_map[arc][c] = flow_map[arc].get(c, 0.0) + prob * val

    return flow_map


def interpolate_color(intensity, start=(60, 60, 60), end=(0, 201, 255)):
    intensity = min(max(intensity, 0.0), 1.0) * 5.0
    r = int(start[0] + (end[0] - start[0]) * intensity)
    g = int(start[1] + (end[1] - start[1]) * intensity)
    b = int(start[2] + (end[2] - start[2]) * intensity)
    return f"rgb({r}, {g}, {b})"


def adjust_intensity(intensity):
    return np.power(max(min(intensity, 1.0), 0.0), 0.8)


def is_nan_safe(val):
    if isinstance(val, numbers.Number):
        return np.isnan(val)
    if isinstance(val, pd.Timestamp):
        return not pd.isna(val)
    return False


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


def create_network_map(flow_map, output_path, title=None):
    if not GEOJSON_FILE.exists():
        raise FileNotFoundError(f"GeoJSON file not found: {GEOJSON_FILE}")

    gdf = gpd.read_file(GEOJSON_FILE)
    nodes = gdf[gdf.geometry.type == "Point"].copy()
    edges = gdf[gdf.geometry.type.isin(["LineString", "MultiLineString"])].copy()

    center = gdf.geometry.unary_union.centroid
    m = folium.Map(location=[center.y, center.x], zoom_start=7, tiles="CartoDB positron")

    h2_values = []
    tot_flow_values = []
    for arc, comp in flow_map.items():
        total = sum(comp.values())
        h2_values.append(comp.get("H2", comp.get(2, comp.get("2", 0.0))))
        tot_flow_values.append(total)

    max_tot_flow = max(tot_flow_values) if tot_flow_values else 1.0

    for _, row in edges.iterrows():
        geom = row.geometry
        lines = geom.geoms if geom.geom_type == "MultiLineString" else [geom]
        a_in = find_node_by_coords(nodes, row.get("from_node"))
        a_out = find_node_by_coords(nodes, row.get("to_node"))

        arc = (a_in, a_out)
        if arc not in flow_map:
            continue

        comp = flow_map[arc]
        h2_flow = comp.get("H2", comp.get(2, comp.get("2", 0.0)))
        total_flow = sum(comp.values())
        intensity = h2_flow / total_flow if total_flow > 0 else 0.0

        color_intensity = adjust_intensity(intensity)
        color = interpolate_color(color_intensity)
        weight_intensity = total_flow / max_tot_flow if max_tot_flow > 0 else 0.0
        weight = 2 + 6 * np.power(weight_intensity, 0.3)

        for line in lines:
            coords = [(lat, lon) for lon, lat in line.coords]
            tooltip = (
                f"<b>H2 flow</b>: {h2_flow:.2f}<br>"
                f"<b>Total flow</b>: {total_flow:.2f}<br>"
                f"<b>Arc</b>: {a_in} → {a_out}"
            )
            folium.PolyLine(
                coords,
                color=color,
                weight=weight,
                opacity=0.85,
                tooltip=folium.Tooltip(tooltip, sticky=True),
            ).add_to(m)

    legend_html = f'''
        <div style="position: fixed; 
            bottom: 50px; right: 50px; width: 280px; height: auto; 
            background-color: white; border:2px solid grey; z-index:9999; 
            font-size:14px; padding: 12px; overflow-y: auto;">
        <p style="margin: 0 0 10px 0; font-weight: bold;">Network legend</p>
        <p><b>Hydrogen fraction</b></p>
        <div style="width: 100%; height: 15px; background: linear-gradient(to right, 
            {interpolate_color(adjust_intensity(0.0))}, 
            {interpolate_color(adjust_intensity(0.25))}, 
            {interpolate_color(adjust_intensity(0.5))}, 
            {interpolate_color(adjust_intensity(0.75))}, 
            {interpolate_color(adjust_intensity(1.0))}); 
            border: 1px solid black;"></div>
        <div style="display: flex; justify-content: space-between;">
            <span>0%</span><span>25%</span><span>50%</span><span>75%</span><span>100%</span>
        </div>
        <p style="margin-top: 10px;"><b>Thickness</b>: total flow</p>
        </div>
        '''

    m.get_root().html.add_child(folium.Element(legend_html))

    if title:
        title_html = f"<h4 style='position: fixed; top: 10px; left: 10px; z-index:9999; background:white; padding:5px 10px; border:1px solid grey;'>{title}</h4>"
        m.get_root().html.add_child(folium.Element(title_html))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output_path))
    print(f"Saved interactive map to: {output_path}")


def find_snapshot_dirs():
    if not BASE_FOLDER.exists():
        raise FileNotFoundError(f"Force hydrogen base folder not found: {BASE_FOLDER}")
    for level_dir in sorted(BASE_FOLDER.glob('h2force_*')):
        for sub_dir in sorted(level_dir.glob('sub*')):
            for run_dir in sorted(sub_dir.glob('run*')):
                snapshot = run_dir / 'model_snapshot.pkl'
                if snapshot.exists():
                    yield level_dir.name, sub_dir.name, run_dir.name, snapshot


def main():
    for level_name, sub_name, run_name, snapshot_path in find_snapshot_dirs():
        snapshot = load_snapshot(snapshot_path)
        flow_map = build_flow_map(snapshot)
        html_name = f"network_{level_name}_{sub_name}_{run_name}.html"
        output_path = OUTPUT_FOLDER / html_name
        title = f"Force hydrogen network: {level_name} / {sub_name} / {run_name}"
        create_network_map(flow_map, output_path, title=title)


if __name__ == '__main__':
    main()
