import math
import os
import pickle
import numpy as np
import networkx as nx
import pandas as pd
import folium
from shapely.geometry import LineString, MultiLineString, Point
import numbers
from pathlib import Path
import sys

# Add parent directory
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from study_case_problem_file import build_base_graph

# -----------------------------
# CONFIG
# -----------------------------
FOLDER_PICKLE = "study_case_model/unit_networks_solves/unit_gen2_networks/"
FOLDER_HTML = "data/data_analysis_results/unit_gen2_maps/"

np.random.seed(123)

# -----------------------------
# GEN 2 POKEMON
# -----------------------------
POKEMON_GEN2 = [
    "Chikorita","Bayleef","Meganium","Cyndaquil","Quilava","Typhlosion",
    "Totodile","Croconaw","Feraligatr","Sentret","Furret","Hoothoot","Noctowl",
    "Ledyba","Ledian","Spinarak","Ariados","Crobat","Chinchou","Lanturn",
    "Pichu","Cleffa","Igglybuff","Togepi","Togetic","Natu","Xatu","Mareep",
    "Flaaffy","Ampharos","Bellossom","Marill","Azumarill","Sudowoodo",
    "Politoed","Hoppip","Skiploom","Jumpluff","Aipom","Sunkern","Sunflora",
    "Yanma","Wooper","Quagsire","Espeon","Umbreon","Murkrow","Slowking",
    "Misdreavus","Unown","Wobbuffet","Girafarig","Pineco","Forretress",
    "Dunsparce","Gligar","Steelix","Snubbull","Granbull","Qwilfish",
    "Scizor","Shuckle","Heracross","Sneasel","Teddiursa","Ursaring",
    "Slugma","Magcargo","Swinub","Piloswine","Corsola","Remoraid",
    "Octillery","Delibird","Mantine","Skarmory","Houndour","Houndoom",
    "Kingdra","Phanpy","Donphan","Porygon2","Stantler","Smeargle",
    "Tyrogue","Hitmontop","Smoochum","Elekid","Magby","Miltank",
    "Blissey","Raikou","Entei","Suicune","Larvitar","Pupitar",
    "Tyranitar","Lugia","Ho-Oh","Celebi"
]

# -----------------------------
# UTILITIES
# -----------------------------
def is_nan_safe(val):
    if isinstance(val, numbers.Number):
        return np.isnan(val)
    if isinstance(val, pd.Timestamp):
        return not pd.isna(val)
    return False


def make_tooltip(row):
    props = []
    for col in row.index:
        if col != "geometry":
            val = row[col]
            if val is not None and not is_nan_safe(val):
                props.append(f"<b>{col}</b>: {val}")
    return "<br>".join(props)


# -----------------------------
# DISTRIBUTION EXTRACTION
# -----------------------------
def extract_distributions(G):
    node_type_counts = {}
    node_attrs_by_type = {}
    edge_attrs = []

    for n, data in G.nodes(data=True):
        t = data.get("node_type", "Junction")
        node_type_counts[t] = node_type_counts.get(t, 0) + 1

        node_attrs_by_type.setdefault(t, []).append(data)

    for _, _, data in G.edges(data=True):
        edge_attrs.append(data)

    count = len(node_type_counts.values())
    total = sum(node_type_counts.values())
    node_type_probs = {k: count / total for k, v in node_type_counts.items()}

    return node_type_probs, node_attrs_by_type, edge_attrs


# -----------------------------
# SAMPLING
# -----------------------------
def sample_node_attributes(node_type, node_attrs_by_type, layer=None, n_layers=None):
    base = node_attrs_by_type[node_type][
        np.random.randint(len(node_attrs_by_type[node_type]))
    ]

    new_data = {}
    for k, v in base.items():
        if isinstance(v, (int, float)) and not is_nan_safe(v):
            new_data[k] = float(v) * np.random.uniform(0.8, 1.2)
        else:
            new_data[k] = v

    new_data["node_type"] = node_type

    # -----------------------------
    # USE LAYER FOR GEOMETRY
    # -----------------------------
    if layer is not None and n_layers is not None:
        lat_min, lat_max = 50, 65
        band_height = (lat_max - lat_min) / n_layers

        # center of the layer band + small noise
        lat_center = lat_min + (layer + 0.5) * band_height
        lat = lat_center 
    else:
        print("Warning: No layering info, assigning random latitude")
        # fallback (no layering info)
        lat = np.random.uniform(50, 65)

    lon = np.random.uniform(-5, 15)

    new_data["geometry"] = Point(lon, lat)
    new_data["layer"] = layer  # store it here as well (optional redundancy)

    return new_data

def sample_edge_attributes(edge_attrs, u_geom, v_geom):
    base = edge_attrs[np.random.randint(len(edge_attrs))]

    new_data = {}
    for k, v in base.items():
        if isinstance(v, (int, float)) and not is_nan_safe(v):
            new_data[k] = float(v) * np.random.uniform(0.8, 1.2)
        else:
            new_data[k] = v

    if u_geom and v_geom:
        new_data["geometry"] = LineString([u_geom.coords[0], v_geom.coords[0]])

    return new_data


# -----------------------------
# STRUCTURE GENERATION
# -----------------------------
def generate_structure(num_nodes, node_types):
    """
    Layered graph:
    Generation → Intermediate → Market
    """

    G = nx.DiGraph()

    # -------------------------
    # Assign nodes
    # -------------------------
    for i in range(num_nodes):
        G.add_node(i, node_type=node_types[i])

    # -------------------------
    # Build layers
    # -------------------------
    gen_layer = [i for i, t in enumerate(node_types) if t == "Generation"]
    market_layer = [i for i, t in enumerate(node_types) if t == "Market"]
    inter_nodes = [i for i, t in enumerate(node_types)
                   if t not in ["Generation", "Market"]]

    # Split intermediate nodes into 2–3 layers
    n_inter_layers = max(1, min(3, len(inter_nodes)))
    split = np.array_split(np.random.permutation(inter_nodes), n_inter_layers)

    layers = [gen_layer] + [list(s) for s in split] + [market_layer]

    # -------------------------
    # Connect layers forward
    # -------------------------
    for l in range(len(layers) - 1):
        current_layer = layers[l]
        next_layer = layers[l + 1]

        if not current_layer or not next_layer:
            continue

        for u in current_layer:
            k = np.random.randint(1, min(4, len(next_layer)) + 1)
            targets = np.random.choice(next_layer, size=k, replace=False)

            for v in targets:
                G.add_edge(u, v)

    # -------------------------
    # Optional: skip-layer edges (adds realism)
    # -------------------------
    for l in range(len(layers) - 2):
        if np.random.rand() < 0.3:
            for u in layers[l]:
                v = np.random.choice(layers[l + 2])
                G.add_edge(u, v)

    # -------------------------
    # Ensure no isolated nodes
    # -------------------------
    for i, layer in enumerate(layers):
        for n in layer:
            G.nodes[n]["layer"] = i

    for n in G.nodes:
        if G.in_degree(n) == 0 and G.nodes[n]["node_type"] != "Generation":
            # connect from previous layer
            for layer in layers:
                if n in layer:
                    idx = layers.index(layer)
                    if idx > 0 and layers[idx - 1]:
                        u = np.random.choice(layers[idx - 1])
                        G.add_edge(u, n)
                    break

        if G.out_degree(n) == 0 and G.nodes[n]["node_type"] != "Market":
            # connect to next layer
            for layer in layers:
                if n in layer:
                    idx = layers.index(layer)
                    if idx < len(layers) - 1 and layers[idx + 1]:
                        v = np.random.choice(layers[idx + 1])
                        G.add_edge(n, v)
                    break

    return G

# -----------------------------
# SYNTHETIC NETWORK
# -----------------------------
def generate_synthetic_network(G_base, num_nodes):
    node_type_probs, node_attrs_by_type, edge_attrs = extract_distributions(G_base)

    # -------------------------
    # Structured node type assignment
    # -------------------------
    n_gen = max(1, int(0.2 * num_nodes))
    n_market = max(1, int(0.2 * num_nodes))
    n_inter = num_nodes - n_gen - n_market

    inter_types = ["Compression", "Junction", "Processing"]

    node_types = (
        ["Generation"] * n_gen +
        list(np.random.choice(inter_types, size=n_inter)) +
        ["Market"] * n_market
    )

    # Shuffle to avoid ordering bias in IDs
    node_types = list(np.random.permutation(node_types))

    # -------------------------
    # Generate structured graph
    # -------------------------
    G_new = generate_structure(num_nodes, node_types)
    # -------------------------
    # Assign node attributes
    # -------------------------
    for n in G_new.nodes:
        t = node_types[n]
        attrs = sample_node_attributes(t, node_attrs_by_type, layer = G_new.nodes[n]['layer'], n_layers = max(G_new.nodes[n]['layer'] for n in G_new.nodes))
        G_new.nodes[n].update(attrs)

    # -----------------------------
    # HARD CAP DEMAND
    # -----------------------------

    # Conversion factor: Mscm → GWh
    MSCm_to_GWh = 10.8

    # Total supply (convert to GWh)
    total_supply_mscm = sum(
    float(v)
    for n in G_new.nodes
    if (v := G_new.nodes[n].get("supply_capacity")) is not None
    and isinstance(v, (int, float))
    and not math.isnan(v)
    )

    total_supply_gwh = total_supply_mscm * MSCm_to_GWh

    # Maximum allowed demand
    max_total_demand = total_supply_gwh / 4

    # Collect demand nodes
    demand_nodes = [
        n for n in G_new.nodes
        if "average_demand_mwh_x1000" in G_new.nodes[n]
    ]

    total_demand = sum(
        float(v)
        for n in demand_nodes
        if (v := G_new.nodes[n].get("average_demand_mwh_x1000")) is not None
        and isinstance(v, (int, float))
        and not math.isnan(v)
    )

    # Apply hard cap (proportional scaling)
    if total_demand > max_total_demand and total_demand > 0:
        print(f"Scaling down demand from {total_demand:.2f} MWh to {max_total_demand:.2f} MWh")
        scale_factor = max_total_demand / total_demand

        for n in demand_nodes:
            G_new.nodes[n]["average_demand_mwh_x1000"] *= scale_factor


    # -------------------------
    # Assign edge attributes
    # -------------------------
    for u, v in G_new.edges:
        u_geom = G_new.nodes[u].get("geometry")
        v_geom = G_new.nodes[v].get("geometry")

        attrs = sample_edge_attributes(edge_attrs, u_geom, v_geom)
        G_new[u][v].update(attrs)

    return G_new


# -----------------------------
# FOLIUM PLOT
# -----------------------------
def plot_network(G):
    nodes = []
    for n, d in G.nodes(data=True):
        if "geometry" in d:
            row = d.copy()
            row["node_id"] = n
            nodes.append(row)

    edges = []
    for u, v, d in G.edges(data=True):
        if "geometry" in d:
            row = d.copy()
            row["from"] = u
            row["to"] = v
            edges.append(row)

    nodes = pd.DataFrame(nodes)
    edges = pd.DataFrame(edges)

    center = (
        nodes.geometry.apply(lambda g: g.y).mean(),
        nodes.geometry.apply(lambda g: g.x).mean()
    )

    m = folium.Map(location=center, zoom_start=5)

    # edges
    for _, row in edges.iterrows():
        geom = row.geometry
        lines = geom.geoms if isinstance(geom, MultiLineString) else [geom]

        for line in lines:
            coords = [(lat, lon) for lon, lat in line.coords]
            folium.PolyLine(coords, color="black", weight=2).add_to(m)

    # nodes
    colors = {
        "Generation": "#FFB3BA",      # pastel red
        "Processing": "#BAFFC9",      # pastel green
        "Market": "#BAE1FF",          # pastel blue
        "Compression": "#FFFFBA",     # pastel yellow
        "Junction": "#E0BBE4",        # pastel purple
    }

    for _, row in nodes.iterrows():
        tooltip = make_tooltip(row)
        g = row.geometry
        folium.CircleMarker(
            location=[g.y, g.x],
            radius=4,
            fill=True,
            color=colors.get(row["node_type"], "gray"),
            fill_color=colors.get(row["node_type"], "gray"),
            fill_opacity=0.9,
            tooltip=folium.Tooltip(tooltip, sticky=True),
        ).add_to(m)

    return m


# -----------------------------
# MAIN GENERATOR
# -----------------------------
def generate_and_store(G, amount=50):
    os.makedirs(FOLDER_PICKLE, exist_ok=True)
    os.makedirs(FOLDER_HTML, exist_ok=True)

    for i in range(amount):
        size = np.random.randint(5, 40)

        G_new = generate_synthetic_network(G, size)

        name = POKEMON_GEN2[i % len(POKEMON_GEN2)]

        pkl_path = os.path.join(FOLDER_PICKLE, f"{i+1:03d}_{name}.pkl")
        html_path = os.path.join(FOLDER_HTML, f"{i+1:03d}_{name}.html")

        with open(pkl_path, "wb") as f:
            pickle.dump(G_new, f)

        m = plot_network(G_new)
        m.save(html_path)

        print(f"Saved {name} ({size} nodes)")


# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    G = build_base_graph()
    generate_and_store(G, amount=100)