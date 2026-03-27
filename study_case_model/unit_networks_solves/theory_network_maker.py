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
FOLDER_PICKLE = "unit_networks_solves/unit_gen2_networks/"
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
def sample_node_attributes(node_type, node_attrs_by_type):
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

    # Add random geometry (Europe-ish bounding box)
    lon = np.random.uniform(-5, 15)
    lat = np.random.uniform(50, 65)
    new_data["geometry"] = Point(lon, lat)

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
    G = nx.DiGraph()

    for i in range(num_nodes):
        G.add_node(i, node_type=node_types[i])

    nodes = list(G.nodes)

    for i in nodes:
        ti = node_types[i]

        # Market nodes should not have outgoing edges
        if ti == "Market":
            continue

        # Possible targets (respecting rules)
        possible_targets = [
            j for j in nodes
            if j != i and node_types[j] != "Generation"
        ]

        if not possible_targets:
            continue

        # Sample number of connections (1 to 5, but not exceeding available nodes)
        k = np.random.randint(1, min(5, len(possible_targets)) + 1)

        targets = np.random.choice(possible_targets, size=k, replace=False)

        for j in targets:
            G.add_edge(i, j)
            
    # Ensure no isolated nodes
    for n in nodes:
        if G.degree(n) == 0:
            target = np.random.choice(nodes)
            if n != target:
                G.add_edge(n, target)

    return G


# -----------------------------
# SYNTHETIC NETWORK
# -----------------------------
def generate_synthetic_network(G_base, num_nodes):
    node_type_probs, node_attrs_by_type, edge_attrs = extract_distributions(G_base)

    node_types = np.random.choice(
        list(node_type_probs.keys()),
        size=num_nodes,
        p=list(node_type_probs.values())
    )

    G_new = generate_structure(num_nodes, node_types)

    # Assign node attributes
    for n in G_new.nodes:
        t = node_types[n]
        attrs = sample_node_attributes(t, node_attrs_by_type)
        G_new.nodes[n].update(attrs)

    # Assign edges
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