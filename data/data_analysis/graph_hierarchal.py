import ast
import geopandas as gpd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import yaml
import os

BIG = False

# -------------------------------
# Load config
# -------------------------------
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



config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config.yaml')
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# -------------------------------
# Config
# -------------------------------
if BIG:
    input_path = config['paths']['bigger_network_geojson']
else:
    input_path = config['paths']['geojson_output']

# -------------------------------
# Load data
# -------------------------------
gdf = gpd.read_file(input_path)

nodes = gdf[gdf.geometry.type == "Point"].copy()
edges = gdf[gdf.geometry.type.isin(["LineString", "MultiLineString"])].copy()

# -------------------------------
# Build directed graph
# -------------------------------
G = nx.DiGraph()

# Adjust these column names if needed
NODE_ID_COL = "location"
SOURCE_COL = "from_node"
TARGET_COL = "to_node"


# Add nodes
for _, row in nodes.iterrows():
    node_id = row[NODE_ID_COL]
    G.add_node(node_id, **row.to_dict())

# Add edges
for _, row in edges.iterrows():
    source = find_node_by_coords(nodes, row[SOURCE_COL])
    target = find_node_by_coords(nodes, row[TARGET_COL])
    G.add_edge(source, target, **row.to_dict())

# -------------------------------
# Handle cycles (condense graph)
# -------------------------------
if not nx.is_directed_acyclic_graph(G):
    print("Graph has cycles → condensing strongly connected components")

    G_condensed = nx.condensation(G)
    
    # Map condensed nodes back to original nodes
    component_map = G_condensed.graph["mapping"]

    # Reverse mapping: component → original nodes
    comp_to_nodes = {}
    for node, comp in component_map.items():
        comp_to_nodes.setdefault(comp, []).append(node)

    G_plot = G_condensed
else:
    G_plot = G
    comp_to_nodes = {n: [n] for n in G.nodes}

# -------------------------------
# Compute hierarchical levels
# -------------------------------
levels = {}

current_level = [n for n in G_plot.nodes if G_plot.in_degree(n) == 0]
level = 0

while current_level:
    next_level = []
    for node in current_level:
        levels[node] = level
        for neighbor in G_plot.successors(node):
            if all(pred in levels for pred in G_plot.predecessors(neighbor)):
                next_level.append(neighbor)

    current_level = list(set(next_level))
    level += 1

# -------------------------------
# Assign positions
# -------------------------------
pos = {}
level_nodes = {}

def parent_center(node):
    preds = list(G_plot.predecessors(node))

    if not preds:
        return 0  # root-ish nodes stay centered
    return np.mean([pos[p][1] for p in preds if p in pos])

for node, lvl in levels.items():
    level_nodes.setdefault(lvl, []).append(node)

for lvl in sorted(level_nodes.keys()):

    nodes = level_nodes[lvl]



    # sort by priority (optional: most constrained first)
    nodes = sorted(nodes, key=parent_center, reverse=True)

    used_positions = set()
    spacing = 1  # try 2 or 3 for more spread
    for node in nodes:
        target = parent_center(node)

        # find closest free slot
        candidate = round(target)

        # search outward if occupied
        step = 0
        base = round(target)

        while candidate in used_positions:
            step += 1
            offset = ((step + 1) // 2) * spacing

            if step % 2 == 1:
                candidate = base + offset
            else:
                candidate = base - offset

        used_positions.add(candidate)

        pos[node] = (lvl, candidate)

# -------------------------------
# Node coloring (same as folium)
# -------------------------------
color_map = {
    "Generation": "#FFB3BA",   # pastel red
    "Processing": "#BAFFC9",   # pastel green
    "Market": "#BAE1FF",       # pastel blue
    "Compression": "#FFFFBA",  # pastel yellow
    "Junction": "#E0BBE4",     # pastel purple
}

node_colors = []

for node in G_plot.nodes:
    original_nodes = comp_to_nodes[node]

    # If condensed node → take first node’s type
    original_node = original_nodes[0]
    node_type = G.nodes[original_node].get("node_type", "Junction")

    node_colors.append(color_map.get(node_type, "#CCCCCC"))



def wrap_label(label, max_line_len=9):
    parts = label.replace("/", " ").replace("-", " ").split()
    
    lines = []
    current = ""
    
    for part in parts:
        if len(current) + len(part) + 1 <= max_line_len:
            current = f"{current} {part}".strip()
        else:
            lines.append(current)
            current = part
    if current:
        lines.append(current)
    
    return "\n".join(lines)

wrapped_labels = {node: wrap_label(str(node)) for node in G_plot.nodes()}


# -------------------------------
# Plot
# -------------------------------
fig, ax = plt.subplots(figsize=(14, 10))

# Build legend
legend_elements = [
    Patch(facecolor=color, edgecolor='black', label=label)
    for label, color in color_map.items()
]

# Add legend to plot
ax.legend(
    handles=legend_elements,
    title="Node Types",
    loc="upper right",   # try "best" or "upper left" if overlapping
    fontsize=8,
    title_fontsize=9
)


if BIG:
    nx.draw(
        G_plot,
        pos,
        with_labels=True,
        labels=wrapped_labels,
        node_color=node_colors,
        node_size=1100,
        font_size=5,
        arrows=True,
        ax=ax
    )
    ax.set_title("Hierarchical Network Layout of the Bigger Network")
else:
    nx.draw(
        G_plot,
        pos,
        with_labels=True,
        labels=wrapped_labels,
        node_color=node_colors,
        node_size=2200,
        font_size=8,
        arrows=True,
        ax=ax
    )
    ax.set_title("Hierarchical Network Layout of the Study Case")


ax.axis("off")
if BIG:
    plt.savefig(config["paths"]["other_experiments_figures"] + "hier_graph_big.png")
else:
    plt.savefig(config["paths"]["other_experiments_figures"] + "hierarchal_graph.png")
plt.show()