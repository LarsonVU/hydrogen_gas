from study_case_stochastic_model import solve_model, create_model, generate_cutting_plane_pairs
from study_case_problem_file import build_base_graph,  create_scenarios
import matplotlib.pyplot as plt
import networkx as nx
from pyomo.opt import TerminationCondition
import numpy as np
import time
import os
import pickle
import folium
import pandas as pd
from shapely.geometry import MultiLineString, LineString, Point
import numbers

# Folder
FOLDER = "study_case_model/unit_networks/"
FOLDER_HTML_NETWORKS = "data/data_analysis_results/unit_networks"

# List of first 151 Pokémon names (Kanto)
POKEMON_NAMES = [
    "Bulbasaur","Ivysaur","Venusaur","Charmander","Charmeleon","Charizard","Squirtle","Wartortle",
    "Blastoise","Caterpie","Metapod","Butterfree","Weedle","Kakuna","Beedrill","Pidgey","Pidgeotto",
    "Pidgeot","Rattata","Raticate","Spearow","Fearow","Ekans","Arbok","Pikachu","Raichu","Sandshrew",
    "Sandslash","Nidoran♀","Nidorina","Nidoqueen","Nidoran♂","Nidorino","Nidoking","Clefairy",
    "Clefable","Vulpix","Ninetales","Jigglypuff","Wigglytuff","Zubat","Golbat","Oddish","Gloom",
    "Vileplume","Paras","Parasect","Venonat","Venomoth","Diglett","Dugtrio","Meowth","Persian",
    "Psyduck","Golduck","Mankey","Primeape","Growlithe","Arcanine","Poliwag","Poliwhirl","Poliwrath",
    "Abra","Kadabra","Alakazam","Machop","Machoke","Machamp","Bellsprout","Weepinbell","Victreebel",
    "Tentacool","Tentacruel","Geodude","Graveler","Golem","Ponyta","Rapidash","Slowpoke","Slowbro",
    "Magnemite","Magneton","Farfetch'd","Doduo","Dodrio","Seel","Dewgong","Grimer","Muk","Shellder",
    "Cloyster","Gastly","Haunter","Gengar","Onix","Drowzee","Hypno","Krabby","Kingler","Voltorb",
    "Electrode","Exeggcute","Exeggutor","Cubone","Marowak","Hitmonlee","Hitmonchan","Lickitung",
    "Koffing","Weezing","Rhyhorn","Rhydon","Chansey","Tangela","Kangaskhan","Horsea","Seadra",
    "Goldeen","Seaking","Staryu","Starmie","Mr. Mime","Scyther","Jynx","Electabuzz","Magmar",
    "Pinsir","Tauros","Magikarp","Gyarados","Lapras","Ditto","Eevee","Vaporeon","Jolteon","Flareon",
    "Porygon","Omanyte","Omastar","Kabuto","Kabutops","Aerodactyl","Snorlax","Articuno","Zapdos",
    "Moltres","Dratini","Dragonair","Dragonite","Mewtwo","Mew"
]


# Base Parameters
NUMBER_OF_STAGES = 3
BRANCHES_PER_STAGE = {1 : 1, 2 : 2, 3: 4}
PRECISION = 0.001

def load_unit_network(filename):
    """
    Load a unit network graph from a pickle file.

    Parameters:
        filename (str): Path to the pickle file

    Returns:
        nx.DiGraph: The loaded graph
    """
    with open(filename, "rb") as f:
        G = pickle.load(f)
    return G


def is_nan_safe(val):
    # Only check numbers
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
            if val is not None and not is_nan_safe(val) and col != "compression_constants":
                props.append(f"<b>{col}</b>: {val}")
    return "<br>".join(props)


def plot_unit_network_folium(G, map_center=None, zoom_start=6):
    """
    Plot a NetworkX unit network on a Folium map using WGS 84 coordinates,
    with tooltips, node colors, and legend.
    Assumes G nodes have 'geometry' (Point) and edges have 'geometry' (LineString/MultiLineString)
    """
    # Extract nodes and edges as GeoDataFrame-like objects
    nodes = []
    for node, data in G.nodes(data=True):
        if "geometry" in data:
            row = data.copy()
            row["geometry"] = data["geometry"]
            row["node_id"] = node
            nodes.append(row)
    nodes = pd.DataFrame(nodes)

    edges = []
    for u, v, data in G.edges(data=True):
        if "geometry" in data:
            row = data.copy()
            row["geometry"] = data["geometry"]
            row["from_node"] = u
            row["to_node"] = v
            edges.append(row)
    edges = pd.DataFrame(edges)

    # Determine map center
    if map_center is None:
        if not nodes.empty:
            lat_mean = nodes.geometry.apply(lambda g: g.y).mean()
            lon_mean = nodes.geometry.apply(lambda g: g.x).mean()
            map_center = (lat_mean, lon_mean)
        else:
            map_center = (0, 0)

    m = folium.Map(location=map_center, zoom_start=zoom_start)

    # -------------------------------
    # Add edges
    # -------------------------------
    for _, row in edges.iterrows():
        geom = row.geometry
        lines = geom.geoms if isinstance(geom, MultiLineString) else [geom]

        for line in lines:
            coords = [(lat, lon) for lon, lat in line.coords]
            tooltip = make_tooltip(row)
            folium.PolyLine(
                coords,
                color="black",
                weight=3,
                opacity=0.8,
                tooltip=folium.Tooltip(tooltip, sticky=True),
            ).add_to(m)

    # -------------------------------
    # Add nodes
    # -------------------------------
    color_map = {
        "Generation": "#FFB3BA",      # pastel red
        "Processing": "#BAFFC9",      # pastel green
        "Market": "#BAE1FF",          # pastel blue
        "Compression": "#FFFFBA",     # pastel yellow
        "Junction": "#E0BBE4",        # pastel purple
    }

    for _, row in nodes.iterrows():
        point = row.geometry
        tooltip = make_tooltip(row)
        node_type = row.get("node_type", "Junction")
        color = color_map.get(node_type, "#CCCCCC")  # default pastel grey
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
    # Add legend
    # -------------------------------
    legend_html = '''
    <div style="position: fixed; 
         bottom: 50px; right: 50px; width: 200px; height: 200px; 
         background-color: white; border:2px solid grey; z-index:9999; 
         font-size:14px; padding: 10px">
         <p style="margin: 0 0 10px 0; font-weight: bold;">Node Types</p>
         <p style="margin: 5px 0;"><span style="background-color: #FFB3BA; width: 15px; height: 15px; display: inline-block; border-radius: 50%;"></span> Generation</p>
         <p style="margin: 5px 0;"><span style="background-color: #BAFFC9; width: 15px; height: 15px; display: inline-block; border-radius: 50%;"></span> Processing</p>
         <p style="margin: 5px 0;"><span style="background-color: #BAE1FF; width: 15px; height: 15px; display: inline-block; border-radius: 50%;"></span> Market</p>
         <p style="margin: 5px 0;"><span style="background-color: #FFFFBA; width: 15px; height: 15px; display: inline-block; border-radius: 50%;"></span> Compression</p>
         <p style="margin: 5px 0;"><span style="background-color: #E0BBE4; width: 15px; height: 15px; display: inline-block; border-radius: 50%;"></span> Junction</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

    return m

def reduce_graph(G, target_num_nodes):
    """
    Reduce a graph G to a target number of nodes by removing nodes,
    reconnecting neighbors, and removing orphaned nodes (no edges).

    Parameters:
        G (nx.DiGraph): Original graph
        target_num_nodes (int): Desired number of nodes

    Returns:
        nx.DiGraph: Reduced graph
    """
    if target_num_nodes >= len(G):
        return G.copy()

    G_reduced = G.copy()
    nodes_to_remove = len(G) - target_num_nodes

    # Randomly select nodes to remove
    removable_nodes = list(G_reduced.nodes)
    np.random.shuffle(removable_nodes)

    def remove_node_and_check_orphans(node):
        """Helper to remove node and recursively remove orphaned neighbors."""
        if node not in G_reduced:
            return 0  # Already removed

        preds = list(G_reduced.predecessors(node))
        succs = list(G_reduced.successors(node))

        # Connect predecessors to successors
        for u in preds:
            for v in succs:
                if u != v:  # Avoid self-loops

                    attrs = {}
                    geom_coords = []

                    # Edge u -> node
                    if G_reduced.has_edge(u, node):
                        edge1 = G_reduced[u][node]
                        attrs.update(edge1)

                        if "geometry" in edge1 and edge1["geometry"] is not None:
                            geom_coords.extend(list(edge1["geometry"].coords))

                    # Edge node -> v
                    if G_reduced.has_edge(node, v):
                        edge2 = G_reduced[node][v]
                        attrs.update(edge2)

                        if "geometry" in edge2 and edge2["geometry"] is not None:
                            coords2 = list(edge2["geometry"].coords)

                            # Avoid duplicate coordinate at join
                            if geom_coords and coords2:
                                if geom_coords[-1] == coords2[0]:
                                    geom_coords.extend(coords2[1:])
                                else:
                                    geom_coords.extend(coords2)
                            else:
                                geom_coords.extend(coords2)

                    # Update edge attributes
                    attrs["from_node"] = u
                    attrs["to_node"] = v

                    if geom_coords:
                        attrs["geometry"] = LineString([geom_coords[0], geom_coords[-1]])

                    G_reduced.add_edge(u, v, **attrs)

        # Remove the node
        G_reduced.remove_node(node)
        removed_count = 1

        # Check for orphaned neighbors
        for neighbor in preds + succs:
            if (
                neighbor in G_reduced
                and G_reduced.in_degree(neighbor) == 0
                and G_reduced.out_degree(neighbor) == 0
            ):
                removed_count += remove_node_and_check_orphans(neighbor)

        return removed_count

    idx = 0
    while nodes_to_remove > 0 and idx < len(removable_nodes):
        node = removable_nodes[idx]
        removed = remove_node_and_check_orphans(node)
        nodes_to_remove -= removed
        idx += 1

    return G_reduced


def plot_graph(G):
    plt.figure(figsize=(10, 10))
    
    # Use a layout for positions
    pos = nx.spring_layout(G, seed=42)  # Positions nodes automatically

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=50, node_color="skyblue")

    # Draw curved edges
    nx.draw_networkx_edges(
        G,
        pos,
        width=0.8,
        connectionstyle="arc3,rad=0.2"
    )

    # Edge labels
    edge_labels = {
        (u, v): d.get("idPipeline", "")
        for u, v, d in G.edges(data=True)
    }

    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=edge_labels,
        font_size=6,
        connectionstyle="arc3,rad=0.2"
    )

    plt.title("NetworkX Graph View with Curved idPipeline")
    plt.axis("off")
    plt.show()

def zero_graph_demands(G):
    """
    Set all node demands in the graph G to zero, if they exist.
    
    Parameters:
        G (nx.DiGraph): The graph to modify
    """
    for node in G.nodes:
        node_data = G.nodes[node]
        if "average_demand_mwh_x1000" in node_data and node_data.get("average_demand_mwh_x1000") is not None:
            node_data["average_demand_mwh_x1000"] = 0
    return G

def make_and_store_unit_networks(G, amount=151, folder=FOLDER):
    """
    Create 'unit networks' from G and store them with Pokémon names.

    Parameters:
        G (nx.DiGraph): Base graph
        amount (int): Number of unit networks to create
        folder (str): Folder to store the graphs
    """
    os.makedirs(folder, exist_ok=True)
    total_nodes = len(G) 

    for i in range(amount):
        target_nodes = max(2, np.random.randint(2, total_nodes+1))  # random smaller graph size
        G_small = reduce_graph(G, target_nodes)

        # Pick Pokémon name
        name = POKEMON_NAMES[i % len(POKEMON_NAMES)]

        # Store as pickle
        filename = os.path.join(folder, f"{i+1:03d}_{name}.pkl")
        with open(filename, "wb") as f:
            pickle.dump(G_small, f)

        print(f"Stored unit network {i+1}: {name} ({target_nodes} nodes)")


def make_and_store_all_unit_network_maps(G, amount=151, folder_pickle=FOLDER, folder_html=FOLDER_HTML_NETWORKS):
    """
    Generate unit networks, store them as pickles, and save Folium maps for each.

    Parameters:
        G (nx.DiGraph): Original base graph
        amount (int): Number of unit networks to generate
        folder_pickle (str): Folder to store pickle networks
        folder_html (str): Folder to store HTML Folium maps
    """
    # 1️⃣ Generate and store unit networks as pickles
    make_and_store_unit_networks(G, amount=amount, folder=folder_pickle)

    # 2️⃣ Ensure output folder exists
    os.makedirs(folder_html, exist_ok=True)

    # 3️⃣ Loop over each pickle file
    pickle_files = sorted([f for f in os.listdir(folder_pickle) if f.endswith(".pkl")])
    for pf in pickle_files:
        # Load network
        G_unit = load_unit_network(os.path.join(folder_pickle, pf))

        # Plot network in Folium
        m = plot_unit_network_folium(G_unit)

        # Save HTML map
        html_filename = os.path.join(folder_html, pf.replace(".pkl", ".html"))
        m.save(html_filename)
        print(f"Saved unit network map: {html_filename}")

if __name__ == "__main__":
    G =build_base_graph()
    G = zero_graph_demands(G)
    np.random.seed(123)
    make_and_store_all_unit_network_maps(G, 151)
