import networkx as nx
import geopandas as gpd
import matplotlib.pyplot as plt
import ast
import numpy as np
import math
from collections import Counter
import pandas as pd
import geopandas as gpd
import os
from shapely.geometry import LineString, Point


np.random.seed(42)

GEOJSON_FILE = "data/data_analysis_results/Geojson_pipelines/study_case_network.geojson"

NUMBER_OF_STAGES = 3
BRANCHES_PER_STAGE = {1 : 1, 2 : 5, 3: 5}



def find_node_by_coords(node_rows, coords, tol=1e-6):
    coords = ast.literal_eval(coords)
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

def aggregate_features(attrs: dict, cluster: list):
    """
    Aggregate node features across a cluster of node attribute dicts.

    Parameters
    ----------
    attrs : dict
        Base dict (will be modified and returned)
    cluster : list[dict]
        List of attribute dicts from nodes being merged
    """

    def to_float(val, default=0.0):
        try:
            if val is None:
                return default
            return float(val)
        except (TypeError, ValueError):
            return default

    def safe_sum(key):
        # Convert values to floats, skipping None or invalid entries
        values = [to_float(x.get(key), default=None) for x in cluster]
        # Filter out None values
        valid_values = [v for v in values if v is not None]
        # Return None if no valid numbers, else sum
        return None if not valid_values else sum(valid_values)
    
    def safe_mean(key):
        return sum(to_float(x.get(key, 0)) or 0 for x in cluster) / len(cluster)

    def safe_max(key):
        vals = [to_float(x.get(key,0)) for x in cluster if x.get(key) is not None]
        return max(vals) if vals else None

    def safe_min(key):
        vals = [to_float(x.get(key, 'inf')) for x in cluster if x.get(key) is not None]
        return min(vals) if vals else None


    # -------------------------------------------------
    # Supply / demand totals
    # -------------------------------------------------
    attrs["supply_capacity"] = safe_sum("supply_capacity")
    attrs["average_demand_mwh_x1000"] = safe_sum("average_demand_mwh_x1000")
    attrs["max_flow"] = safe_sum("max_flow")

    # -------------------------------------------------
    # Compression / pressure constraints
    # -------------------------------------------------
    attrs["min_outlet_pressure"] = safe_max("min_outlet_pressure")

    # -------------------------------------------------
    # Dominant supplier selection
    # -------------------------------------------------
    dominant = None
    if attrs["supply_capacity"] is not None:
        dominant = max(
            cluster,
            key=lambda x: to_float(x.get("supply_capacity", 0)) or 0
        )

    if dominant:
        attrs["supplier"] = dominant.get("supplier")
        attrs["generation_cost"] = dominant.get("generation_cost")
        attrs["component_ratio"] = dominant.get("component_ratio")
    else:
        attrs["supplier"] = None
        attrs["generation_cost"] = None
        attrs["component_ratio"] = None
        attrs["compression_constants"] = None

    # -------------------------------------------------
    # Demand-weighted market stats
    # -------------------------------------------------
    total_demand = attrs["average_demand_mwh_x1000"]

    def demand_weighted(key):
        if total_demand == 0 or total_demand is None:
            return None
        return sum(
            (to_float(x.get(key, 0)) or 0) * (to_float(x.get("average_demand_mwh_x1000", 0)) or 0)
            for x in cluster
        ) / total_demand

    dominant = None
    if attrs["average_demand_mwh_x1000"] is not None:
        dominant = max(
            cluster,
            key=lambda x: to_float(x.get("average_demand_mwh_x1000", 0)) or 0
        )

    attrs["average_market_price"] = demand_weighted("average_market_price")
    attrs["long_term_price_std"] = demand_weighted("long_term_price_std")
    attrs["day_ahead_price_std"] = demand_weighted("day_ahead_price_std")
    attrs["demand_variance"] = demand_weighted("demand_variance")

    if dominant:
        attrs["supplier_ratios"] = dominant.get("supplier_ratios")
        attrs["max_fractions"] = dominant.get("max_fractions")
    else:
        attrs["supplier_ratios"] = None
        attrs["max_fractions"] =None

    # -------------------------------------------------
    # Booking costs → weighted by supply
    # -------------------------------------------------
    attrs["base_booking_cost"] = safe_mean("base_booking_cost")

    # -------------------------------------------------
    # Node type resolution
    # -------------------------------------------------
    types = [x.get("node_type") for x in cluster]
    if attrs["supply_capacity"] is not None:
        attrs["node_type"] = "Generation"
    elif  attrs["average_market_price"] is not None:
        attrs["node_type"] = "Market"
    elif "Processing" in types:
        attrs["node_type"] = "Processing"
    else:
        attrs["node_type"] = "Junction"

    if attrs["node_type"] == "Compression":
        attrs["compression_increase"] = [x.get("compression_increase") for x in cluster]
        attrs["compression_constants"] = [x.get("compression_constants") for x in cluster]

    return attrs


def merge_edge_attributes(G_merged, new_u, new_v, edge_data):
    # Sum capacity
    G_merged[new_u][new_v]["capacity"] = (
        G_merged[new_u][new_v].get("capacity", 0) + edge_data.get("capacity", 0)
    )

    # Sum arc cost
    G_merged[new_u][new_v]["pressure_cost"] = (
        G_merged[new_u][new_v].get("pressure_cost", 0) + edge_data.get("pressure_cost", 0)
    )

    # Max pressure (take the max of existing and new)
    G_merged[new_u][new_v]["max_inlet_pressure"] = max(
        G_merged[new_u][new_v].get("max_inlet_pressure", 0),
        edge_data.get("max_inlet_pressure", 0)
    )

    # Average arc max ratios
    existing_ratios = G_merged[new_u][new_v].get("max_pipe_fractions", [])
    new_ratios = edge_data.get("max_pipe_fractions", [])
    if existing_ratios and new_ratios:
        # Assuming list of dicts
        averaged_ratios = []
        for d1, d2 in zip(existing_ratios, new_ratios):
            avg_dict = {k: (d1.get(k, 0) + d2.get(k, 0)) / 2 for k in set(d1) | set(d2)}
            averaged_ratios.append(avg_dict)
        G_merged[new_u][new_v]["max_pipe_fractions"] = averaged_ratios
    else:
        G_merged[new_u][new_v]["max_pipe_fractions"] = existing_ratios or new_ratios

    # Average Weymouth constant
    existing_const = G_merged[new_u][new_v].get("weymouth_constant", 0)
    new_const = edge_data.get("weymouth_constant", 0)
    if existing_const and new_const:
        G_merged[new_u][new_v]["weymouth_constant"] = (existing_const + new_const) / 2
    else:
        G_merged[new_u][new_v]["weymouth_constant"] = existing_const or new_const
    

def merge_close_nodes(G, merge_distance=50):
    """
    Merge nodes whose geometry is within merge_distance (km).

    Returns
    -------
    G_merged : nx.DiGraph
        New graph with merged nodes
    merge_map : dict
        {new_node_id: [old_node_ids]}
    """

    G = G.copy()
    nodes = list(G.nodes(data=True))

    # Track which nodes have been merged
    visited = set()
    clusters = []

    # -----------------------------------
    # 1. Build clusters of close nodes
    # -----------------------------------
    for i, (n1, data1) in enumerate(nodes):
        if n1 in visited:
            continue

        cluster = [n1]
        visited.add(n1)

        pt1 = data1["geometry"]

        for j in range(i + 1, len(nodes)):
            n2, data2 = nodes[j]
            if n2 in visited:
                continue

            pt2 = data2["geometry"]
            dist = haversine_km(pt1, pt2)

            if dist <= merge_distance:
                cluster.append(n2)
                visited.add(n2)

        clusters.append(cluster)

    # -----------------------------------
    # 2. Build new merged graph
    # -----------------------------------
    G_merged = nx.DiGraph()
    merge_map = {}

    for cluster in clusters:

        if len(cluster) == 1:
            # Keep node as-is
            node_id = cluster[0]
            G_merged.add_node(node_id, **G.nodes[node_id])
            merge_map[node_id] = [node_id]
            continue

        # Create new merged node id
        new_node_id = "_".join(sorted(cluster))

        # Store which nodes were merged
        merge_map[new_node_id] = cluster

        # -----------------------------------
        # Aggregate attributes
        # -----------------------------------
        
        cluster_attrs = [G.nodes[n] for n in cluster]

        attrs = aggregate_features({}, cluster_attrs)
        attrs["location"] = new_node_id
        # Use centroid of merged nodes
        geometries = [G.nodes[n]["geometry"] for n in cluster]
        centroid = gpd.GeoSeries(geometries).union_all().centroid
        attrs["geometry"] = centroid

        G_merged.add_node(new_node_id, **attrs)
        
    # For all market nodes replace demand fractions
    for node in G_merged.nodes():
        G_merged.nodes[node]["supplier_ratios"] = [{"Equinor Energy AS": 1.0, "SHELL": 0.0, "Vår Energi ASA": 0.0}]

    # -----------------------------------
    # 3. Reconnect edges
    # -----------------------------------

    for u, v, edge_data in G.edges(data=True):

        # Find merged node representatives
        new_u = next(k for k, vals in merge_map.items() if u in vals)
        new_v = next(k for k, vals in merge_map.items() if v in vals)

        if new_u == new_v:
            continue  # remove self-loops after merge

        # Get new node geometries
        geom_u = G_merged.nodes[new_u].get("geometry")
        geom_v = G_merged.nodes[new_v].get("geometry")

        # Rebuild edge geometry
        new_geom = None
        if geom_u is not None and geom_v is not None:
            new_geom = LineString([geom_u, geom_v])

        if G_merged.has_edge(new_u, new_v):
            merge_edge_attributes(G_merged, new_u, new_v, edge_data)

            # Replace geometry with updated one
            if new_geom is not None:
                G_merged[new_u][new_v]["geometry"] = new_geom

        else:
            # Copy attributes but overwrite geometry
            attrs = dict(edge_data)
            if new_geom is not None:
                attrs["geometry"] = new_geom

            G_merged.add_edge(new_u, new_v, **attrs)

    return G_merged, merge_map

# Graph with scenario independent attributes
def build_smaller_graph(merge_distance = 100):
    """
    Build scenario-independent base graph from study_case_network.geojson
    """

    gdf = gpd.read_file(GEOJSON_FILE)

    G = nx.DiGraph()

    # -----------------------------
    # Add Nodes
    # -----------------------------
    node_rows = gdf[gdf["type"] == "node"]

    for _, row in node_rows.iterrows():

        node_id = row["location"]  # use readable name as node id

        # Convert row to dict and remove geometry
        attributes = row.to_dict()

        G.add_node(node_id, **attributes)

    # -----------------------------
    # Add Edges
    # -----------------------------
    edge_rows = gdf[gdf["type"] == "edge"]

    for _, row in edge_rows.iterrows():        
        # Find matching node locations by coordinates
        from_node = find_node_by_coords(node_rows, row["from_node"])
        to_node = find_node_by_coords(node_rows, row["to_node"])

        attributes = row.to_dict()

        G.add_edge(from_node, to_node, **attributes)
    
    G, merge_map = merge_close_nodes(G, merge_distance= merge_distance)

    return G, merge_map


def haversine_km(pt1, pt2):
    # pt1 and pt2 are shapely Points in (lon, lat)
    lon1, lat1 = pt1.x, pt1.y
    lon2, lat2 = pt2.x, pt2.y
    R = 6371.0  # Earth radius in km

    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c

def plot_network(G):
    plt.figure(figsize=(10, 10))

    # Extract coordinates from shapely geometry
    pos = {
        node: (data["geometry"].x, data["geometry"].y)
        for node, data in G.nodes(data=True)
        if "geometry" in data
    }

    nx.draw(G, pos, node_size=10, with_labels=False, width=0.8)

    plt.title("NetworkX Graph View")
    plt.show()


def save_geojson(destination_folder = "data_analysis_results"):
    nodes_records = []
    for node, data in G.nodes(data=True):
        record = data.copy()
        nodes_records.append(record)

    nodes_gdf = gpd.GeoDataFrame(
        nodes_records, geometry="geometry",
        crs="EPSG:4326"
    )

    edge_records = []
    for u, v, data in G.edges(data=True):
        record = data.copy()
        record["geometry"] = data["geometry"]
        record["from_node"] = u
        record["to_node"] = v
        edge_records.append(record)

    edges_gdf = gpd.GeoDataFrame(edge_records, crs="EPSG:4326", geometry="geometry")

    # Combined export
    nodes_gdf["type"] = "node"
    edges_gdf["type"] = "edge"

    combined_gdf = gpd.GeoDataFrame(
        pd.concat([nodes_gdf, edges_gdf], ignore_index=True),
        crs= "EPSG:4326"
    )

    output_path = destination_folder + f"smaller_network.geojson"
    combined_gdf.to_file(output_path, driver="GeoJSON")

    print("Exported:", output_path) 

if __name__ == "__main__":
    G, merge_map = build_smaller_graph(150)
    save_geojson("data/data_analysis_results/Geojson_pipelines/")
    