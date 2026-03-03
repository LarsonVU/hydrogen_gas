import geopandas as gpd
import networkx as nx
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point
from shapely import set_precision
import pandas as pd
from collections import Counter
import math
import os
import random



# ==============================
# SETTINGS
# ==============================
NODE_MERGE_THRESHOLD = 20  # km
FOLDER = "data/data_analysis_results/Geojson_pipelines/"
FILE_PATH = FOLDER + "pipLine.geojson"
REGULAR_GRAPH = FOLDER + "study_case_network.geojson"

# ==============================
# UTILITIES
# ==============================
def majority_feature(values):
    if not values:
        return None
    return Counter(values).most_common(1)[0][0]


def haversine_km(pt1, pt2):
    lon1, lat1 = pt1.x, pt1.y
    lon2, lat2 = pt2.x, pt2.y
    R = 6371.0

    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def reverse_linestring(geom):
    if isinstance(geom, LineString):
        return LineString(list(geom.coords)[::-1])
    return geom


# ==============================
# 1. LOAD + FILTER DATA
# ==============================
def load_and_filter_data(file_path):
    gdf = gpd.read_file(file_path)

    print("Loaded features:", len(gdf))
    print("Geometry types:", gdf.geom_type.unique())

    gdf = gdf[gdf["medium"] == "Gas"]
    gdf = gdf[gdf["curPhase"] == "IN SERVICE"]

    pipelines_to_reverse = [314542, 322498, 323212, 327394,
                            375427, 406790, 414837, 442969]

    mask = gdf["idPipeline"].isin(pipelines_to_reverse)
    gdf.loc[mask, "geometry"] = gdf.loc[mask, "geometry"].apply(reverse_linestring)

    pipelines_to_remove = [321818, 370369, 326068, 438653]
    gdf = gdf[~gdf["idPipeline"].isin(pipelines_to_remove)]

    print("Filtered features:", len(gdf))
    return gdf


# ==============================
# 2. CLEAN GEOMETRIES
# ==============================
def clean_geometries(gdf):
    if gdf.geom_type.isin(["Polygon", "MultiPolygon"]).any():
        print("Converting polygons to boundaries...")
        gdf["geometry"] = gdf.geometry.boundary

    gdf["geometry"] = gdf.geometry.apply(lambda g: set_precision(g, 1e-8))
    return gdf


# ==============================
# 3. FLATTEN MULTILINES
# ==============================
def flatten_lines(gdf):
    records = []

    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        props = row.drop("geometry").to_dict()

        if geom.geom_type == "LineString":
            records.append({"geometry": geom, "properties": props})

        elif geom.geom_type == "MultiLineString":
            for part in geom.geoms:
                records.append({"geometry": part, "properties": props})

    print("Total pipelines after flattening:", len(records))
    return records


# ==============================
# 4. CLUSTER HUBS
# ==============================
def cluster_endpoints(lines):
    endpoints = []
    for rec in lines:
        coords = list(rec["geometry"].coords)
        endpoints.append(Point(coords[0]))
        endpoints.append(Point(coords[-1]))

    print("Total endpoints:", len(endpoints))

    unique_nodes = []
    used = set()

    for i, pt1 in enumerate(endpoints):
        if i in used:
            continue

        cluster = [pt1]
        used.add(i)

        for j, pt2 in enumerate(endpoints):
            if j <= i or j in used:
                continue
            if haversine_km(pt1, pt2) < NODE_MERGE_THRESHOLD:
                cluster.append(pt2)
                used.add(j)

        avg_x = sum(p.x for p in cluster) / len(cluster)
        avg_y = sum(p.y for p in cluster) / len(cluster)
        unique_nodes.append(Point(avg_x, avg_y))

    print("Detected hubs:", len(unique_nodes))
    return unique_nodes


# ==============================
# 5. SNAP TO HUBS
# ==============================
def find_closest_node(point, nodes):
    return min(nodes, key=lambda n: point.distance(n))


def snap_pipelines(lines, unique_nodes):
    connected = []

    for rec in lines:
        geom = rec["geometry"]
        props = rec["properties"]

        coords = list(geom.coords)
        start_pt = Point(coords[0])
        end_pt = Point(coords[-1])

        closest_start = find_closest_node(start_pt, unique_nodes)
        closest_end = find_closest_node(end_pt, unique_nodes)

        new_line = LineString([closest_start, closest_end])

        if new_line.coords[0] != new_line.coords[-1]:
            connected.append({
                "geometry": new_line,
                "original_length": geom.length,
                "properties": props
            })

    print("Connected pipelines:", len(connected))
    return connected


# ==============================
# 6. BUILD GRAPH
# ==============================
def build_graph(connected_lines):
    G = nx.DiGraph()

    for rec in connected_lines:
        geom = rec["geometry"]
        props = rec["properties"]

        coords = list(geom.coords)
        start, end = coords[0], coords[-1]

        G.add_node(start)
        G.add_node(end)

        attrs = props.copy()
        attrs.update({"geometry": geom, "weight": rec["original_length"]})
        edge_key = attrs.get("idPipeline", None)
        G.add_edge(start, end, key=edge_key, **attrs)

    print("Graph built successfully")
    print("Nodes:", G.number_of_nodes())
    print("Edges:", G.number_of_edges())
    return G


# ==============================
# 7. ADD NODE FEATURES
# ==============================
def add_node_features(G):

    def get_majority(values):
        vals = [v for v in values if v]
        return majority_feature(vals) if vals else None

    for node in G.nodes():
        inlet = list(G.in_edges(node, data=True))
        outlet = list(G.out_edges(node, data=True))

        # -------------------------
        # Collect location features
        # -------------------------
        inlet_loc = [d.get("toFacility") for _, _, d in inlet if "toFacility" in d]
        outlet_loc = [d.get("fromFacili") for _, _, d in outlet if "fromFacili" in d]

        loc = get_majority(inlet_loc + outlet_loc)
        if loc:
            G.nodes[node]["location"] = loc

        # -------------------------
        # Collect location_id features
        # -------------------------
        inlet_loc_id = [d.get("idToFacili") for _, _, d in inlet if "idToFacili" in d]
        outlet_loc_id = [d.get("idFrmFacil") for _, _, d in outlet if "idFrmFacil" in d]

        loc_id = get_majority(inlet_loc_id + outlet_loc_id)

        if loc_id:
            # -------------------------
            # Deduplicate IDs (A/B/C suffix)
            # -------------------------
            existing_ids = [
                G.nodes[n].get("location_id")
                for n in G.nodes()
                if G.nodes[n].get("location_id")
            ]

            id_count = sum(1 for eid in existing_ids if str(eid).startswith(str(loc_id)))

            if id_count > 0:
                used_letters = set()
                for eid in existing_ids:
                    if str(eid).startswith(str(loc_id)):
                        suffix = str(eid)[len(str(loc_id)):]
                        if suffix and suffix[0].isalpha():
                            used_letters.add(suffix[0])

                for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                    if letter not in used_letters:
                        loc_id = f"{loc_id}{letter}"
                        break

            G.nodes[node]["location_id"] = loc_id
            print(G.nodes[node]["location"], G.nodes[node]["location_id"] )
    return G

# ==============================
# Add features
# ==============================

def add_node_types(G):
    processing_locations = {"NYHAMNA", "KOLLSNES", "KÅRSTØ"}

    for node in G.nodes():
        location = G.nodes[node].get("location", None)

        # 1. Processing nodes (priority rule)
        if location in processing_locations:
            G.nodes[node]["node_type"] = "Processing"
            continue

        in_deg = G.in_degree(node)
        out_deg = G.out_degree(node)

        # 2. Generation node (no incoming edges)
        if in_deg == 0 and out_deg > 0:
            G.nodes[node]["node_type"] = "Generation"

        # 3. Market node (no outgoing edges)
        elif out_deg == 0 and in_deg > 0:
            G.nodes[node]["node_type"] = "Market"

        # 4. Compression node (incoming edges but exactly one outgoing)
        elif in_deg > 0 and out_deg == 1:
            G.nodes[node]["node_type"] = "Compression"

        # 5. Otherwise junction
        else:
            G.nodes[node]["node_type"] = "Junction"

    return G

def add_features_from_base_graph(G, filename=REGULAR_GRAPH):

    case_study_graph = gpd.read_file(filename)

    # Ensure fast lookup by location_id
    base_lookup = case_study_graph["location_id"].unique()
    # Group existing nodes in case_study_graph by node_type for fallback sampling
    nodes_by_type = {}
    for idx, row in case_study_graph.iterrows():
        node_type = row.get("node_type")
        if node_type:
            nodes_by_type.setdefault(node_type, []).append(row)

    for node, data in G.nodes(data=True):
        loc_id = str(data.get("location_id"))

        # -------------------------
        # 1️⃣ Direct match by location_id
        # -------------------------
        if loc_id in base_lookup:
            print("exact match for", G.nodes[node]["location"])
            base_row = case_study_graph[case_study_graph["location_id"] == loc_id].iloc[0]

            # Copy all columns except geometry
            for col in case_study_graph.columns:
                if col != "geometry":
                    G.nodes[node][col] = base_row[col]

        # -------------------------
        # 2️⃣ No match → sample from same node_type
        # -------------------------
        else:
            node_type = data.get("node_type")

            if node_type in nodes_by_type and nodes_by_type[node_type]:
                print("Sampled node for", G.nodes[node]["location"])
                # Sample a random node of same type
                sampled_row = random.choice(nodes_by_type[node_type])
                for col in case_study_graph.columns:
                    if col not in ["geometry", "location",  "location_id"]:
                        G.nodes[node][col] = sampled_row[col]

            else:
                print(f"Warning: No fallback nodes found for type {node_type}")

    return G

def add_edge_features_from_base_graph(G, filename=REGULAR_GRAPH):
    case_study_graph = gpd.read_file(filename)

    # Only keep edges (assuming pipelines are LineStrings)
    base_lookup = case_study_graph["idPipeline"].dropna().unique()
    for u, v, data in G.edges(data=True):
        pipe_id = data.get("idPipeline")

        # -------------------------
        # 1️⃣ Direct match by idPipeline
        # -------------------------
        if pipe_id in base_lookup:
            print("Exact match for pipeline", pipe_id)
            base_row = case_study_graph.loc[
                case_study_graph["idPipeline"] == pipe_id
            ].iloc[0]

            # Copy all columns except geometry
            for col in case_study_graph.columns:
                if col != "geometry":
                    G.edges[u, v][col] = base_row[col]

        # -------------------------
        # 2️⃣ No match → sample from same edge_type
        # -------------------------
        else:
            print("Sampled random pipeline from base graph for", pipe_id)

            # Drop rows without idPipeline (optional but cleaner)
            valid_rows = case_study_graph.dropna(subset=["idPipeline"])

            if not valid_rows.empty:
                sampled_row = valid_rows.sample(1).iloc[0]

                for col in case_study_graph.columns:
                    if col not in ["geometry", 'pipName', 'idPipeline', 'mapLabel', 
                                   'belongs_to', 'curOperNam', 'curPhase', 'curPhDate', 
                                   'fromFacili', 'toFacility', 'mainGrp', 'dimension',
                                     'WaterDepth', 'medium', 'idBelongTo', 'idFrmFacil',
                                       'idToFacili', 'idOperator', 'dtUpdated', 'FactUrl',
                                         'MapUrl', 'weight']:
                        G.edges[u, v][col] = sampled_row[col]
            else:
                print("Warning: No pipelines available in base graph")

    return G


# ==============================
# 8. EXPORT
# ==============================
def export_geojson(G, crs, output_path):
    nodes_records = []
    for node, data in G.nodes(data=True):
        rec = data.copy()
        rec["geometry"] = Point(node)
        nodes_records.append(rec)

    nodes_gdf = gpd.GeoDataFrame(nodes_records, geometry="geometry", crs=crs)

    edge_records = []
    for u, v, data in G.edges(data=True):
        rec = data.copy()
        rec["geometry"] = data["geometry"]
        rec["from_node"] = u
        rec["to_node"] = v
        edge_records.append(rec)
        print(
            rec.get("idPipeline"),
            rec.get("fromFacili"),
            rec.get("toFacility")
        )

    edges_gdf = gpd.GeoDataFrame(edge_records, geometry="geometry", crs=crs)

    nodes_gdf["type"] = "node"
    edges_gdf["type"] = "edge"

    combined = gpd.GeoDataFrame(
        pd.concat([nodes_gdf, edges_gdf], ignore_index=True),
        crs=crs
    )

    combined.to_file(output_path, driver="GeoJSON")
    print("Exported:", output_path)


# ==============================
# 9. VISUALIZATION
# ==============================
def plot_graph(G):
    plt.figure(figsize=(10, 10))
    pos = {node: node for node in G.nodes()}

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=10)

    # Draw curved edges
    nx.draw_networkx_edges(
        G,
        pos,
        width=0.8,
        connectionstyle="arc3,rad=0.2"   # <-- controls curvature
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
        connectionstyle="arc3,rad=0.2"   # must match edge curvature
    )

    plt.title("NetworkX Graph View with Curved idPipeline")
    plt.axis("off")
    plt.show()

def get_node_attribute_columns(G):
    cols = set()
    for _, data in G.nodes(data=True):
        cols.update(data.keys())
    return sorted(cols)

def get_edge_attribute_columns(G):
    cols = set()
    for _,_, data in G.edges(data=True):
        cols.update(data.keys())
    return sorted(cols)

# ==============================
# MAIN
# ==============================
def main():
    gdf = load_and_filter_data(FILE_PATH)
    gdf = clean_geometries(gdf)

    lines = flatten_lines(gdf)
    hubs = cluster_endpoints(lines)
    connected = snap_pipelines(lines, hubs)

    G = build_graph(connected)
    G = add_node_features(G)
    G = add_node_types(G)
    G = add_features_from_base_graph(G)
    G =add_edge_features_from_base_graph(G)
    print("Node columns",get_node_attribute_columns(G))
    print("Edge columns",get_edge_attribute_columns(G))


    output_path = FOLDER + "bigger_network.geojson"
    export_geojson(G, gdf.crs, output_path)

    plot_graph(G)


# ==============================
# ENTRY POINT
# ==============================
if __name__ == "__main__":
    main()