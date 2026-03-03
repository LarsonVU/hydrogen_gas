import geopandas as gpd
import networkx as nx
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point
from shapely import set_precision
import pandas as pd
from collections import Counter
import math


# ==============================
# SETTINGS
# ==============================
NODE_MERGE_THRESHOLD = 20  # Distance (km) threshold for merging endpoints into hubs (in the same units as the CRS)
FOLDER = "data/data_analysis_results/Geojson_pipelines/"
file_path = FOLDER + "pipLine.geojson"

# ==============================
# 1. LOAD DATA
# ==============================
gdf = gpd.read_file(file_path)

print("Loaded features:", len(gdf))
print("Geometry types:", gdf.geom_type.unique())

# ==============================
# 2. FILTER AND EDIT PIPELINES
# ==============================
#gdf = gdf[gdf["curOperNam"].isin(["Gassco AS"])]
gdf = gdf[gdf["medium"] == "Gas"]
gdf = gdf[gdf["curPhase"] == "IN SERVICE"]

pipelines_to_reverse = [323212, 327394, 406790]  # Example pipeline IDs that need to be reversed

def reverse_linestring(geom):
    if isinstance(geom, LineString):
        return LineString(list(geom.coords)[::-1])
    return geom  # leave other geometries untouched

# Apply flip only to selected pipelines
gdf.loc[gdf["idPipeline"].isin(pipelines_to_reverse), "geometry"] = \
    gdf.loc[gdf["idPipeline"].isin(pipelines_to_reverse), "geometry"].apply(reverse_linestring)

# Remove pipelines by ID
pipelines_to_remove = [442969, 321818]  # Add pipeline IDs to remove here
gdf = gdf[~gdf["idPipeline"].isin(pipelines_to_remove)]

print("Filtered features:", len(gdf))

# ==============================
# 3. CONVERT POLYGONS → BOUNDARIES
# ==============================
if gdf.geom_type.isin(["Polygon", "MultiPolygon"]).any():
    print("Converting polygons to boundary lines...")
    gdf["geometry"] = gdf.geometry.boundary

# Precision cleanup
gdf["geometry"] = gdf.geometry.apply(lambda geom: set_precision(geom, 1e-8))

# ==============================
# 4. FLATTEN GEOMETRIES + KEEP ATTRIBUTES
# ==============================
pipeline_records = []

for _, row in gdf.iterrows():
    geom = row.geometry
    if geom is None or geom.is_empty:
        continue
    props = row.drop("geometry").to_dict()

    if geom.geom_type == "LineString":
        pipeline_records.append({"geometry": geom, "properties": props})

    elif geom.geom_type == "MultiLineString":
        for part in geom.geoms:
            pipeline_records.append({"geometry": part, "properties": props})

lines = pipeline_records
print("Total pipelines after flattening:", len(lines))
# ==============================
# 5. COLLECT ENDPOINTS
# ==============================
endpoints = []

for record in lines:
    geom = record["geometry"]
    coords = list(geom.coords)
    endpoints.append(Point(coords[0]))
    endpoints.append(Point(coords[-1]))

print("Total endpoints:", len(endpoints))

# ==============================
# 6. CLUSTER ENDPOINTS INTO HUBS
# ==============================

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

    avg_x = sum(pt.x for pt in cluster) / len(cluster)
    avg_y = sum(pt.y for pt in cluster) / len(cluster)
    unique_nodes.append(Point(avg_x, avg_y))

print("Detected hubs:", len(unique_nodes))

# ==============================
# 7. SNAP PIPELINES TO HUBS
# ==============================
def find_closest_node(point, nodes):
    closest_node = None
    min_dist = float("inf")
    for node in nodes:
        dist = point.distance(node)
        if dist < min_dist:
            min_dist = dist
            closest_node = node
    return closest_node

connected_lines = []

for record in lines:
    geom = record["geometry"]
    props = record["properties"]
    
    coords = list(geom.coords)
    start_pt = Point(coords[0])
    end_pt = Point(coords[-1])

    closest_start = find_closest_node(start_pt, unique_nodes)
    closest_end = find_closest_node(end_pt, unique_nodes)

    if closest_start and closest_end:
        new_line = LineString([closest_start, closest_end])

        connected_lines.append({
            "geometry": new_line,
            "original_length": geom.length,
            "properties": props
        })
        

# Remove self-loops
connected_lines = [
    rec for rec in connected_lines
    if rec["geometry"].coords[0] != rec["geometry"].coords[-1]
]

print("Connected pipelines:", len(connected_lines))

# ==============================
# 8. BUILD GRAPH (MULTIGRAPH)
# ==============================
G = nx.MultiDiGraph()

for rec in connected_lines:
    geom = rec["geometry"]
    props = rec["properties"]
    original_length = rec["original_length"]

    coords = list(geom.coords)
    start = coords[0]
    end = coords[-1]

    G.add_node(start)
    G.add_node(end)

    edge_attributes = props.copy()
    edge_attributes.update({
        "geometry": geom,
        "weight": original_length
    })

    edge_key = edge_attributes.get("idPipeline", None)

    G.add_edge(start, end, key=edge_key, **edge_attributes)


print("Graph built successfully")
print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())


def majority_feature(values):
    """
    Return the most common value in a list of values.
    If tie, returns one of them arbitrarily.
    """
    if not values:
        return None
    counter = Counter(values)
    return counter.most_common(1)[0][0]

# Add location attribute to nodes based on connected edges
node_features = {}

for node in G.nodes():
    # Collect all edges connected to the node
    inlet_edges = list(G.in_edges(node, data=True))
    outlet_edges = list(G.out_edges(node, data=True))

    if not inlet_edges and not outlet_edges:
        continue
    
    inlet_feature_loc = [edata.get("toFacility") for u, v, edata in inlet_edges if "toFacility" in edata]
    inlet_feature_loc_id = [edata.get("idToFacili") for u, v, edata in inlet_edges if "idFrmFacil" in edata]
    outlet_feature_loc = [edata.get("fromFacili") for u, v, edata in outlet_edges if "fromFacili" in edata]
    outlet_feature_loc_id = [edata.get("idFrmFacil") for u, v, edata in outlet_edges if "idToFacili" in edata]

    node_features.setdefault(node, {})["location"] = majority_feature(inlet_feature_loc + outlet_feature_loc)
    node_features.setdefault(node, {})["location_id"] = majority_feature(inlet_feature_loc_id + outlet_feature_loc_id)

for node, features in node_features.items():
    for key, value in features.items():
        G.nodes[node][key] = value
        
        # If it's a location_id, check for duplicates and append a letter if needed
        if key == "location_id" and value is not None:
            existing_ids = [G.nodes[n].get("location_id") for n in G.nodes() if G.nodes[n].get("location_id")]
            id_count = sum(1 for eid in existing_ids if str(eid).startswith(str(value)))
            
            if id_count > 1:
                # Find the next available letter
                used_letters = set()
                for eid in existing_ids:
                    if str(eid).startswith(str(value)):
                        suffix = str(eid)[len(str(value)):]
                        if suffix and suffix[0].isalpha():
                            used_letters.add(suffix[0])
                
                for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                    if letter not in used_letters:
                        G.nodes[node][key] = f"{value}{letter}"
                        break
        
        print(f"Set node {node} feature {key} to {G.nodes[node][key]}")

# ==============================
# 9. EXPORT NODES & EDGES
# ==============================
nodes_records = []
for node, data in G.nodes(data=True):
    record = data.copy()
    record["geometry"] = Point(node)
    nodes_records.append(record)

nodes_gdf = gpd.GeoDataFrame(
    nodes_records, geometry="geometry",
    crs=gdf.crs
)

edge_records = []
for u, v, data in G.edges(data=True):
    record = data.copy()
    record["geometry"] = data["geometry"]
    record["from_node"] = u
    record["to_node"] = v
    edge_records.append(record)

edges_gdf = gpd.GeoDataFrame(edge_records, crs=gdf.crs, geometry="geometry")

# Combined export
nodes_gdf["type"] = "node"
edges_gdf["type"] = "edge"

combined_gdf = gpd.GeoDataFrame(
    pd.concat([nodes_gdf, edges_gdf], ignore_index=True),
    crs=gdf.crs
)

output_path = FOLDER + f"graphed_pipeline_network_threshold_{NODE_MERGE_THRESHOLD}.geojson"
combined_gdf.to_file(output_path, driver="GeoJSON")

print("Exported:", output_path)

# ==============================
# 11. NETWORKX VIEW
# ==============================
plt.figure(figsize=(10, 10))
pos = {node: node for node in G.nodes()}

nx.draw(G, pos, node_size=10, with_labels=False, width=0.8)

plt.title("NetworkX Graph View")
plt.show()