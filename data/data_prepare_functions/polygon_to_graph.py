import geopandas as gpd
import networkx as nx
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point
from shapely.ops import unary_union, split
from shapely import set_precision
import itertools

# --------------------------------------------------
# 1. Load GeoJSON
# --------------------------------------------------
file_path = "data/Geojson_pipelines/pipLine.geojson"
gdf = gpd.read_file(file_path)

print("Loaded features:", len(gdf))
print("Geometry type:", gdf.geom_type.unique())

# --------------------------------------------------
# 2. Convert Polygons → Boundary Lines (if needed)
# --------------------------------------------------
if gdf.geom_type.iloc[0] in ["Polygon", "MultiPolygon"]:
    print("Converting polygons to boundary lines...")
    gdf["geometry"] = gdf.geometry.boundary

# Clean precision to avoid tiny topology errors
gdf["geometry"] = gdf.geometry.apply(lambda geom: set_precision(geom, 1e-8))

# --------------------------------------------------
# 3. Merge Lines (topological cleanup)
# --------------------------------------------------
merged = unary_union(gdf.geometry)

if merged.geom_type == "MultiLineString":
    lines = list(merged.geoms)
elif merged.geom_type == "LineString":
    lines = [merged]
else:
    raise ValueError("Unexpected geometry type after merge.")

print("Total lines after merge:", len(lines))

# --------------------------------------------------
# 4. Detect Hubs
# --------------------------------------------------
proximity_threshold = 0.3  # hyperparameter: distance threshold for considering lines "close"

# First, collect all endpoints from all lines
endpoints = []
for line in lines:
    coords = list(line.coords)
    endpoints.append(Point(coords[0]))   # start point
    endpoints.append(Point(coords[-1]))  # end point

# Cluster nearby endpoints and merge them
unique_nodes = []
used = set()

for i, pt1 in enumerate(endpoints):
    if i in used:
        continue
    
    # Find all endpoints close to pt1
    cluster = [pt1]
    used.add(i)
    
    for j, pt2 in enumerate(endpoints):
        if j <= i or j in used:
            continue
        if pt1.distance(pt2) < proximity_threshold:
            cluster.append(pt2)
            used.add(j)
    
    # Merge by taking average position
    avg_x = sum(pt.x for pt in cluster) / len(cluster)
    avg_y = sum(pt.y for pt in cluster) / len(cluster)
    merged_point = Point(avg_x, avg_y)
    unique_nodes.append(merged_point)

print("Detected intersections:", len(unique_nodes))

# --------------------------------------------------
# 5. Merge lines at hubs
# --------------------------------------------------

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
for line in lines:
    coords = list(line.coords)
    start_pt = Point(coords[0])
    end_pt = Point(coords[-1])
    
    closest_start = find_closest_node(start_pt, unique_nodes)
    closest_end = find_closest_node(end_pt, unique_nodes)
    
    if closest_start is not None and closest_end is not None:
        new_line = LineString([closest_start, closest_end])
        connected_lines.append(new_line)

# ------------------------------------------------
# Remove self loops
# ------------------------------------------------
connected_lines = [line for line in connected_lines if line.coords[0] != line.coords[-1]]


# --------------------------------------------------
# 6. Build Network Graph
# --------------------------------------------------
G = nx.Graph()

for line in connected_lines:
    coords = list(line.coords)
    start = coords[0]
    end = coords[-1]
    length = line.length

    G.add_node(start)
    G.add_node(end)
    G.add_edge(start, end, geometry=line, weight=length)

print("Graph built successfully")
print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())

# --------------------------------------------------
# 7. Export Nodes & Edges (optional)
# --------------------------------------------------
nodes_gdf = gpd.GeoDataFrame(
    geometry=[Point(n) for n in G.nodes()],
    crs=gdf.crs
)
edges_gdf = gpd.GeoDataFrame(
    geometry=[G.edges[e]["geometry"] for e in G.edges()],
    crs=gdf.crs
)

nodes_gdf.to_file("nodes.geojson", driver="GeoJSON")
edges_gdf.to_file("edges.geojson", driver="GeoJSON")

print("Exported nodes.geojson and edges.geojson")

# --------------------------------------------------
# 8. Visualization
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 10))

# Plot edges
edges_gdf.plot(ax=ax, linewidth=1)

# Plot nodes
nodes_gdf.plot(ax=ax, markersize=10)

plt.title("Pipeline Network (Edges & Nodes)")
plt.axis("equal")
plt.show()

# --------------------------------------------------
# 9. NetworkX Graph Visualization
# --------------------------------------------------

plt.figure(figsize=(10, 10))

pos = {node: node for node in G.nodes()}  # spatial coordinates

nx.draw(
    G,
    pos,
    node_size=10,
    with_labels=False,
    width=0.8
)

plt.title("NetworkX Graph View")
plt.show()
