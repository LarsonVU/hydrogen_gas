import geopandas as gpd
import folium
from folium.plugins import PolyLineTextPath
import math
from folium.features import DivIcon
import numpy as np
import numbers
import pandas as pd

# -------------------------------
# Config
# -------------------------------

FOLDER = "data/data_analysis_results/Geojson_pipelines/"
input_path = FOLDER + f"study_case_network.geojson"
output_html = "data/data_analysis_results/graphed_pipeline_network.html"

# Smaller network
# input_path = FOLDER + f"smaller_network.geojson"
# output_html = "data/data_analysis_results/graphed_smaller_network.html"

# Bigger Network
input_path = FOLDER + f"bigger_network.geojson"
output_html = "data/data_analysis_results/graphed_bigger_network.html"

# -------------------------------
# Load data
# -------------------------------
gdf = gpd.read_file(input_path)

# Split nodes and edges
nodes = gdf[gdf.geometry.type == "Point"].copy()
edges = gdf[gdf.geometry.type.isin(["LineString", "MultiLineString"])].copy()

# -------------------------------
# Create base map
# -------------------------------
# Center map on data
center = gdf.geometry.union_all().centroid
m = folium.Map(location=[center.y, center.x], zoom_start=7, tiles="CartoDB positron")

# -------------------------------
# Helper: tooltip builder
# -------------------------------
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

# -------------------------------
# Add edges with arrows
# -------------------------------
def calculate_bearing(start, end):
    lat1 = math.radians(start[0])
    lon1 = math.radians(start[1])
    lat2 = math.radians(end[0])
    lon2 = math.radians(end[1])

    dlon = lon2 - lon1

    x = math.sin(dlon) * math.cos(lat2)
    y = (
        math.cos(lat1) * math.sin(lat2)
        - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    )

    initial_bearing = math.atan2(x, y)
    bearing = (math.degrees(initial_bearing) + 360) % 360 -90
    return bearing

for _, row in edges.iterrows():
    geom = row.geometry

    # Handle MultiLineString
    lines = geom.geoms if geom.geom_type == "MultiLineString" else [geom]

    for line in lines:
        coords = [(lat, lon) for lon, lat in line.coords]

        tooltip = make_tooltip(row)

        poly = folium.PolyLine(
            coords,
            color="black",
            weight=3,
            opacity=0.8,
            tooltip=folium.Tooltip(tooltip, sticky=True),
        ).add_to(m)

# -------------------------------
# Add nodes
# -------------------------------
for _, row in nodes.iterrows():
    point = row.geometry
    tooltip = make_tooltip(row)
    
    # Color mapping for node types (pastel colors)
    color_map = {
        "Generation": "#FFB3BA",      # pastel red
        "Processing": "#BAFFC9",         # pastel green
        "Market": "#BAE1FF",          # pastel blue
        "Compression": "#FFFFBA",      # pastel yellow
        "Junction": "#E0BBE4",        # pastel purple
    }
    
    node_type = row.get("node_type", "junction")
    color = color_map.get(node_type, "#CCCCCC")  # default to pastel grey

    folium.CircleMarker(
        location=[point.y, point.x],
        radius=5,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.9,
        tooltip=folium.Tooltip(tooltip, sticky=True),
    ).add_to(m)

# Add legend
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

# -------------------------------
# Save map
# -------------------------------
m.save(output_html)
print(f"Saved interactive map to: {output_html}")