import pandas as pd
import folium
from pathlib import Path
import geopandas as gpd

FOLDER = "data/data_analysis_results/Geojson_pipelines/"
NETWORK_FILE = FOLDER + "graphed_pipeline_network_with_features.geojson"

# -------------------------------------------------
# 1. Define hydrogen production / hub locations
#    (approximate coordinates, for visualization)
# -------------------------------------------------
sites = [
    {
        "name": "Kårstø (CHE)",
        "lat": 59.2356,
        "lon": 5.5217,
        "notes": "Cancelled project of Equinor",
        "color": "lightred"
    },
    {
        "name": "Nyhamna (Aukra Hydrogen Hub)",
        "lat": 62.830755,
        "lon": 6.916013,
        "notes": "Cancelled hydrogen hub near Nyhamna gas terminal.",
        "color": "lightred"
    },
    {
        "name": "Langstranda – Bodø (GreenH)",
        "lat": 67.2709,
        "lon": 14.3449,
        "notes": "Planned GreenH hydrogen production project at Langstranda, Bodø.",
        "color": "lightgreen"
    },
    {
        "name": "Vestbase – Kristiansund (GreenH)",
        "lat": 63.1105,
        "lon": 7.7280,
        "notes": "GreenH hydrogen project located at the Vestbase industrial area in Kristiansund.",
        "color": "lightgreen"
    },
    {
        "name": "Slagentangen – Tønsberg (GreenH)",
        "lat": 59.2675,
        "lon": 10.4076,
        "notes": "GreenH hydrogen project at the Slagentangen terminal near Tønsberg.",
        "color": "lightgreen"
    },
    {
        "name": "Hammerfest (GreenH)",
        "lat": 70.6634,
        "lon": 23.6821,
        "notes": "Planned GreenH hydrogen hub in Hammerfest.",
        "color": "lightgreen"
    },
    {
        "name": "Årdalsfjorden – Fiskå industriområde (GreenH)",
        "lat": 59.4330,
        "lon": 5.4330,
        "notes": "GreenH hydrogen project at the Fiskå industrial area along Årdalsfjorden, Rogaland.",
        "color": "lightgreen"
    },
    {
        "name": "Sandnessjøen – Horvnes (GreenH)",
        "lat": 66.0217,
        "lon": 12.6316,
        "notes": "GreenH hydrogen project at the Horvnes industrial area near Sandnessjøen.",
        "color": "lightgreen"
    },
    {
        "name": "HyFuel – Florø (HydrogenSolutions)",
        "lat": 61.5996,
        "lon": 5.0328,
        "notes": "HyFuel 20 MW green hydrogen production project at Fjord Base in Florø, Kinn municipality (Hydrogen Solutions AS ongoing project).",
        "color": "lightblue"
    },
    {
        "name": "Kaupanes Expansion – Egersund (HydrogenSolutions)",
        "lat": 58.4514,
        "lon": 5.9998,
        "notes": "Kaupanes Expansion Project — expansion of the Kaupanes hydrogen plant in the industrial area of Egersund (Hydrogen Solutions AS ongoing project).",
        "color": "lightblue"
    },
    {
        "name": "Meråker Hydrogen – Kopperå (HydrogenSolutions)",
        "lat": 63.3923,
        "lon": 11.8488,
        "notes": "Meråker Hydrogen plant to be built at Kopperå in Meråker Municipality, producing green hydrogen from renewable energy (Hydrogen Solutions AS ongoing project).",
        "color": "lightblue"
    }


]

df = pd.DataFrame(sites)


gdf = gpd.read_file(NETWORK_FILE)

processing_nodes = gdf[(gdf.geometry.type == "Point") & (gdf.node_type == "Processing")].copy()
processing_node_locations = pd.DataFrame([{"name": row["location"],
                            "lon": row.geometry.x,
                            "lat": row.geometry.y,
                            "color": "lightgreen"
                            } for idx, row in processing_nodes.iterrows()])



# -------------------------------------------------
# 2. Create base map centered on Norway
# -------------------------------------------------
map_center_lat = 62.0
map_center_lon = 8.0

m = folium.Map(
    location=[map_center_lat, map_center_lon],
    zoom_start=5,
    control_scale=True,
    tiles="CartoDB Positron No Labels"
)

# Optional alternative base layers
#folium.TileLayer("OpenStreetMap").add_to(m)
folium.TileLayer("CartoDB Positron No Labels").add_to(m)

# # -------------------------------------------------
# # 3. Add processing nodes 
# # -------------------------------------------------
# for _, row in processing_node_locations.iterrows():
#     folium.CircleMarker(
#         location=[row["lat"], row["lon"]],
#         radius=4,  # size of the dot
#         popup=f"<b>{row['name']}</b><br>{row.get('notes','')}",
#         tooltip=row["name"],
#         color=row["color"],        # border color
#         fill=True,
#         fill_color=row["color"],
#         fill_opacity=0.9
#     ).add_to(m)

# -------------------------------------------------
# 4. Add hydrogen site markers
# -------------------------------------------------

for _, row in df.iterrows():
    folium.Marker(
        location=[row["lat"], row["lon"]],
        popup=f"<b>{row['name']}</b><br>{row['notes']}",
        tooltip=row["name"],
        icon=folium.Icon(
            color=row["color"],
            icon="flask",
            prefix="fa"   # FontAwesome icon
        )
    ).add_to(m)

# -------------------------------------------------
# 5. Add layer control
# -------------------------------------------------
folium.LayerControl().add_to(m)

# -------------------------------------------------
# 4. Add html legend
# -------------------------------------------------
legend_html = """
<div style="
    position: fixed;
    bottom: 50px;
    left: 50px;
    width: 220px;
    z-index: 9999;
    background-color: white;
    border: 2px solid grey;
    border-radius: 5px;
    padding: 10px;
    font-size: 14px;
">
<b>Hydrogen Projects</b><br><br>
<i class="fa fa-map-marker fa-2x" style="color:#FFB3BA"></i>&nbsp; On hold / Cancelled<br>
<i class="fa fa-map-marker fa-2x" style="color:lightblue"></i>&nbsp; HYDS projects<br>
<i class="fa fa-map-marker fa-2x" style="color:#BAFFC9"></i>&nbsp; GreenH projects
</div>
"""

m.get_root().html.add_child(folium.Element(legend_html))

# -------------------------------------------------
# 5. Save map to file
# -------------------------------------------------
output_path = Path("data/data_analysis_results/h2_norway_overlay_map.html")
m.save(output_path)

print(f"Map saved to: {output_path.resolve()}")
