import geopandas as gpd
import networkx as nx
import pandas as pd

DATA_FOLDER = "data/"
GDF_FILE = DATA_FOLDER + "data_analysis_results/Geojson_pipelines/graphed_pipeline_network_threshold_20.geojson"
NODE_INFO_FILE = DATA_FOLDER + "data_sources/node_info.xlsx"
GENERATION_NODES_FILE = DATA_FOLDER + "data_sources/generation_fields.xlsx"

OUTPUT_FILE = DATA_FOLDER + "data_analysis_results/Geojson_pipelines/graphed_pipeline_network_with_features.geojson"



# ==============================
# Units
# flow in million standard cubic meters per day (mmscmd)
# Costs in Thousand NOK
# distance in kilometers (km)



def read_geojson(file_path):
    gdf = gpd.read_file(file_path)
    return gdf


def add_max_flow(gdf):
    for idx, row in gdf.iterrows():
        if row["type"] == "node":
            gdf.at[idx, "max_flow"] = 100000  # Nodes have no meaningful flow capacity
    return gdf

def add_node_types(gdf, node_info):
    gdf["location_id"] = gdf["location_id"].astype(str)
    node_info["Location ID"] = node_info["Location ID"].astype(str)
    for idx, row in gdf.iterrows():
        if row["type"] == "node":
            node_id = row["location_id"]
            node_information = node_info[node_info["Location ID"] == str(node_id)]
            if not node_information.empty:
                gdf.at[idx, "node_type"] = node_information.iloc[0]["Node Type"]
            else:
                gdf.at[idx, "node_type"] = "Unknown"
            print(f"Processed node {node_id} with type {gdf.at[idx, 'node_type']}")
    return gdf

def add_supplier(gdf, supplier_data):
    gdf["supplier"] = None  # ensures object dtype
    gdf.loc[gdf["location"] == "AASTA HANSTEEN PLEM", "supplier"] = supplier_data[supplier_data["Field Name"] == "AASTA HANSTEEN"]["Operatør"].iloc[0]
    gdf.loc[gdf["location"] == "GJØA", "supplier"] = supplier_data[supplier_data["Field Name"] == "GJØA"]["Operatør"].iloc[0]
    gdf.loc[gdf["location"] == "NORNE ERB", "supplier"] = supplier_data[supplier_data["Field Name"] == "NORNE"]["Operatør"].iloc[0]
    gdf.loc[gdf["location"] == "OSEBERG D", "supplier"] = supplier_data[supplier_data["Field Name"] == "OSEBERG"]["Operatør"].iloc[0]
    gdf.loc[gdf["location"] == "VISUND", "supplier"] = supplier_data[supplier_data["Field Name"] == "VISUND"]["Operatør"].iloc[0]
    return gdf

def add_generation_capacity(gdf):
    gdf["generation_capacity"] = None  # ensures object dtype
    gdf.loc[gdf["location"] == "AASTA HANSTEEN PLEM", "generation_capacity"] =  8.92 /365 *1000
    gdf.loc[gdf["location"] == "GJØA", "generation_capacity"] = 3.4 /365 *1000
    gdf.loc[gdf["location"] == "NORNE ERB", "generation_capacity"] = 0.1 /365 *1000
    gdf.loc[gdf["location"] == "OSEBERG D", "generation_capacity"] = 8.37 /365 *1000
    gdf.loc[gdf["location"] == "VISUND", "generation_capacity"] = 5.76 /365 *1000
    return gdf

def add_generation_cost(gdf):
    gdf["generation_cost"] = None  # ensures object dtype
    gdf.loc[gdf["location"] == "AASTA HANSTEEN PLEM", "generation_cost"] =  122.85 / 11.28 # NOK to EUR Ex rate 23-2-2026
    gdf.loc[gdf["location"] == "GJØA", "generation_cost"] = 194.50 / 11.28
    gdf.loc[gdf["location"] == "NORNE ERB", "generation_cost"] = 384.11 / 11.28
    gdf.loc[gdf["location"] == "OSEBERG D", "generation_cost"] = 476.97 / 11.28
    gdf.loc[gdf["location"] == "VISUND", "generation_cost"] = 295.56  / 11.28
    return gdf

def add_gas_composition(gdf):
    gdf["gas_composition"] = None  # ensures object dtype
    gdf.loc[gdf["location"] == "AASTA HANSTEEN PLEM"]["gas_composition"] = [ {"CO2": 0.02, "H2": 0.00, "NG": 0.97}]
    gdf.loc[gdf["location"] == "GJØA", "gas_composition"] = [{"CO2": 0.02, "H2": 0.00, "NG": 0.98}]
    gdf.loc[gdf["location"] == "NORNE ERB", "gas_composition"] = [ {"CO2": 0.005, "H2": 0.00, "NG": 0.99}]
    gdf.loc[gdf["location"] == "OSEBERG D", "gas_composition"] = [{"CO2": 0.01, "H2": 0.00, "NG": 0.98}]
    gdf.loc[gdf["location"] == "VISUND", "gas_composition"] =[ {"CO2": 0.03, "H2": 0.00, "NG": 0.97}]
    return gdf


def add_generation_parameters(gdf):
    gdf = add_max_flow(gdf)
    gdf = add_node_types(gdf, node_info)
    gdf = add_generation_capacity(gdf)
    gdf = add_generation_cost(gdf)
    gdf = add_gas_composition(gdf)
    return gdf

def add_compression_factor(gdf):
    gdf["compression_factor"] = None  # ensures object dtype
    gdf.loc[gdf["location"] == "B-11", "compression_factor"] =  1.2
    gdf.loc[gdf["location"] == "EUROPIPE-SCP", "compression_factor"] = 1.1
    gdf.loc[gdf["location"] == "NORPIPE Y", "compression_factor"] = 1.3
    gdf.loc[gdf["location"] == "ZEEPIPE-SCP", "compression_factor"] = 1.25
    return gdf

def add_compression_constants(gdf):
    gdf["compression_constants"] = None  # ensures object dtype
    compression_nodes = ["B-11", "EUROPIPE-SCP", "NORPIPE Y", "ZEEPIPE-SCP"]
    normal_inlet_pressure = 40  # bar
    normal_flow  = 8
    for node in compression_nodes:
        normal_discharge_pressure = normal_inlet_pressure * gdf.loc[gdf["location"] == node, "compression_factor"].iloc[0]  # bar
        inlet_constant= (normal_inlet_pressure * normal_flow) / ((2/3 * normal_inlet_pressure + 1/3 * normal_discharge_pressure) ** 2)
        outlet_constant = (normal_discharge_pressure * normal_flow) / ((2/3 * normal_inlet_pressure + 1/3 * normal_discharge_pressure) ** 2)
        flow_constant = (normal_discharge_pressure- normal_inlet_pressure ) / (2/3 * normal_inlet_pressure + 1/3 * normal_discharge_pressure)
        gdf.loc[gdf["location"] == node, "compression_constants"] = [{"NG": {"K_into_pipe": inlet_constant, "K_out_pipe": outlet_constant, "K_flow": flow_constant},
                                                                      "CO2": {"K_into_pipe": inlet_constant, "K_out_pipe": outlet_constant, "K_flow": flow_constant},
                                                                      "H2": {"K_into_pipe": inlet_constant, "K_out_pipe": outlet_constant, "K_flow": flow_constant}}]
    return gdf

def add_compression_node_parameters(gdf):
    gdf = add_compression_factor(gdf)
    gdf = add_compression_constants(gdf)
    return gdf

def add_market_prices(gdf):
    gdf["average_market_price"] = None  # ensures object dtype
    gdf.loc[gdf["location"] == "DUNKERQUE", "average_market_price"] = 30
    gdf.loc[gdf["location"] == "EASINGTON", "average_market_price"] = 26
    gdf.loc[gdf["location"] == "EMDEN", "average_market_price"] = 28
    gdf.loc[gdf["location"] == "DORNUM", "average_market_price"] = 28
    gdf.loc[gdf["location"] == "ST. FERGUS", "average_market_price"] = 26
    gdf.loc[gdf["location"] == "ZEEBRUGGE", "average_market_price"] = 29
    return gdf

def add_long_term_std(gdf):
    gdf["long_term_price_std"] = None  # ensures object dtype
    gdf.loc[gdf["location"] == "DUNKERQUE", "long_term_price_std"] = 0.10
    gdf.loc[gdf["location"] == "EASINGTON", "long_term_price_std"] = 0.15
    gdf.loc[gdf["location"] == "EMDEN", "long_term_price_std"] = 0.10
    gdf.loc[gdf["location"] == "DORNUM", "long_term_price_std"] = 0.10
    gdf.loc[gdf["location"] == "ST. FERGUS", "long_term_price_std"] = 0.15
    gdf.loc[gdf["location"] == "ZEEBRUGGE", "long_term_price_std"] = 0.10
    return gdf

def add_day_ahead_std(gdf):
    gdf["day_ahead_price_std"] = None  # ensures object dtype
    gdf.loc[gdf["location"] == "DUNKERQUE", "day_ahead_price_std"] = 0.01
    gdf.loc[gdf["location"] == "EASINGTON", "day_ahead_price_std"] = 0.01
    gdf.loc[gdf["location"] == "EMDEN", "day_ahead_price_std"] = 0.01
    gdf.loc[gdf["location"] == "DORNUM", "day_ahead_price_std"] = 0.01
    gdf.loc[gdf["location"] == "ST. FERGUS", "day_ahead_price_std"] = 0.01
    gdf.loc[gdf["location"] == "ZEEBRUGGE", "day_ahead_price_std"] = 0.01
    return gdf

def add_average_demand(gdf):
    # Average Demand in Mwh x 1000
    gdf["average_demand_mwh_x1000"] = None
    gdf.loc[gdf["location"] == "DUNKERQUE", "average_demand_mwh_x1000"] = 50.0
    gdf.loc[gdf["location"] == "EASINGTON", "average_demand_mwh_x1000"] = 12.5
    gdf.loc[gdf["location"] == "ST. FERGUS", "average_demand_mwh_x1000"] = 12.5
    gdf.loc[gdf["location"] == "EMDEN", "average_demand_mwh_x1000"] = 25.0
    gdf.loc[gdf["location"] == "DORNUM", "average_demand_mwh_x1000"] = 25.0
    gdf.loc[gdf["location"] == "ZEEBRUGGE", "average_demand_mwh_x1000"] = 25.0
    return gdf

def add_demand_variance(gdf):
    # Variance expressed as fraction (e.g. 0.3 == 30%)
    gdf["demand_variance"] = None
    for loc in ["DUNKERQUE", "EASINGTON", "ST. FERGUS", "EMDEN", "DORNUM", "ZEEBRUGGE"]:
        gdf.loc[gdf["location"] == loc, "demand_variance"] = 0.3
    return gdf

def add_market_and_demand_parameters(gdf):
    gdf = add_market_prices(gdf)
    gdf = add_long_term_std(gdf)
    gdf = add_day_ahead_std(gdf)
    gdf = add_average_demand(gdf)
    gdf = add_demand_variance(gdf)
    return gdf

def add_arc_capacities(gdf):
    # Known ones
    gdf.loc[gdf["idPipeline"] == 444927.0, "max_flow"] = 57.6  # Polarled
    gdf.loc[gdf["idPipeline"] == 319132.0, "max_flow"] = 11.0  # Norne Gasstransport

    # Add remaining pipes

    gdf.loc[gdf["idPipeline"] == 323042.0, "max_flow"] = 44.4  # Statpipe (1)
    gdf.loc[gdf["idPipeline"] == 323212.0, "max_flow"] = 73.5  # Europipe II
    gdf.loc[gdf["idPipeline"] == 310326.0, "max_flow"] = 42.2  # Zeepipe I (1)
    gdf.loc[gdf["idPipeline"] == 313556.0, "max_flow"] = 21.1  # Statpipe (2)
    gdf.loc[gdf["idPipeline"] == 319166.0, "max_flow"] = 44.4  # Norpipe Gassledning
    gdf.loc[gdf["idPipeline"] == 324844.0, "max_flow"] = 44.4  # Norpipe Gassledning
    gdf.loc[gdf["idPipeline"] == 319676.0, "max_flow"] = 35  # Oseberg Gasstransport
    gdf.loc[gdf["idPipeline"] == 322464.0, "max_flow"] = 30.7  # Statpipe (3)
    gdf.loc[gdf["idPipeline"] == 322532.0, "max_flow"] = 36.9  # Vesterled
    gdf.loc[gdf["idPipeline"] == 321002.0, "max_flow"] = 55  # Zeepipe I (2)
    gdf.loc[gdf["idPipeline"] == 321036.0, "max_flow"] = 42.2  # Zeepipe I (3)
    gdf.loc[gdf["idPipeline"] == 369498.0, "max_flow"] = 73.8  # Langeled (south)
    gdf.loc[gdf["idPipeline"] == 326340.0, "max_flow"] = 45.7  # Europipe I
    gdf.loc[gdf["idPipeline"] == 326374.0, "max_flow"] = 54.2  # Franpipe
    gdf.loc[gdf["idPipeline"] == 326408.0, "max_flow"] = 30.7  # Statpipe (4)
    gdf.loc[gdf["idPipeline"] == 321784.0, "max_flow"] = 44.4  # Norpipe Gassledning
    gdf.loc[gdf["idPipeline"] == 327394.0, "max_flow"] = 45.7  # Europipe I
    gdf.loc[gdf["idPipeline"] == 357492.0, "max_flow"] = 33.5  # Kvitebjørn Gassrør
    gdf.loc[gdf["idPipeline"] == 369461.0, "max_flow"] = 74.4  # Langeled (north)
    gdf.loc[gdf["idPipeline"] == 406790.0, "max_flow"] = 18.2  # Gjøa Gassrør

    # Not provided in map
    gdf.loc[gdf["idPipeline"] == 364645.0, "max_flow"] = 40  # No map label
    gdf.loc[gdf["idPipeline"] == 307674.0, "max_flow"] = 40  # Åsgard Transport
    gdf.loc[gdf["idPipeline"] == 323076.0, "max_flow"] = 40  # Zeepipe II A
    gdf.loc[gdf["idPipeline"] == 323110.0, "max_flow"] = 40  # Zeepipe II B
    return gdf


def add_moap(gdf):
    for idx, row in gdf.iterrows():
        if row["type"] == "edge":
            row["max_inlet_pressure"] = 150
    return gdf

def add_arc_parameters(gdf):
    gdf = add_arc_capacities(gdf)
    gdf = add_moap(gdf)
    return gdf

if __name__ == "__main__":
    gdf = read_geojson(GDF_FILE)
    node_info = pd.read_excel(NODE_INFO_FILE)
    gdf = add_generation_parameters(gdf)
    gdf = add_compression_node_parameters(gdf)
    gdf = add_market_and_demand_parameters(gdf)
    gdf = add_arc_parameters(gdf)

    print(gdf[["idPipeline", "mapLabel", "pipName", "max_flow"]])
    gdf.to_file(OUTPUT_FILE, driver="GeoJSON")