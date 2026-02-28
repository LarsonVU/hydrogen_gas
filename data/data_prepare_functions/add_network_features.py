import geopandas as gpd
import networkx as nx
import pandas as pd
import math
import ast

DATA_FOLDER = "data/"
GDF_FILE = DATA_FOLDER + "data_analysis_results/Geojson_pipelines/graphed_pipeline_network_threshold_20.geojson"
NODE_INFO_FILE = DATA_FOLDER + "data_sources/node_info.xlsx"
GENERATION_NODES_FILE = DATA_FOLDER + "data_sources/generation_fields.xlsx"

OUTPUT_FILE = DATA_FOLDER + "data_analysis_results/Geojson_pipelines/study_case_network.geojson"

# ==============================
# Units
# flow in million standard cubic meters per day (mscmd)
# Costs in Euro
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
    return gdf

def add_booking_cost(gdf, node_info):
    for idx, row in gdf.iterrows():
        if row["type"] == "node":
            node_id = row["location_id"]
            booking_cost = node_info[node_info["Location ID"] == str(node_id)]["Tariff"]
            gdf.at[idx, "base_booking_cost"] = booking_cost.iloc[0] / 11.28 # NOK to Euro 23-2-2026
    return gdf

def add_supplier(gdf, supplier_data):
    gdf["supplier"] = None  # ensures object dtype
    gdf.loc[gdf["location"] == "AASTA HANSTEEN PLEM", "supplier"] = supplier_data.loc[supplier_data["Field name"] == "AASTA HANSTEEN", "Operatør"].iloc[0]
    gdf.loc[gdf["location"] == "GJØA", "supplier"] = supplier_data.loc[supplier_data["Field name"] == "GJØA","Operatør"].iloc[0]
    gdf.loc[gdf["location"] == "NORNE ERB", "supplier"] = supplier_data.loc[supplier_data["Field name"] == "NORNE","Operatør"].iloc[0]
    gdf.loc[gdf["location"] == "OSEBERG D", "supplier"] = supplier_data.loc[supplier_data["Field name"] == "OSEBERG","Operatør"].iloc[0]
    gdf.loc[gdf["location"] == "VISUND", "supplier"] = supplier_data.loc[supplier_data["Field name"] == "VISUND","Operatør"].iloc[0]
    # Hydrogen
    gdf.loc[gdf["location"] == "NYHAMNA", "supplier"] = "SHELL"
    gdf.loc[gdf["location"] == "KÅRSTØ", "supplier"] = "Equinor Energy AS"
    return gdf

def add_supply_capacity(gdf):
    gdf["supply_capacity"] = None  # ensures object dtype
    gdf.loc[gdf["location"] == "AASTA HANSTEEN PLEM", "supply_capacity"] =  8.92 /365 *1000
    gdf.loc[gdf["location"] == "GJØA", "supply_capacity"] = 3.4 /365 *1000
    gdf.loc[gdf["location"] == "NORNE ERB", "supply_capacity"] = 0.1 /365 *1000
    gdf.loc[gdf["location"] == "OSEBERG D", "supply_capacity"] = 8.37 /365 *1000
    gdf.loc[gdf["location"] == "VISUND", "supply_capacity"] = 5.76 /365 *1000
    # Hydrogen
    gdf.loc[gdf["location"] == "NYHAMNA", "supply_capacity"] = 14.47
    gdf.loc[gdf["location"] == "KÅRSTØ", "supply_capacity"] = 14.47
    return gdf



def add_generation_cost(gdf):
    gdf["generation_cost"] = None  # ensures object dtype # Euro per Mscm
    gdf.loc[gdf["location"] == "AASTA HANSTEEN PLEM", "generation_cost"] =  122.85 * 1000  / 11.28 # NOK to EUR Ex rate 23-2-2026
    gdf.loc[gdf["location"] == "GJØA", "generation_cost"] = 194.50 * 1000  / 11.28 
    gdf.loc[gdf["location"] == "NORNE ERB", "generation_cost"] = 384.11 * 1000 / 11.28
    gdf.loc[gdf["location"] == "OSEBERG D", "generation_cost"] = 476.97 * 1000 / 11.28
    gdf.loc[gdf["location"] == "VISUND", "generation_cost"] = 295.56  * 1000 / 11.28
    # Hydrogen
    gdf.loc[gdf["location"] == "NYHAMNA", "generation_cost"] = 195 * 1000
    gdf.loc[gdf["location"] == "KÅRSTØ", "generation_cost"] = 195 * 1000
    return gdf

def add_component_ratio(gdf):
    gdf["component_ratio"] = None  # ensures object dtype
    gdf.loc[gdf["location"] == "AASTA HANSTEEN PLEM", "component_ratio"] = [ {"CO2": 0.02, "H2": 0.00, "NG": 0.98}]
    gdf.loc[gdf["location"] == "GJØA", "component_ratio"] = [{"CO2": 0.02, "H2": 0.00, "NG": 0.98}]
    gdf.loc[gdf["location"] == "NORNE ERB", "component_ratio"] = [ {"CO2": 0.005, "H2": 0.00, "NG": 0.995}]
    gdf.loc[gdf["location"] == "OSEBERG D", "component_ratio"] = [{"CO2": 0.01, "H2": 0.00, "NG": 0.99}]
    gdf.loc[gdf["location"] == "VISUND", "component_ratio"] =[ {"CO2": 0.03, "H2": 0.00, "NG": 0.97}]

    # Hydrogen
    gdf.loc[gdf["location"] == "NYHAMNA", "component_ratio"] = [ {"CO2": 0.00, "H2": 1.00, "NG": 0}]
    gdf.loc[gdf["location"] == "KÅRSTØ", "component_ratio"] = [ {"CO2": 0.00, "H2": 1.00, "NG": 0}]
    return gdf


def add_generation_parameters(gdf, node_info, supplier_data):
    gdf = add_max_flow(gdf)
    gdf = add_node_types(gdf, node_info)
    gdf = add_booking_cost(gdf, node_info)
    gdf = add_supplier(gdf, supplier_data)
    gdf = add_supply_capacity(gdf)
    gdf = add_generation_cost(gdf)
    gdf = add_component_ratio(gdf)
    return gdf

def add_compression_factor(gdf):
    gdf["compression_increase"] = None  # ensures object dtype
    gdf.loc[gdf["location"] == "B-11", "compression_increase"] =  1.9
    gdf.loc[gdf["location"] == "EUROPIPE-SCP", "compression_increase"] = 1.9
    gdf.loc[gdf["location"] == "NORPIPE Y", "compression_increase"] = 1.9
    gdf.loc[gdf["location"] == "ZEEPIPE-SCP", "compression_increase"] = 1.9
    return gdf

def add_compression_constants(gdf):
    gdf["compression_constants"] = None  # ensures object dtype
    compression_nodes = ["B-11", "EUROPIPE-SCP", "NORPIPE Y", "ZEEPIPE-SCP"]
    normal_inlet_pressure = 100# 40  # bar
    normal_flow  = 25# 8
    T = 288.15
    T_std = 288.15
    eta =0.72
    p_std =1.01325
    w_to_hour = 3600
    constant = T * p_std / (T_std * eta * w_to_hour)
    for node in compression_nodes:
        normal_discharge_pressure = normal_inlet_pressure * gdf.loc[gdf["location"] == node, "compression_increase"].iloc[0]  # bar
        inlet_constant= constant * (normal_inlet_pressure * normal_flow) / ((2/3 * normal_inlet_pressure + 1/3 * normal_discharge_pressure) ** 2)
        outlet_constant = constant * (normal_discharge_pressure * normal_flow) / ((2/3 * normal_inlet_pressure + 1/3 * normal_discharge_pressure) ** 2)
        flow_constant = constant * (normal_discharge_pressure- normal_inlet_pressure ) / (2/3 * normal_inlet_pressure + 1/3 * normal_discharge_pressure)
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
    gdf.loc[gdf["location"] == "DUNKERQUE", "average_market_price"] = 30 #  euro / Mwh
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
    # Average Demand in Gwh
    gdf["average_demand_mwh_x1000"] = None
    gdf.loc[gdf["location"] == "DUNKERQUE", "average_demand_mwh_x1000"] = 50.0 #*2 #+ 56.25 + 43.75
    gdf.loc[gdf["location"] == "EASINGTON", "average_demand_mwh_x1000"] = 12.5 #* 4
    gdf.loc[gdf["location"] == "ST. FERGUS", "average_demand_mwh_x1000"] = 12.5 #* 4
    gdf.loc[gdf["location"] == "EMDEN", "average_demand_mwh_x1000"] = 25.0 # *2 + 56.25 
    gdf.loc[gdf["location"] == "DORNUM", "average_demand_mwh_x1000"] = 25.0 #*4 + 56.25 
    gdf.loc[gdf["location"] == "ZEEBRUGGE", "average_demand_mwh_x1000"] = 25.0  # * 4
    return gdf

def add_demand_variance(gdf):
    # Variance expressed as fraction (e.g. 0.3 == 30%)
    gdf["demand_variance"] = None
    for loc in ["DUNKERQUE", "EASINGTON", "ST. FERGUS", "EMDEN", "DORNUM", "ZEEBRUGGE"]:
        gdf.loc[gdf["location"] == loc, "demand_variance"] = 0.3
    return gdf

def add_max_component_percentages(gdf):
    gdf["max_fractions"] = None
    gdf.loc[gdf["location"] == "DUNKERQUE", "max_fractions"] = [{"NG": 1, "CO2": 0.025, "H2": 0.06}]
    gdf.loc[gdf["location"] == "EASINGTON", "max_fractions"] = [{"NG": 1, "CO2": 0.025, "H2": 0.001}]
    gdf.loc[gdf["location"] == "ST. FERGUS", "max_fractions"] = [{"NG": 1, "CO2": 0.04, "H2": 0.001}]
    gdf.loc[gdf["location"] == "EMDEN", "max_fractions"] = [{"NG": 1, "CO2": 0.026, "H2": 0.1}]
    gdf.loc[gdf["location"] == "DORNUM", "max_fractions"] =[ {"NG": 1, "CO2": 0.025, "H2": 0.1}]
    gdf.loc[gdf["location"] == "ZEEBRUGGE", "max_fractions"] = [{"NG": 1, "CO2": 0.025, "H2": 0}]
    return gdf

def add_min_pressure(gdf):
    gdf["min_outlet_pressure"] = None
    # Set minimum outlet pressures (bar)
    gdf.loc[gdf["location"] == "DUNKERQUE", "min_outlet_pressure"] = 60
    gdf.loc[gdf["location"] == "EASINGTON", "min_outlet_pressure"] = 70
    gdf.loc[gdf["location"] == "ST. FERGUS", "min_outlet_pressure"] = 41
    gdf.loc[gdf["location"] == "EMDEN", "min_outlet_pressure"] = 45
    gdf.loc[gdf["location"] == "DORNUM", "min_outlet_pressure"] = 84
    gdf.loc[gdf["location"] == "ZEEBRUGGE", "min_outlet_pressure"] = 80
    return gdf

def add_supplier_ratios(gdf):
    gdf["supplier_ratios"] = None
    gdf.loc[gdf["location"] == "DUNKERQUE", "supplier_ratios"] = [{"Equinor Energy AS": 1.0, "SHELL": 0.0, "Vår Energi ASA": 0.0}]
    gdf.loc[gdf["location"] == "EASINGTON", "supplier_ratios"] = [{"Equinor Energy AS": 1.0, "SHELL": 0.0, "Vår Energi ASA": 0.0}]
    gdf.loc[gdf["location"] == "ST. FERGUS", "supplier_ratios"] = [{"Equinor Energy AS": 1.0, "SHELL": 0.0, "Vår Energi ASA": 0.0}]
    gdf.loc[gdf["location"] == "EMDEN", "supplier_ratios"] = [{"Equinor Energy AS": 1.0, "SHELL": 0.0, "Vår Energi ASA": 0.0}]
    gdf.loc[gdf["location"] == "DORNUM", "supplier_ratios"] =  [{"Equinor Energy AS": 0.8, "SHELL": 0.0, "Vår Energi ASA": 0.2}]
    gdf.loc[gdf["location"] == "ZEEBRUGGE", "supplier_ratios"] = [{"Equinor Energy AS": 0.9, "SHELL": 0.1, "Vår Energi ASA": 0.0}]
    return gdf

def add_market_and_demand_parameters(gdf):
    gdf = add_market_prices(gdf)
    gdf = add_long_term_std(gdf)
    gdf = add_day_ahead_std(gdf)
    gdf = add_average_demand(gdf)
    gdf = add_demand_variance(gdf)
    gdf = add_max_component_percentages(gdf)
    gdf = add_min_pressure(gdf)
    gdf = add_supplier_ratios(gdf)
    return gdf

def add_arc_capacities(gdf):
    # Known ones
    gdf.loc[gdf["idPipeline"] == 444927.0, "max_flow"] = 57.6  # Polarled
    gdf.loc[gdf["idPipeline"] == 319132.0, "max_flow"] = 11.0  # Norne Gasstransport
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

def haversine_km(pt1, pt2):
    # pt1 and pt2 are shapely Points in (lon, lat)
    lon1, lat1 = pt1
    lon2, lat2 = pt2
    R = 6371.0  # Earth radius in km

    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c

def add_weymouth_constant(gdf):
    constant = 3.7435 * 10**(-9)
    standard_temp = 288.15
    standard_press = 1.01325
    compressibility_factor = 0.9 # Choice from menon 2005
    pipe_efficiency = 0.95 # Choice from menon 2005
    ambient_temp = 288.15
    const_exp = 2 + 2/3


    for idx, row in gdf.iterrows():
        if row["type"] == "edge":
             diameter = row["dimension"] * 25.4
             from_coords =  ast.literal_eval(row["from_node"])
             to_coords = ast.literal_eval(row["to_node"])
             length = haversine_km(from_coords, to_coords)

             numerator = constant * pipe_efficiency * standard_temp * diameter ** const_exp
             denominator = standard_press * math.sqrt(ambient_temp * length * compressibility_factor)
             
             gdf.at[idx,"weymouth_constant"] = numerator /denominator
    return gdf

def add_moap(gdf):
    for idx, row in gdf.iterrows():
        if row["type"] == "edge":
             gdf.at[idx,"max_inlet_pressure"] = 150
    return gdf

def add_arc_pressure_costs(gdf):
    for idx, row in gdf.iterrows():
        if row["type"] == "edge":
             gdf.at[idx,"pressure_cost"] = 0.1
    return gdf

def add_arc_max_ratios(gdf):
    for idx, row in gdf.iterrows():
        if row["type"] == "edge":
             gdf.at[idx, "max_pipe_fractions"] = [{"NG": 1, "CO2": 1, "H2": 0.2}]
    return gdf
    

def add_arc_parameters(gdf):
    gdf = add_arc_capacities(gdf)
    gdf = add_weymouth_constant(gdf)
    gdf = add_moap(gdf)
    gdf = add_arc_pressure_costs(gdf)
    gdf = add_arc_max_ratios(gdf)
    return gdf

if __name__ == "__main__":
    gdf = read_geojson(GDF_FILE)
    gdf.loc[gdf["location_id"] == "286141A", "location"] = "DORNUM"

    node_info = pd.read_excel(NODE_INFO_FILE)
    supplier_data  = pd.read_excel(GENERATION_NODES_FILE)
    gdf = add_generation_parameters(gdf, node_info, supplier_data)
    gdf = add_compression_node_parameters(gdf)
    gdf = add_market_and_demand_parameters(gdf)
    gdf = add_arc_parameters(gdf)

    gdf.to_file(OUTPUT_FILE, driver="GeoJSON")