import networkx as nx
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import ast
import numpy as np
import pandas as pd
import os

np.random.seed(42)

GEOJSON_FILE = "data/data_analysis_results/Geojson_pipelines/study_case_network.geojson"

NUMBER_OF_STAGES = 3
BRANCHES_PER_STAGE = {1 : 1, 2 : 5, 3: 1}

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

# Graph with scenario independent attributes
def build_base_graph(geojspn_file = GEOJSON_FILE):
    """
    Build scenario-independent base graph from study_case_network.geojson
    """

    gdf = gpd.read_file(geojspn_file)

    G = nx.DiGraph()

    # -----------------------------
    # Add Nodes
    # -----------------------------
    node_rows = gdf[gdf["type"] == "node"]
    for _, row in node_rows.iterrows():
        node_id = row["location"]  # use readable name as node id
        # Convert row to dict and remove geometry
        attributes = row.drop("geometry").to_dict()
        G.add_node(node_id, **attributes)

    # -----------------------------
    # Add Edges
    # -----------------------------
    edge_rows = gdf[gdf["type"] == "edge"]

    for _, row in edge_rows.iterrows():        
        # Find matching node locations by coordinates
        from_node = find_node_by_coords(node_rows, row["from_node"])
        to_node = find_node_by_coords(node_rows, row["to_node"])

        attributes = row.drop("geometry").to_dict()

        G.add_edge(from_node, to_node, **attributes)
    
    return G


def prob_per_stage(n_stages, b_stages):
    prob_per_stage = {}
    prob_value = 1.0
    
    for k in range(1, n_stages + 1):
        prob_value /= b_stages[k]
        branches = 1
        for i in range(1, k + 1):
            branches *= b_stages[i]
        
        for m in range(1, branches + 1):
            prob_per_stage[(k, m)] = prob_value
    
    return prob_per_stage

class Scenario:
    def __init__(self, stage, index, probability, G, predecessor=None):
        self.stage = stage
        self.index = index
        self.probability = probability
        self.predecessor = predecessor
        self.G = G

def add_demand_scenarios(scenarios, branches_per_stage = BRANCHES_PER_STAGE, filename = "demand_scenarios.xlsx"):
    # DataFrame to store sampled demands for reproducibility
    demand_df = pd.DataFrame(columns=["stage", "scenario_index", "node", "supplier", "demand"])
    # Stage 2: Sample demand based on average and variance
    for scenario in scenarios[2]:
        for node in scenario.G.nodes:
            node_data = scenario.G.nodes[node]
            if "average_demand_mwh_x1000" in node_data and node_data["average_demand_mwh_x1000"] is not None:
                avg_demand =  float(node_data["average_demand_mwh_x1000"]) *1000
                variance = float(node_data.get("demand_variance", 0))
                
                # Sample variance multiplier from normal distribution
                variance_multiplier = np.random.normal(1, variance)
                sampled_demand = max(avg_demand * variance_multiplier,0)
                
                # Apply supplier ratios if available
                if "supplier_ratios" in node_data and node_data["supplier_ratios"]:
                    supplier_ratios = node_data["supplier_ratios"][0] if isinstance(node_data["supplier_ratios"], list) else node_data["supplier_ratios"]
                    scenario.G.nodes[node]["demand"] = {supplier: sampled_demand * ratio for supplier, ratio in supplier_ratios.items()}
                    for supplier, demand_value in scenario.G.nodes[node]["demand"].items():
                        demand_df.loc[len(demand_df)] = [scenario.stage, scenario.index, node, supplier, demand_value]

    demand_df.to_excel(filename, index=False)

    for scenario in scenarios[3]:
        predecessor = scenario.predecessor
        for node in scenario.G.nodes:
            if node in predecessor.G.nodes and "demand" in predecessor.G.nodes[node]:
                scenario.G.nodes[node]["demand"] = predecessor.G.nodes[node]["demand"]

def add_price_scenarios(scenarios, branches_per_stage = BRANCHES_PER_STAGE, filename =  "price_scenarios.xlsx"):
    # DataFrame to store sampled demands for reproducibility
    price_df = pd.DataFrame(columns=["stage", "scenario_index", "node", "price"])
    for scenario in scenarios[2]:
        for node in scenario.G.nodes:
            node_data = scenario.G.nodes[node]
            if "average_market_price" in node_data and node_data["average_market_price"] is not None:
                avg_price = float(node_data["average_market_price"]) 
                price_std = float(node_data.get("long_term_price_std", 0))
                
                # Sample price multiplier from normal distribution
                price_multiplier = np.random.normal(1, price_std)
                sampled_price = avg_price * price_multiplier
                
                scenario.G.nodes[node]["price"] = sampled_price

    for scenario in scenarios[3]:
        predecessor = scenario.predecessor
        for node in scenario.G.nodes:
            if node in predecessor.G.nodes:
                if  "price" in predecessor.G.nodes[node]:
                    node_data = scenario.G.nodes[node]
                    price_std = float(node_data.get("day_ahead_price_std", 0))
                    # Sample price multiplier from normal distribution
                    price_multiplier = np.random.normal(1, price_std)
                    price = max(predecessor.G.nodes[node]["price"] * price_multiplier, 0)
                    scenario.G.nodes[node]["price"] = price
                    price_df.loc[len(price_df)] = [scenario.stage, scenario.index, node, price]
                else:
                    scenario.G.nodes[node]["price"] = 0

    price_df.to_excel(filename, index=False)

# This function is currently useless and repeats data, but allows for changes per scenario if needed
def add_generation_costs(scenarios, branches_per_stage = BRANCHES_PER_STAGE):
    for scenario in scenarios[3]:
        for node in scenario.G.nodes:
            node_data = scenario.G.nodes[node]
            if "generation_cost" in node_data and node_data["generation_cost"] is not None:
                scenario.G.nodes[node]["generation_cost"] = node_data["generation_cost"] 

def add_booking_costs(scenarios, branches_per_stage = BRANCHES_PER_STAGE, percentage_increase = {2: 0.04, 3: 0.08}):
    for k in range(1, NUMBER_OF_STAGES + 1):
        for scenario in scenarios[k]:
            for node in scenario.G.nodes:
                if scenario.G.nodes[node]["compression_increase"] is None:  # Compression node
                    scenario.G.nodes[node]["booking_cost"] = scenario.G.nodes[node]["base_booking_cost"] * (1+percentage_increase.get(k, 0)) * 1000000 # Cost in euro/MScm
                    
                    
def add_scenario_attributes(scenarios, branches_per_stage = BRANCHES_PER_STAGE, folder = "study_case_model/scenario_variables/"):
    add_demand_scenarios(scenarios, branches_per_stage=branches_per_stage, filename=folder + f"demand_scenarios{''.join(str(v) for v in branches_per_stage.values())}.xlsx") 
    add_price_scenarios(scenarios, branches_per_stage=branches_per_stage, filename=folder + f"price_scenarios{''.join(str(v) for v in branches_per_stage.values())}.xlsx")
    add_generation_costs(scenarios, branches_per_stage=branches_per_stage)
    add_booking_costs(scenarios, branches_per_stage=branches_per_stage)

def create_scenarios(n_stages, b_stages, G, folder = "study_case_model/scenario_variables/"):
    # Ensure the folder exists
    os.makedirs(folder, exist_ok=True)
    
    scenarios = {k: [] for k in range(1, n_stages + 1)}

    stage_probs = prob_per_stage(n_stages, b_stages)

    branches = 1
    for k in range(1, n_stages + 1):
        branches = b_stages[k] * branches
        for m in range(1, branches + 1):
            scenario = Scenario(k, m, stage_probs[(k, m)], G.copy(), predecessor=scenarios[k-1][int((m-1) // b_stages[k])] if k > 1 else None)
            scenarios[k].append(scenario)

    add_scenario_attributes(scenarios, branches_per_stage=b_stages, folder=folder)

    return scenarios

def print_edges_network_scenario(scenario):
    print(f"Scenario: Stage {scenario.stage}, Index {scenario.index}, Probability {scenario.probability}")
    print("Edges:")
    for u, v, attr in scenario.G.edges(data=True):
        print(f"  Edge ({u}, {v}):")
        for key, value in attr.items():
            print(f"    {key}: {value}")


def print_nodes_network_scenario(scenario):
    print(f"Scenario: Stage {scenario.stage}, Index {scenario.index}, Probability {scenario.probability}")
    print("Nodes:")
    for n, attr in scenario.G.nodes(data=True):
        print(f"  Node {n}:")
        for key, value in attr.items():
            print(f"    {key}: {value}")



if __name__ == "__main__":
    G = build_base_graph()
    print(G)
    scenarios = create_scenarios(NUMBER_OF_STAGES, BRANCHES_PER_STAGE, G, folder="study_case_model/scenario_variables/")

    #Print a specific scenario for verification
    #print_nodes_network_scenario(scenarios[3][4])
    #print_edges_network_scenario(scenarios[3][4])