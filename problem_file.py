import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

NUMBER_OF_STAGES = 3
BRANCHES_PER_STAGE = {1 : 1, 2 : 5, 3: 1}

# Graph with scenario independent attributes
def build_base_graph():
    G = nx.DiGraph()

    compression_constants = {
        "NG": {"K_out_pipe": 0.05, "K_into_pipe": 0.05, "K_flow": 0.2},
        "H2": {"K_out_pipe": 0.06, "K_into_pipe": 0.06, "K_flow": 0.25},
        "CO2": {"K_out_pipe": 0.05, "K_into_pipe": 0.05, "K_flow": 0.22},
    }

    G = nx.DiGraph()

    # --- Nodes only ---
    G.add_nodes_from(["A", "B", "C", "D", "E", "F"])

    # --- Edges only ---
    G.add_edges_from([
        ("A", "C"),
        ("A", "B"),
        ("B", "C"),
        ("C", "D"),
        ("C", "F"),
        ("F", "E"),
    ])


    # --- Node attributes ---
    def add_max_flow(G):
        for n in G.nodes:
            G.nodes[n]["max_flow"] = 200
        return G
    
    def add_supplier(G):
        G.nodes["A"]["supplier"] = "Shell"
        G.nodes["B"]["supplier"] = "Equinor"
        return G

    def add_supply_capacity(G):
        G.nodes["A"]["supply_capacity"] = 140
        G.nodes["B"]["supply_capacity"] = 50
        return G
    
    def add_component_ratios(G):
        G.nodes["A"]["component_ratio"] = {"NG": 0.98, "CO2": 0.02, "H2": 0.00}
        G.nodes["B"]["component_ratio"] = {"NG": 0.00, "CO2": 0.00, "H2": 1.00}
        return G

    def add_pressure_constraints(G):
        G.nodes["D"]["min_outlet_pressure"] = 5
        G.nodes["E"]["min_outlet_pressure"] = 5
        return G

    def add_quality_constraints(G):
        G.nodes["D"]["max_fractions"] = {"NG": 1, "CO2": 0.025, "H2": 0.02}
        G.nodes["E"]["max_fractions"] = {"NG": 1, "CO2": 0.025, "H2": 0.1}
        return G

    def add_compression(G):
        G.nodes["F"]["compression_increase"] = 1.5
        G.nodes["F"]["compression_constants"] = compression_constants
        return G

    G = add_supplier(G)
    G = add_max_flow(G)
    G = add_supply_capacity(G)
    G = add_component_ratios(G)
    G = add_pressure_constraints(G)
    G = add_quality_constraints(G)
    G = add_compression(G)

    # --- Edge attributes ---
    def max_flow(G):
        for u, v in G.edges:
            G.edges[u, v]["max_flow"] = 200
        return G
    
    def weymouth_constant(G):
        for u, v in G.edges:
            G.edges[u, v]["weymouth_constant"] = 18
        return G
    
    def max_pipe_fractions(G):
        G.edges["A", "C"]["max_pipe_fractions"] = {"NG": 1, "CO2": 1, "H2": 0.2}
        G.edges["A", "B"]["max_pipe_fractions"] = {"NG": 1, "CO2": 1, "H2": 0.0}
        G.edges["B", "C"]["max_pipe_fractions"] = {"NG": 1, "CO2": 1, "H2": 0.15}
        G.edges["C", "D"]["max_pipe_fractions"] = {"NG": 1, "CO2": 1, "H2": 0.05}
        G.edges["C", "F"]["max_pipe_fractions"] = {"NG": 1, "CO2": 1, "H2": 0.3}
        G.edges["F", "E"]["max_pipe_fractions"] = {"NG": 1, "CO2": 1, "H2": 0.25}
        return G

    def add_pressure_costs(G):
        for u, v in G.edges:
            G.edges[u, v]["pressure_cost"] = 0.05
        return G
    
    def add_max_inlet_pressure(G):
        G.edges["A", "C"]["max_inlet_pressure"] = 10
        G.edges["A", "B"]["max_inlet_pressure"] = 10
        G.edges["B", "C"]["max_inlet_pressure"] = 9
        G.edges["C", "D"]["max_inlet_pressure"] = 8
        G.edges["C", "F"]["max_inlet_pressure"] = 7
        G.edges["F", "E"]["max_inlet_pressure"] = 6
        return G

    G = max_flow(G)
    G = weymouth_constant(G)
    G = max_pipe_fractions(G)
    G = add_pressure_costs(G)
    G = add_max_inlet_pressure(G)
    
    return G


def get_equal_probs_stage(k, branches=1):
    if k in range(1, NUMBER_OF_STAGES + 1):
        return [1/branches] * branches
    else:
        raise ValueError(f"Stage {k} not defined in branches_per_stage")


def prob_per_stage(n_stages, b_stages):
    prob_per_stage = {}

    branches = 1
    for k in range(1, n_stages + 1):
        branches = b_stages[k] * branches
        for m in range(1, branches + 1):
            prob_per_stage[(k, m)] = get_equal_probs_stage(k, branches=branches)[m-1]

    return prob_per_stage

class Scenario:
    def __init__(self, stage, index, probability, G, predecessor=None):
        self.stage = stage
        self.index = index
        self.probability = probability
        self.predecessor = predecessor
        self.G = G


def add_demand_scenarios(scenarios, branches_per_stage = BRANCHES_PER_STAGE):
    for scenario in scenarios[2]:
        demand_fraction = scenario.index / branches_per_stage[2]  # Fraction of demand based on branch number
        scenario.G.nodes["D"]["demand"] = {"Shell": 80 * demand_fraction, "Equinor": 5 * demand_fraction}
        scenario.G.nodes["E"]["demand"] = {"Shell": 20 * demand_fraction, "Equinor": 1 * demand_fraction}
        
    for scenario in scenarios[3]:
        predecessor = scenario.predecessor
        scenario.G.nodes["D"]["demand"] = predecessor.G.nodes["D"]["demand"]
        scenario.G.nodes["E"]["demand"] = predecessor.G.nodes["E"]["demand"]

def add_price_scenarios(scenarios, branches_per_stage = BRANCHES_PER_STAGE):
    for scenario in scenarios[3]:
        price_fraction = (scenario.index % branches_per_stage[3]) / branches_per_stage[3]  # Fraction of price based on branch number
        scenario.G.nodes["D"]["price"] = 1 + 1 * price_fraction
        scenario.G.nodes["E"]["price"] = 1 + 1 * price_fraction

def add_generation_costs(scenarios, branches_per_stage = BRANCHES_PER_STAGE):
    for scenario in scenarios[3]:
        scenario.G.nodes["A"]["generation_cost"] = 0.5
        scenario.G.nodes["B"]["generation_cost"] = 1.2

def add_booking_costs(scenarios, branches_per_stage = BRANCHES_PER_STAGE):
    for k in range(1, NUMBER_OF_STAGES + 1):
        for scenario in scenarios[k]:
            for node in scenario.G.nodes:
                if not "compression_increase" in scenario.G.nodes[node]:  # Compression node
                    scenario.G.nodes[node]["booking_cost"] = 0.1 + 0.01 * k


def add_scenario_attributes(scenarios, branches_per_stage = BRANCHES_PER_STAGE):
    add_demand_scenarios(scenarios, branches_per_stage=branches_per_stage)
    add_price_scenarios(scenarios, branches_per_stage=branches_per_stage)
    add_generation_costs(scenarios, branches_per_stage=branches_per_stage)
    add_booking_costs(scenarios, branches_per_stage=branches_per_stage)

def create_scenarios(n_stages, b_stages, G):
    scenarios = {k: [] for k in range(1, n_stages + 1)}

    stage_probs = prob_per_stage(n_stages, b_stages)

    branches = 1
    for k in range(1, n_stages + 1):
        branches = b_stages[k] * branches
        for m in range(1, branches + 1):
            scenario = Scenario(k, m, stage_probs[(k, m)], G.copy(), predecessor=scenarios[k-1][int((m-1) // b_stages[k])] if k > 1 else None)
            scenarios[k].append(scenario)

    add_scenario_attributes(scenarios, branches_per_stage=b_stages)

    return scenarios

def print_network_scenario(scenario):
    print(f"Scenario: Stage {scenario.stage}, Index {scenario.index}, Probability {scenario.probability}")
    print("Nodes:")
    for n, attr in scenario.G.nodes(data=True):
        print(f"  Node {n}:")
        for key, value in attr.items():
            print(f"    {key}: {value}")
    print("Edges:")
    for u, v, attr in scenario.G.edges(data=True):
        print(f"  Edge ({u}, {v}):")
        for key, value in attr.items():
            print(f"    {key}: {value}")


if __name__ == "__main__":
    G = build_base_graph()
    scenarios = create_scenarios(NUMBER_OF_STAGES, BRANCHES_PER_STAGE, G)

    # Print a specific scenario for verification
    print_network_scenario(scenarios[3][4])