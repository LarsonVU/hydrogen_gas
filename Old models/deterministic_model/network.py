import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def plot_network(G):
    """
    Plot a directed network with supply, demand, storage information
    and arc capacities using only NetworkX attributes.
    """

    pos = nx.circular_layout(G)  # For consistent layout across runs

    # Categorize nodes
    gas_supply_nodes = []
    hydrogen_supply_nodes = []
    demand_nodes = []
    storage_nodes = []
    normal_nodes = []

    for n, data in G.nodes(data=True):
        if data.get("component_ratio", {}).get("NG", 0) > 0:
            gas_supply_nodes.append(n)
        elif data.get("supply_capacity", 0) > 0:
            hydrogen_supply_nodes.append(n)
        elif any(v > 0 for v in data.get("demand", {}).values()):
            demand_nodes.append(n)
        elif data.get("compression_increase", 0) > 0:
            storage_nodes.append(n)
        else:
            normal_nodes.append(n)

    plt.figure(figsize=(8, 6))

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=normal_nodes,
        node_color="lightgray",
        node_size=1000
    )

    nx.draw_networkx_nodes(
        G, pos,
        nodelist=gas_supply_nodes,
        node_color="wheat",
        node_size=1000
    )

    nx.draw_networkx_nodes(
        G, pos,
        nodelist=hydrogen_supply_nodes,
        node_color="lightskyblue",
        node_size=1000
    )

    nx.draw_networkx_nodes(
        G, pos,
        nodelist=storage_nodes,
        node_color="lightgreen",
        node_size=1000
    )

    nx.draw_networkx_nodes(
        G, pos,
        nodelist=demand_nodes,
        node_color="salmon",
        node_size=1000
    )

    # Draw edges
    nx.draw_networkx_edges(
        G, pos,
        arrowstyle="->",
        arrowsize=20,
        width=2
    )

    # Node labels
    nx.draw_networkx_labels(G, pos, font_size=10 )

    # Edge capacity labels
    # edge_labels = {
    #     (u, v): f"cap={d['max_flow']}"
    #     for u, v, d in G.edges(data=True)
    #     if "max_flow" in d
    # }

    # nx.draw_networkx_edge_labels(
    #     G,
    #     pos,
    #     edge_labels=edge_labels,
    #     font_size=9
    # )

    # Legend
    legend_elements = [
        Patch(facecolor="wheat", label="Gas Supply node"),
        Patch(facecolor="lightskyblue", label="Hydrogen Supply node"),
        Patch(facecolor="salmon", label="Demand node"),
        Patch(facecolor="lightgreen", label="Compression node"),
        Patch(facecolor="lightgray", label="Intermediate node"),
    ]

    plt.legend(handles=legend_elements, loc="best")
    plt.axis("off")
    plt.title("Network with Arc Capacities")
    plt.tight_layout()
    plt.show()

def example_graph():
    G = nx.DiGraph()

    compression_constants = {
        "NG": {"K_out_pipe": 0.05, "K_into_pipe": 0.05, "K_flow": 0.2},
        "H2": {"K_out_pipe": 0.06, "K_into_pipe": 0.06, "K_flow": 0.25},
        "CO2": {"K_out_pipe": 0.05, "K_into_pipe": 0.05, "K_flow": 0.22},
    }

    nodes = {
        "A": dict(max_flow=200, supply_capacity=140, component_ratio={"NG": 0.98, "CO2": 0.02, "H2": 0.00}, generation_cost=0.5, supplier="Shell", split_homogeneous=True),
        "B": dict(max_flow=200, supply_capacity=50, component_ratio={"NG": 0.00, "CO2": 0.00, "H2": 1.00}, generation_cost=1.2, supplier="Equinor"),
        "C": dict(max_flow=200),
        "D": dict(max_flow=200, demand={"Shell": 80, "Equinor": 5}, price=1.1, min_outlet_pressure=5, max_fractions = {"NG": 1, "CO2": 0.025, "H2": 0.02}),
        "E": dict(max_flow=200, demand={"Shell": 20, "Equinor": 1}, price=1.3, min_outlet_pressure=5, max_fractions = {"NG": 1, "CO2": 0.025, "H2": 0.1}), 
        "F": dict(max_flow=200, compression_increase=1.5, compression_constants=compression_constants),
    }

    edges = [
        ("A", "C", dict(max_flow=90, max_pipe_fractions={"NG": 1, "CO2": 1, "H2": 0.2},
                        max_inlet_pressure=10, pressure_cost=0.05, weymouth_constant=18)),
        ("A", "B", dict(max_flow=90, max_pipe_fractions={"NG": 1, "CO2": 1, "H2": 0.00},
                max_inlet_pressure=10, pressure_cost=0.05, weymouth_constant=18)),
        ("B", "C", dict(max_flow=40, max_pipe_fractions={"NG": 1, "CO2": 1, "H2": 0.15},
                        max_inlet_pressure=9, pressure_cost=0.04, weymouth_constant=18)),
        ("C", "D", dict(max_flow=140, max_pipe_fractions={"NG": 1, "CO2": 1, "H2": 0.05},
                        max_inlet_pressure=8, pressure_cost=0.06, weymouth_constant=18)),
        ("C", "F", dict(max_flow=70, max_pipe_fractions={"NG": 1, "CO2": 1, "H2": 0.3},
                         max_inlet_pressure=7, pressure_cost=0.01, weymouth_constant=18)),
        ("F", "E", dict(max_flow=60, max_pipe_fractions={"NG": 1, "CO2": 1, "H2": 0.25},
                         max_inlet_pressure=6,
                        pressure_cost=0.07, weymouth_constant=18)),
    ]

    G.add_nodes_from((n, attr) for n, attr in nodes.items())
    G.add_edges_from((u, v, attr) for u, v, attr in edges)

    return G

if __name__ == "__main__":
    # Plot
    G = example_graph()
    plot_network(G)
