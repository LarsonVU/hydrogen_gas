import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def plot_network(G):
    """
    Plot a directed network with supply, demand, storage information
    and arc capacities using only NetworkX attributes.
    """

    pos = nx.spring_layout(G, seed=100)  # For consistent layout across runs

    # Categorize nodes
    gas_supply_nodes = []
    hydrogen_supply_nodes = []
    demand_nodes = []
    storage_nodes = []
    normal_nodes = []

    for n, data in G.nodes(data=True):
        if data.get("h2_fraction", 0) > 0:
            gas_supply_nodes.append(n)
        elif data.get("supply_capacity", 0) > 0:
            hydrogen_supply_nodes.append(n)
        elif data.get("demand", 0) > 0:
            demand_nodes.append(n)
        elif data.get("compression_max", 0) > 0:
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
    edge_labels = {
        (u, v): f"cap={d['max_flow']}"
        for u, v, d in G.edges(data=True)
        if "max_flow" in d
    }

    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=edge_labels,
        font_size=9
    )

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

    nodes = {
        "A": dict(max_flow=200, supply_capacity=140, component_ratio={"NG": 0.98, "CO2": 0.02, "H2": 0.00}, generation_cost=0.5, supplier="Shell"),
        "B": dict(max_flow=200, supply_capacity=50, component_ratio={"NG": 0.00, "CO2": 0.00, "H2": 1.00}, generation_cost=1.2, supplier="Equinor"),
        "C": dict(max_flow=200, split_homogeneous=True),
        "D": dict(max_flow=200, demand={"Shell": 80, "Equinor": 40}, price=1.1),
        "E": dict(max_flow=200, demand={"Shell": 20, "Equinor": 20}, price=1.3),
        "F": dict(max_flow=200, compression_max=30),
    }

    edges = [
        ("A", "C", dict(max_flow=90, max_hydrogen_fraction=0.1,
                        max_inlet_pressure=10, pressure_cost=0.05, weymouth_constant=18)),
        ("B", "C", dict(max_flow=40, max_hydrogen_fraction=0.15,
                        max_inlet_pressure=9, pressure_cost=0.04, weymouth_constant=18)),
        ("C", "D", dict(max_flow=140, max_hydrogen_fraction=0.05,
                        max_inlet_pressure=8, pressure_cost=0.06, weymouth_constant=18)),
        ("C", "F", dict(max_flow=70, max_hydrogen_fraction=0.3,
                        min_inlet_pressure=5, max_inlet_pressure=7,
                        pressure_cost=0.05, weymouth_constant=18)),
        ("F", "E", dict(max_flow=60, max_hydrogen_fraction=0.25,
                        min_inlet_pressure=5, max_inlet_pressure=6,
                        pressure_cost=0.07, weymouth_constant=18)),
    ]

    G.add_nodes_from((n, attr) for n, attr in nodes.items())
    G.add_edges_from((u, v, attr) for u, v, attr in edges)

    return G

if __name__ == "__main__":
    # Plot
    G = example_graph()
    plot_network(G)
