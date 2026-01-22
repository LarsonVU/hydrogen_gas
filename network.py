import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def plot_network(G):
    """
    Plot a directed network with supply, demand, storage information
    and arc capacities using only NetworkX attributes.
    """

    pos = nx.spring_layout(G, seed=42)

    # Categorize nodes
    gas_supply_nodes = []
    hydrogen_supply_nodes = []
    demand_nodes = []
    storage_nodes = []
    normal_nodes = []

    for n, data in G.nodes(data=True):
        if data.get("gas_supply_capacity", 0) > 0:
            gas_supply_nodes.append(n)
        elif data.get("hydrogen_supply_capacity", 0) > 0:
            hydrogen_supply_nodes.append(n)
        elif data.get("demand", 0) > 0:
            demand_nodes.append(n)
        elif data.get("storage_capacity", 0) > 0:
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
    nx.draw_networkx_labels(G, pos, font_size=10)

    # Edge capacity labels
    edge_labels = {
        (u, v): f"cap={d['capacity']}"
        for u, v, d in G.edges(data=True)
        if "capacity" in d
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
        Patch(facecolor="lightgreen", label="Storage node"),
        Patch(facecolor="lightgray", label="Intermediate node"),
    ]

    plt.legend(handles=legend_elements, loc="best")
    plt.axis("off")
    plt.title("Network with Arc Capacities")
    plt.tight_layout()
    plt.show()

def example_graph():
    # Create directed graph
    G = nx.DiGraph()

    # Add nodes with attributes
    G.add_node("A", gas_supply_capacity=140)
    G.add_node("B")
    G.add_node("C", storage_capacity=50)
    G.add_node("D", demand=80)
    G.add_node("E", hydrogen_supply_capacity=50)

    # Add edges with attributes
    G.add_edge("A", "B", capacity=90, flow_cost=2, max_hydrogen_fraction=0.1)
    G.add_edge("A", "C", capacity=60, flow_cost=1, max_hydrogen_fraction=0.2)
    G.add_edge("C", "B", capacity=40, flow_cost=1.5, max_hydrogen_fraction=0.15)
    G.add_edge("B", "D", capacity=140, flow_cost=2.5, max_hydrogen_fraction=0.05)
    G.add_edge("E", "B", capacity=70, flow_cost=3, max_hydrogen_fraction=0.3)
    G.add_edge("C", "E", capacity=999, flow_cost=2, max_hydrogen_fraction=0.25)
    return G

# Plot
G = example_graph()
plot_network(G)
