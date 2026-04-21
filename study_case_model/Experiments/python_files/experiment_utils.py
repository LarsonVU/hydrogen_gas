import pandas as pd
import networkx as nx
import pandas as pd


# =========================
# Helper functions
# =========================
# https://www.engineeringtoolbox.com/fuels-higher-calorific-values-d_169.html
# 120 MJ/kg for hydrogen, which is 33.3 kWh/kg. With a density of 0.08988 kg/m3, this is 2.78 MWh per 1000 scm (or 1 mscm)
def subsidy_per_mwh_to_mscm(mwh_subsidy, LHV_mwh_per_kscm=2.78):
    return mwh_subsidy * LHV_mwh_per_kscm * 1000


def apply_subsidy(G, subsidy_value, variable_name="generation_cost"):
    G_changed = G.copy()

    for node in G.nodes:
        if not pd.isna(G.nodes[node][variable_name]):
            if G.nodes[node]["component_ratio"]["H2"] > 0:
                G_changed.nodes[node][variable_name] = (
                    float(G.nodes[node][variable_name]) - subsidy_value
                )
    return G_changed


def apply_market_restriction(G, market, allowed_hydrogen):
    G_changed = G.copy()
    G_changed.nodes[market]["max_fractions"]["H2"] = allowed_hydrogen

    return G_changed


def map_name(name):
    mapping = {
        "GJØA": "GJOA",
        "VISUND": "VISUND",
        "NORNE ERB": "NORNE_ERB",
        "KÅRSTØ": "KARSTO",
        "DRAUPNER S": "DRAUPNER_S",
        "DORNUM": "DORNUM",
        "DUNKERQUE": "DUNKERQUE",
        "H-7 BP": "H-7_BP",
        "EMDEN": "EMDEN"
    }
    return mapping.get(name, name)

def apply_technical_restriction(G, failed_pipe_from=None, failed_pipe_to=None, failed_plant=None):
    G_changed = G.copy()

    # Rename all nodes using the mapping
    node_mapping = {node: map_name(node) for node in G_changed.nodes()}
    G_changed = nx.relabel_nodes(G_changed, node_mapping)

    # Also update edge metadata that stores original node IDs
    for u, v, data in G_changed.edges(data=True):
        for key in ("from_node", "to_node", "from", "to"):
            if key in data:
                data[key] = node_mapping.get(data[key], data[key])

    print("Renamed nodes according to mapping.")
    print("Nodes after renaming:", G_changed.nodes(data=False))
    print("Edges after renaming:", G_changed.edges(data=False))
    if failed_pipe_from is not None and failed_pipe_to is not None:
        edge = (map_name(failed_pipe_from), map_name(failed_pipe_to))
        G_changed.edges[edge]["max_flow"] = 0
        print(G_changed.edges[edge])

    if failed_plant is not None:
        if failed_plant == "None":
            print("No plant failure specified (baseline).")
            return G_changed
        node = map_name(failed_plant)
        G_changed.nodes[node]["supply_capacity"] = 0

    return G_changed

def subsidy_per_mwh_to_mscm(mwh_subsidy, gcv_mwh_per_kscm=2.78):
    return mwh_subsidy * gcv_mwh_per_kscm * 1000


def apply_subsidy(G, subsidy_value, variable_name="generation_cost"):
    G_changed = G.copy()

    for node in G.nodes:
        if not pd.isna(G.nodes[node][variable_name]):
            if G.nodes[node]["component_ratio"]["H2"] > 0:
                G_changed.nodes[node][variable_name] = (
                    float(G.nodes[node][variable_name]) - subsidy_value
                )
    return G_changed


def apply_technical_restriction_h2(G, allowed_hydrogen):
    G_changed = G.copy()
    for edge in G_changed.edges:
        G_changed.edges[edge]["max_pipe_fractions"][0]["H2"] = allowed_hydrogen

    return G_changed