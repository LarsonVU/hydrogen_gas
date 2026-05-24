import pandas as pd
import numpy as np
import pickle
from pathlib import Path


def compute_overprovision(snapshot):
    """
    Compute entry and exit overprovision values from a snapshot.

    Returns:
        entry_overprovisions (list), exit_overprovisions (list)
    """

    f_vals = snapshot["variables"]["f"]
    entry_vals = snapshot["variables"]["x_entry"]
    exit_vals = snapshot["variables"]["x_exit"]

    # =========================
    # Total capacity per node
    # =========================
    entry_overprovision = []
    exit_overprovision = []

    M_3 = set(s for (a_in, a_out, c, s) in f_vals.keys())
    K = [1,2,3]

    N = set(n for (n, k, s) in entry_vals.keys())

    for s in M_3:
        entry_overprovision_nodes = 0
        exit_overprovision_nodes = 0
        entry_amount = 0
        exit_amount = 0

        for n in N:

            entry_val = 0
            exit_val = 0

            for k in K:
                if k ==1:
                    s_prime =1
                elif k ==2:
                    s_prime = (s-1) // 8 +1
                else:
                    s_prime = s
                # average over scenarios of that stage
                entry_val += entry_vals[n, k, s_prime]
                exit_val += exit_vals[n, k, s_prime]

            inflow = sum(val for (a_in, a_out, c, s_prime), val in f_vals.items() if a_out == n and s_prime == s )
            outflow = sum(val for (a_in, a_out, c, s_prime), val in f_vals.items() if a_in == n and s_prime == s )


            net_flow = inflow - outflow
            # if abs(net_flow) > 1:
            #     print(n, net_flow)


            if net_flow >0.01:
                e_overprovision = entry_val
                ex_overprovision = max(exit_val - net_flow, 0)
                # print(n, e_overprovision)
                # print(n, ex_overprovision)
            elif net_flow < -0.01:
                e_overprovision = max(entry_val + net_flow, 0)
                ex_overprovision = exit_val
                # print(n, e_overprovision)
                # print(n, ex_overprovision)
            else:
                e_overprovision = 0
                ex_overprovision = 0
                entry_val =0 
                ex_val = 0

            
            entry_overprovision_nodes += e_overprovision
            exit_overprovision_nodes += ex_overprovision
            entry_amount += entry_val
            exit_amount += exit_val
        
        entry_overprovision.append(entry_overprovision_nodes/ entry_amount if entry_amount > 0 else 0)
        exit_overprovision.append(exit_overprovision_nodes/ exit_amount if exit_amount > 0 else 0)

    return entry_overprovision, exit_overprovision

def analyze_overprovision(base_folder, vol_multiplier, correlation, shock_type, subsidy):
    base = Path(base_folder)

    sub_dir = (
    base
    / f"vol{vol_multiplier}_{correlation}_price_{shock_type}"
    / f"sub{subsidy}"
    )
    print(sub_dir)
    entry_overprovisions = []
    exit_overprovisions = []

    for run_dir in sorted(sub_dir.glob("run*/")):
        files = list(run_dir.glob("*.pkl"))
        if not files:
            continue

        with open(files[0], "rb") as f:
            snapshot = pickle.load(f)
            entry_overprovision, exit_overprovision = compute_overprovision(snapshot)
            entry_overprovisions.append(np.mean(entry_overprovision))
            exit_overprovisions.append(np.mean(exit_overprovision))

    return entry_overprovisions, exit_overprovisions


def build_overprovision_df(base_folder):
    records = []

    for vol_dir in Path(base_folder).glob("vol*/"):
        vol_string = vol_dir.name.replace("vol", "")
        parameters_string = vol_string.replace("price_", "")
        volatility, correlation, shock_type = [param for param in parameters_string.split("_")]
        volatility = float(volatility)
        correlation = float(correlation)

        for sub_dir in vol_dir.glob("sub*/"):
            subsidy = float(sub_dir.name.replace("sub", ""))

            entry_vals, exit_vals = analyze_overprovision(base_folder, volatility, correlation, shock_type, subsidy)

            if len(entry_vals) == 0:
                continue

            for i, values in enumerate(zip(entry_vals, exit_vals)):
                records.append({
                    "vol_multiplier": volatility,
                    "correlation": correlation,
                    "shock_type" : shock_type,
                    "subsidy": subsidy,
                    "run": i, 
                    "entry_overprov": values[0],
                    "exit_overprov": values[1]
                })

    return pd.DataFrame(records)
