from network import example_graph, plot_network
import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use('Agg')  # Ensure it works in Codespaces terminal
import pyomo.environ as pyo
import networkx
import numpy as np
import itertools

NUMBER_OF_HOMOGENEOUS_SPLITS =11

splits_per_arc = np.linspace(0, 1, NUMBER_OF_HOMOGENEOUS_SPLITS)


NUMBER_OF_CUTTING_PLANES_P_IN = 10
P_IN_LOW = 4
P_IN_HIGH = 10

NUMBER_OF_CUTTING_PLANES_P_OUT = 10
P_OUT_LOW = 4
P_OUT_HIGH = 10

# Function to generate cutting plane pairs using the constants
def generate_cutting_plane_pairs(
    n_p_out=NUMBER_OF_CUTTING_PLANES_P_OUT,
    p_out_low=P_OUT_LOW,
    p_out_high=P_OUT_HIGH,
    n_p_in=NUMBER_OF_CUTTING_PLANES_P_IN,
    p_in_low=P_IN_LOW,
    p_in_high=P_IN_HIGH
):
    # Evenly spaced p_out
    p_out_values = np.linspace(p_out_low, p_out_high, n_p_out)
    
    # Function to skew p_in toward lower values (denser near p_out)
    def skew_p_in(p_out, n_points=n_p_in, low=None, high=p_in_high):
        if low is None:
            low = max(p_in_low, p_out)  # ensure p_in >= p_out
        t = np.linspace(0, 1, n_points)
        t_skewed = np.power(t,2)  # adjust exponent for more/less skew
        return low + (high - low) * t_skewed
    
    # Create p_in values for each p_out
    p_in_grid = [skew_p_in(p_out) for p_out in p_out_values]
    
    # Flatten into pairs (p_in, p_out), only take p_in > p_out
    pairs = [
        (p_in, p_out)
        for p_out, p_in_list in zip(p_out_values, p_in_grid)
        for p_in in p_in_list
        if p_in > p_out
    ]
    
    return pairs

# Usage
pairs = generate_cutting_plane_pairs()

NUMBER_OF_DENSITY_BOUNDS = 10
RHO_LOW = 0.55
RHO_HIGH = 0.70

def create_base_model(network: networkx.Graph):
    model = pyo.ConcreteModel()

    #-----------------------------
    # Sets
    #-----------------------------
    model.N = pyo.Set(initialize=list(network.nodes))
    model.A = pyo.Set(initialize=list(network.edges), dimen=2)

    model.N_hg = pyo.Set(
        within=model.N,
        initialize=[n for n, d in network.nodes(data=True) if "supply_capacity" in d]
    ) # Production nodes

    model.N_m = pyo.Set(
        within=model.N,
        initialize=[n for n, d in network.nodes(data=True) if "demand" in d]
    )# Market nodes

    model.N_gamma = pyo.Set(
        within=model.N,
        initialize=[n for n, d in network.nodes(data=True) if "compression_increase" in d]
    ) # Nodes with compressors

    model.N_s = pyo.Set(
        within=model.N,
        initialize=[n for n, d in network.nodes(data=True) if d.get("split_homogeneous", False)]
    ) # Nodes where gas is split homogeneously

    # Incoming and outgoing arcs per node
    def A_in_rule(model, n):
        return [a for a in model.A if a[1] == n]  # arcs ending at node n
    model.A_n_plus = pyo.Set(model.N, initialize=A_in_rule)

    def A_out_rule(model, n):
        return [a for a in model.A if a[0] == n]  # arcs starting from node n
    model.A_n_minus = pyo.Set(model.N, initialize=A_out_rule)

    # Components of the gas mix
    model.C = pyo.Set(initialize=['NG', 'CO2', 'H2'])  # Set of components

    # Special order sets
    
    model.Z_theta = pyo.Set(initialize = range(NUMBER_OF_DENSITY_BOUNDS))  # Type 1, pressure upper bound
    
    E_nz_init = {}

    # Homogeneous splitting parameters
    for n, arcs in model.A_n_minus.items():
        if n in model.N_s:
            order_arcs = len(arcs)
            E_nz_init[n] = [z for z in itertools.product(splits_per_arc, repeat=order_arcs) if sum(z) == 1]

    model.Z_v = pyo.Set(model.N_s, initialize = lambda model ,n: range(len(E_nz_init[n])) if n in E_nz_init else range(0))      # Type 1, homogeneous splitting

    # Auxiliary set for defining E
    def E_index_init(model):
        return (
            (n, z, a)
            for n in model.N_s
            for z in model.Z_v[n]
            for a in model.A_n_minus[n]
        )

    model.E_index = pyo.Set(initialize=E_index_init)

    # Auxilliary set for defining e
    def e_index_init(m):
        return (
            (n, c, z)
            for n in m.N_s
            for c in m.C
            for z in m.Z_v[n]
        )

    model.e_index = pyo.Set(dimen=3, initialize=e_index_init)

    # Auxilliary set for defining v
    def v_index_init(m):
        return (
            (n, z)
            for n in m.N_s
            for z in m.Z_v[n]
        )
   
    model.v_index = pyo.Set(dimen=2, initialize=v_index_init)

    # Other sets
    model.L = pyo.Set(initialize = range(len(pairs)))      # Set of cutting planes
    model.H = pyo.Set(initialize=list(set().union(
                        *(network.nodes[n].get('demand', {}).keys()
                        for n in model.N_m if 'demand' in network.nodes[n])
                        )))
      # Set of suppliers
    
    #-----------------------------
    # Parameters
    #-----------------------------

    # Objective function parameters
    model.o_n_d = pyo.Param(model.N, initialize={n: network.nodes[n]['price'] for n in model.N_m})
    model.o_n_g = pyo.Param(model.N, initialize={n: network.nodes[n]['generation_cost'] for n in model.N_hg})
    model.o_p_a = pyo.Param(model.A, initialize={a: network.edges[a]['pressure_cost'] for a in model.A})

    # Parameters for production constraints
    model.G = pyo.Param(
        model.N_hg,
        initialize={n: network.nodes[n].get('supply_capacity', 0) 
                    for n in model.N_hg}
    )
    model.alpha = pyo.Param(
        model.N_hg, model.C,
        initialize={(n, c): network.nodes[n].get('component_ratio', {}).get(c, 0) 
                    for n in model.N_hg for c in model.C}
    )

    # Parameters for demand
    def demand_rule(m, h, n):
        return network.nodes[n].get('demand', {}).get(h, 0)

    model.D = pyo.Param(
        model.H,
        model.N_m,
        initialize=demand_rule,
    )

    # Suppliers parameter
    model.supplier = pyo.Param(
        model.N_hg,
        initialize={n: network.nodes[n].get('supplier', 'Unknown') for n in model.N_hg},
    )

    # Max flow (Big-M)
    model.M_n = pyo.Param(model.N, initialize = {n: network.nodes[n]['max_flow'] for n in model.N})
    model.M_a =  pyo.Param(model.A, initialize = {a: network.edges[a]['max_flow'] for a in model.A})

    # Relative density Parameters
    model.rho_c = pyo.Param(model.C, initialize ={"NG" : 0.65, "CO2"  : 1.53, "H2" : 0.070})

    rho_values = np.linspace(RHO_LOW, RHO_HIGH, NUMBER_OF_DENSITY_BOUNDS + 1)[1:]

    model.rho_Z = pyo.Param(
        model.Z_theta,
        initialize={z: rho_values[i] for i, z in enumerate(model.Z_theta)}
    )

    # Weymouth constants
    model.K_a = pyo.Param(
        model.A,
        initialize={a: network.edges[a]['weymouth_constant'] for a in model.A}
    )

    def K_az_rule(model, a_in, a_out, z):
        return model.K_a[a_in, a_out] / (model.rho_Z[z] ** 0.5)

    model.K_az = pyo.Param(model.A, model.Z_theta, initialize=K_az_rule)

    # Cutting plane parameters
    model.P_in = pyo.Param(
        model.L,
        initialize={l: pairs[l][0] for l in range(len(pairs))}
    )

    model.P_out = pyo.Param(
        model.L,
        initialize={l: pairs[l][1] for l in range(len(pairs))}
    )

    # Maximum pressure per arc
    model.P_max = pyo.Param(model.A, initialize={a: network.edges[a].get('max_inlet_pressure', 1000) for a in model.A})

    # Minimum pressure at demand nodes
    model.P_min = pyo.Param(model.N_m, initialize={n: network.nodes[n].get('min_outlet_pressure', 0) for n in model.N_m})

    # Fuel consumption parameters for compression nodes (per component)
    model.K_out_pipe = pyo.Param(
        model.N_gamma,
        model.C,
        initialize=lambda model, n, c:
            network.nodes[n]
            .get('compression_constants', {})
            .get(c, {})
            .get('K_out_pipe', 0),
        default=0
    )

    model.K_into_pipe = pyo.Param(
        model.N_gamma,
        model.C,
        initialize=lambda model, n, c:
            network.nodes[n]
            .get('compression_constants', {})
            .get(c, {})
            .get('K_into_pipe', 0),
        default=0
    )

    model.K_flow = pyo.Param(
        model.N_gamma,
        model.C,
        initialize=lambda model, n, c:
            network.nodes[n]
            .get('compression_constants', {})
            .get(c, {})
            .get('K_flow', 0),
        default=0
    )

    # Pressure increase from compression
    model.P_hat = pyo.Param(model.N_gamma, initialize={n: network.nodes[n].get('compression_increase', 1) for n in model.N_gamma})

    # Pipe quality constraints
    model.q_plus_arc = pyo.Param(model.A, model.C, initialize=lambda model, a1,a2, c: network.edges[a1,a2].get('max_pipe_fractions', {}).get(c, 0))
    # Unused
    model.q_minus_arc = pyo.Param(model.A, model.C, initialize=lambda model, a1,a2, c: 0)

    # Node quality constraints
    model.q_plus_node = pyo.Param(model.N_m, model.C, initialize=lambda model, n, c: network.nodes[n].get('max_fractions', {}).get(c, 0))
    # Unused
    model.q_minus_node = pyo.Param(model.N_m, model.C, initialize=lambda model, n, c: 0)


    def E_init(m, n, z, a1,a2):
        arc_idx = list(m.A_n_minus[n]).index((a1,a2))
        return E_nz_init[n][z][arc_idx]

    model.E_nza = pyo.Param(model.E_index, initialize=E_init)

    # -----------------------------
    # Variables
    #-----------------------------
    model.f = pyo.Var(model.A, model.C, within=pyo.NonNegativeReals, initialize=0)  # Flow on arc a of component c
    model.p_in = pyo.Var(model.A, within=pyo.NonNegativeReals, initialize=0)  # Inlet pressure on arc a
    model.p_out = pyo.Var(model.A, within=pyo.NonNegativeReals, initialize=0)  # Outlet pressure on arc a
    model.theta = pyo.Var(model.A, model.Z_theta, within=pyo.Binary, initialize=0)  # Binary variables for pressure bounds
    model.v = pyo.Var(model.v_index, within=pyo.Binary, initialize=0)  # Binary variables for homogeneous splitting
    model.w = pyo.Var(model.N_gamma, model.C, within=pyo.NonNegativeReals, initialize=0)  # Continuous variables for compression fuel consumption
    model.e = pyo.Var(model.e_index, domain=pyo.NonNegativeReals, initialize=0)  # Auxiliary variable for homogeneous splitting flow

    #-----------------------------
    # Objective Function
    #-----------------------------
    def objective_rule(model):
        revenue = sum(model.o_n_d[n] * sum(model.f[a, c] for a in model.A_n_plus[n] for c in model.C) 
                      for n in model.N_m)
        generation_cost = sum(model.o_n_g[n] * sum(model.f[a, c] for a in model.A_n_minus[n] for c in model.C) 
                              for n in model.N_hg)
        pressure_cost = sum(model.o_p_a[a] * model.p_in[a] for a in model.A)
        return revenue - generation_cost - pressure_cost

    model.objective = pyo.Objective(rule=objective_rule, sense=pyo.maximize)

    #  -----------------------------
    # Constraints
    #  -----------------------------

    # ________________
    # Flow constraints
    # ________________

    # Constraint 1: Production capacity constraint
    def production_capacity_rule(model, n):
        if n in model.N_hg:
            return sum(model.f[a, c] for a in model.A_n_minus[n] for c in model.C) - sum(model.f[a, c] for a in model.A_n_plus[n] for c in model.C) <= model.G[n]
        return pyo.Constraint.Skip
    model.production_capacity = pyo.Constraint(model.N_hg, rule=production_capacity_rule)

    # Constraint 2: Component ratio constraint (byproduct/stoichiometric)
    def component_ratio_rule(model, n, c):
        if n in model.N_hg and c in model.C:
            return sum(model.f[a, c] for a in model.A_n_minus[n]) - sum(model.f[a, c] for a in model.A_n_plus[n]) == model.alpha[n, c] * (sum(model.f[a, cp] for a in model.A_n_minus[n] for cp in model.C) - sum(model.f[a, cp] for a in model.A_n_plus[n] for cp in model.C))
        return pyo.Constraint.Skip
    model.component_ratio = pyo.Constraint(model.N_hg, model.C, rule=component_ratio_rule)

    # Constraint 3: Demand satisfaction constraint for production nodes
    def demand_satisfaction_production_rule(model, h):
        return sum(model.f[a, c] for c in model.C for n_prime in model.N_hg if model.supplier[n_prime] == h for a in model.A_n_minus[n_prime] ) -sum(model.f[a, c] for c in model.C for n_prime in model.N_hg if model.supplier[n_prime] == h for a in model.A_n_plus[n_prime] ) >= sum(model.D[h,n] for n in model.N_m)
    model.demand_satisfaction_production = pyo.Constraint( model.H, rule=demand_satisfaction_production_rule)

    # Constraint 4: Demand satisfaction constraint for market nodes
    def demand_satisfaction_market_rule(model, n):
        if n in model.N_m:
            return sum(model.f[a, c] for a in model.A_n_plus[n] for c in model.C) >= sum(model.D[h,n] for h in model.H)
        return pyo.Constraint.Skip
    model.demand_satisfaction_market = pyo.Constraint(model.N_m, rule=demand_satisfaction_market_rule)

    # Constraint 5: Flow balance constraint for intermediate nodes
    def flow_balance_rule(model, n, c):
        # Applies to nodes that are not production, market, or compression nodes
        intermediate_nodes = model.N - model.N_hg - model.N_m - model.N_gamma
        if n in intermediate_nodes:
            inflow = sum(model.f[a, c] for a in model.A_n_plus[n])
            outflow = sum(model.f[a, c] for a in model.A_n_minus[n])
            return inflow == outflow
        return pyo.Constraint.Skip

    model.flow_balance = pyo.Constraint(model.N, model.C, rule=flow_balance_rule)

    # ________________
    # Weymouth constraints
    # ________________

    def SOS1_theta(model, a1, a2):
        return sum(model.theta[a1,a2,z] for z in model.Z_theta) == 1 

    model.S1_theta = pyo.Constraint(model.A, rule =SOS1_theta)

    def upperbound_rho_rule(model, a1, a2, z):
        total_flow = sum(model.f[a1,a2, c] for c in model.C)
        weighted_flow = sum(model.rho_c[c] * model.f[a1,a2, c] for c in model.C)

        return (
            model.rho_Z[z] * total_flow
            + model.M_a[a1,a2] * (1 - model.theta[a1, a2, z])
            >= weighted_flow
        )

    model.upperbound_rho = pyo.Constraint(model.A, model.Z_theta, rule=upperbound_rho_rule)

    def weymouth_cutting_plane_rule(model, a1, a2, z, l):
        total_flow = sum(model.f[a1,a2, c] for c in model.C)
        
        # Cutting plane coefficients
        p_in_l = model.P_in[l]
        p_out_l = model.P_out[l]
        denominator = pyo.sqrt(p_in_l**2 - p_out_l**2)
        
        coeff_in = p_in_l / denominator
        coeff_out = p_out_l / denominator
        
        return (
            total_flow
            <= model.K_az[a1,a2, z] * (coeff_in * model.p_in[a1, a2] - coeff_out * model.p_out[a1, a2])
            + model.M_a[a1,a2] * (1 - model.theta[a1, a2, z])
        )

    model.weymouth_cutting_plane = pyo.Constraint(model.A, model.Z_theta, model.L, rule=weymouth_cutting_plane_rule)

    # ________________
    # Other pressure constraints
    # ________________

    #Node pressure relationship
    def node_pressure_rule(model, n, a_into_node_1, a_into_node_2, a_out_node_1, a_out_node_2):
        return model.p_in[a_out_node_1, a_out_node_2] <= model.p_out[a_into_node_1, a_into_node_2]

    model.node_pressure = pyo.Constraint(
        ((n, a_into_node, a_out_node)
        for n in model.N if n not in model.N_gamma
        for a_into_node in model.A_n_plus[n]
        for a_out_node in model.A_n_minus[n]),
        rule=node_pressure_rule
    )

    # Inlet pressure must be greater than outlet pressure for arcs
    def pressure_drop_rule(model, a1, a2):
        return model.p_in[a1, a2] >= model.p_out[a1, a2]
    
    model.pressure_drop = pyo.Constraint(model.A, rule=pressure_drop_rule)

    # Maximum pressure 
    def max_pressure_rule(model, a1, a2):
        return model.p_in[a1, a2] <= model.P_max[a1, a2]
    
    model.max_pressure = pyo.Constraint(model.A, rule=max_pressure_rule)

    # Minimum pressure demand nodes
    def min_pressure_demand_rule(model, n, a1,a2):
        return model.p_out[a1,a2] >= model.P_min[n]

    model.min_pressure_demand = pyo.Constraint(
        ((n, a) for n in model.N_m for a in model.A_n_plus[n]),
        rule=min_pressure_demand_rule)
    
    # Compression fuel consumption constraints
    def fuel_consumption_rule(model, n,c):
        if n in model.N_gamma:
            # For each arc into the compression node, compute fuel consumption
            a = model.A_n_minus[n].first()
            a_prime = model.A_n_plus[n].first()
            return model.w[n, c] == model.K_out_pipe[n, c] * model.p_in[a] - model.K_into_pipe[n, c] * model.p_out[a_prime] + model.K_flow[n, c] * model.f[a,c]
        return pyo.Constraint.Skip

    model.fuel_consumption = pyo.Constraint(model.N_gamma, model.C, rule=fuel_consumption_rule)

    def fuel_flow_balance_rule(model, n,c):
        if n in model.N_gamma:
            return sum(model.f[a, c] for a in model.A_n_minus[n]) == sum(model.f[a, c] for a in model.A_n_plus[n]) -  model.w[n, c] 
        return pyo.Constraint.Skip

    model.fuel_flow_balance = pyo.Constraint(model.N_gamma, model.C, rule=fuel_flow_balance_rule)

    def compression_increase_rule(model, n):
        if n in model.N_gamma:
            a = model.A_n_minus[n].first()
            a_prime = model.A_n_plus[n].first()
            return model.p_in[a] == model.P_hat[n] * model.p_out[a_prime]
        return pyo.Constraint.Skip

    model.compression_increase = pyo.Constraint(model.N_gamma, rule=compression_increase_rule)

    # ________________
    # Quality constraints
    # ________________

    # Composition per pipe
    def pipe_composition_max_rule(model, a1, a2, c):
        return model.f[a1, a2,c] <= model.q_plus_arc[a1, a2,c] * sum(model.f[a1, a2, cp] for cp in model.C)

    model.pipe_composition_max = pyo.Constraint(model.A, model.C, rule=pipe_composition_max_rule)

    # Unused
    def pipe_composition_min_rule(model, a1, a2, c):
        return model.f[a1, a2,c] >= model.q_minus_arc[a1, a2,c] * sum(model.f[a1, a2, cp] for cp in model.C)

    model.pipe_composition_min = pyo.Constraint(model.A, model.C, rule =pipe_composition_min_rule)

    # Composition at market nodes
    def market_node_max_rule(model, n, c):
        return sum(model.f[a, c] for a in model.A_n_plus[n]) <= model.q_plus_node[n, c] * sum(model.f[a, cp] for a in model.A_n_plus[n] for cp in model.C)
    
    model.market_node_max = pyo.Constraint(model.N_m, model.C, rule=market_node_max_rule)

    # Unused
    def market_node_min_rule(model, n, c):
        return sum(model.f[a, c] for a in model.A_n_plus[n]) >= model.q_minus_node[n, c] * sum(model.f[a, cp] for a in model.A_n_plus[n] for cp in model.C)
    
    model.market_node_min = pyo.Constraint(model.N_m, model.C, rule=market_node_min_rule)

    #_______________
    # Homogeneous splitting constraints
    #_______________

    def SOS1_v(model, n):
        return sum(model.v[n, z] for z in model.Z_v[n]) == 1 

    model.S1_v = pyo.Constraint(model.N_s, rule =SOS1_v)

    def choice_v_flow_rule(model, n, z):
        if n in model.N_s:
            return sum(model.e[n, c, z] for c in model.C) <= model.v[n, z] * model.M_n[n]
        return pyo.Constraint.Skip

    model.choice_v_flow = pyo.Constraint(model.v_index, rule=choice_v_flow_rule)

    def homogeneous_split_rule(model, a1, a2, n, c):
        if (a1,a2) in model.A_n_minus[n] and n in model.N_s:
            return model.f[a1, a2, c] == sum(model.e[n, c, z] * model.E_nza[n, z, a1, a2] for z in model.Z_v[n])
        else:
            return pyo.Constraint.Skip

    model.homogeneous_split = pyo.Constraint(model.A, model.N_s, model.C, rule=homogeneous_split_rule)

    #------------------------------
    # Expressions
    #------------------------------

    # Total flow on each arc
    def total_flow_rule(model, a1,a2):
        return sum(model.f[a1,a2, c] for c in model.C)
    model.total_flow = pyo.Expression(model.A, rule=total_flow_rule)

    # Relative density on each arc
    model.rho = pyo.Expression(model.A, rule=lambda model, a1, a2: sum(model.rho_c[c] * model.f[a1,a2,c] for c in model.C) / (sum(model.f[a1,a2,c] for c in model.C) + 1e-6))  # Adding small value to avoid division by zero

    # Weymouth flow on each arc
    model.weymouth_flow = pyo.Expression(model.A, rule=lambda model, a1, a2: sum(model.theta[a1,a2,z] * model.K_az[a1,a2,z] * (model.p_in[a1,a2]**2 - model.p_out[a1,a2]**2)**0.5 for z in model.Z_theta))

    return model

def solve_model(model):
    solver = pyo.SolverFactory('gurobi')  # You can choose a different solver if needed
    solver.options['OutputFlag'] = 1  # ensures full output
    solver.options['TimeLimit'] = 30  # optional
    results = solver.solve(model, tee=True)  # tee=True to display solver output
    return results

def plot_flow_per_arc(model):
    arcs = [str(a) for a in model.A]
    components = list(model.C)

    fig, ax = plt.subplots(figsize=(10, 6))

    print("Arcs:", arcs)
    bottom = [0] * len(arcs)
    
    for c in components:
        flows = [pyo.value(model.f[a, c]) for a in model.A]
        print(f"Flows for component {c}:", flows)
        ax.bar(arcs, flows, bottom=bottom, alpha=0.7, label=f'Component {c}')
        bottom = [bottom[i] + flows[i] for i in range(len(arcs))]

    ax.set_xlabel("Arc")
    ax.set_ylabel("Flow")
    ax.set_title("Flow per Arc per Component")
    ax.legend()
    plt.savefig("flow_per_arc_plot.png")  # Save the plot as a PNG file
    plt.show()


def plot_inlet_outlet_pressures(model):
    arcs = [str(a) for a in model.A]
    p_in_vals = [pyo.value(model.p_in[a]) for a in model.A]
    p_out_vals = [pyo.value(model.p_out[a]) for a in model.A]

    print("Inlet pressures:", p_in_vals)
    print("Outlet pressures:", p_out_vals)

    fig, ax = plt.subplots(figsize=(10, 6))
    width = 0.35
    x = range(len(arcs))

    ax.bar([i - width/2 for i in x], p_in_vals, width, label='Inlet Pressure', color='skyblue')
    ax.bar([i + width/2 for i in x], p_out_vals, width, label='Outlet Pressure', color='orange')

    ax.set_xticks(x)
    ax.set_xticklabels(arcs)
    ax.set_xlabel("Arc")
    ax.set_ylabel("Pressure")
    ax.set_title("Inlet and Outlet Pressure per Arc")
    ax.legend()
    plt.savefig("inlet_outlet_pressure_plot.png")  # Save the plot as a PNG file
    plt.show()


def plot_relative_density(model):
    arcs = [str(a) for a in model.A]
    rho_vals = [float(pyo.value(model.rho[a])) for a in model.A]
    
    # Calculate the upper bound for each arc based on which theta is active
    rho_upper_vals = []
    for a in model.A:
        for z in model.Z_theta:
            if pyo.value(model.theta[a, z]) > 0.5:  # Binary variable, check if active
                rho_upper_vals.append(float(pyo.value(model.rho_Z[z])))
                break

    print("Relative densities:", rho_vals)
    print("Relative density upper bounds:", rho_upper_vals)

    fig, ax = plt.subplots(figsize=(10, 6))
    width = 0.35
    x = range(len(arcs))
    
    # Use standard color names instead of 'tab:cyan' and 'tab:magenta'
    ax.bar([i - width/2 for i in x], rho_vals, width, label='Relative Density', color='skyblue')
    ax.bar([i + width/2 for i in x], rho_upper_vals, width, label='Density Upper Bound', color='orange')

    ax.set_xticks(x)
    ax.set_xticklabels(arcs)
    ax.set_xlabel("Arc")
    ax.set_ylabel("Relative Density")
    ax.set_title("Relative Density per Arc with Upper Bounds")
    ax.legend()
    plt.savefig("relative_density_plot.png")  # Save the plot as a PNG file
    plt.show()


def total_v_weymouth_flow_plot(model):
    arcs = [str(a) for a in model.A]
    weymouth_flows = [float(pyo.value(model.weymouth_flow[a])) for a in model.A]
    total_flows = [float(pyo.value(model.total_flow[a])) for a in model.A]

    print("Weymouth flows:", weymouth_flows)
    print("Total flows:", total_flows)

    fig, ax = plt.subplots(figsize=(10, 6))
    width = 0.35
    x = range(len(arcs))

    ax.bar([i - width/2 for i in x], weymouth_flows, width, label='Weymouth Flow', color='lightgreen')
    ax.bar([i + width/2 for i in x], total_flows, width, label='Total Flow', color='lightgray')

    ax.set_xticks(x)
    ax.set_xticklabels(arcs)
    ax.set_xlabel("Arc")
    ax.set_ylabel("Flow")
    ax.set_title("Weymouth Flow vs Total Flow per Arc")
    ax.legend()
    plt.savefig("weymouth_vs_total_flow_plot.png")  # Save the plot as a PNG file
    plt.show()

def plot_demand_and_composition_per_node(model):
    """
    Plot demand per supplier and gas composition delivered side-by-side for each demand node.
    """
    demand_nodes = list(model.N_m)
    suppliers = list(model.H)
    components = list(model.C)
    
    # Define consistent colors for suppliers and components
    supplier_colors = {h: plt.cm.Set1(i) for i, h in enumerate(suppliers)}
    component_colors = {c: plt.cm.Set2(i) for i, c in enumerate(components)}
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    bar_width = 0.35
    x_pos = 0
    x_ticks = []
    x_labels = []
    
    for n in demand_nodes:
        # Plot 1: Demand per supplier (stacked)
        demand_values = [pyo.value(model.D[h, n]) for h in suppliers]
        
        bottom = 0
        for h in suppliers:
            demand = pyo.value(model.D[h, n])
            ax.bar(x_pos, demand, bar_width, bottom=bottom, 
                   color=supplier_colors[h], label=f'Supplier {h}' if demand_nodes.index(n) == 0 else "")
            bottom += demand
        
        x_ticks.append(x_pos)
        x_labels.append(f'{n}\n(Demand)')
        x_pos += bar_width + 0.05
        
        # Plot 2: Composition delivered (stacked)
        composition_values = {c: pyo.value(sum(model.f[a, c] for a in model.A_n_plus[n])) for c in components}
        
        bottom = 0
        for c in components:
            ax.bar(x_pos, composition_values[c], bar_width, bottom=bottom, 
                   color=component_colors[c], label=f'Component {c}' if demand_nodes.index(n) == 0 else "")
            bottom += composition_values[c]
        
        x_ticks.append(x_pos)
        x_labels.append(f'{n}\n(Composition)')
        
        x_pos += bar_width + 0.3
    
    ax.set_ylabel("Flow")
    ax.set_title("Demand vs Composition Delivered at Each Node")
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("demand_and_composition_per_node_plot.png")
    plt.show()


def plot_results(model):
    plot_flow_per_arc(model)
    plot_inlet_outlet_pressures(model)
    plot_relative_density(model)
    total_v_weymouth_flow_plot(model)
    plot_demand_and_composition_per_node(model)


if __name__ == "__main__":
    G = example_graph()
    model = create_base_model(G)

    results = solve_model(model)
    print(results)
    plot_results(model)