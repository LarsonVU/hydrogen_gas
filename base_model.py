from network import example_graph, plot_network
import pyomo.environ as pyo
import networkx

NUMBER_OF_HOMOGENEOUS_SPLITS =10 
NUMBER_OF_CUTTING_PLANES = 9
NUMBER_OF_PRESSURE_BOUNDS = 4

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
        initialize=[n for n, d in network.nodes(data=True) if "compression_max" in d]
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
    
    model.Z_theta = pyo.Set(initialize = range(NUMBER_OF_PRESSURE_BOUNDS))  # Type 1, pressure upper bound
    model.Z_v = pyo.Set(initialize = range(NUMBER_OF_HOMOGENEOUS_SPLITS))      # Type 1, homogeneous splitting

    # Other sets
    model.L = pyo.Set()        # Set of cutting planes
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

    # Demand parameters
    
    # Parameters for demand
    def demand_rule(m, h, n):
        return network.nodes[n].get('demand', {}).get(h, 0)

    model.D = pyo.Param(
        model.H,
        model.N_m,
        initialize=demand_rule,
        mutable=True
    )
    # -----------------------------
    # Variables
    #-----------------------------
    model.f = pyo.Var(model.A, model.C, within=pyo.NonNegativeReals)  # Flow on arc a of component c
    model.p_in = pyo.Var(model.A, within=pyo.NonNegativeReals)  # Inlet pressure on arc a
    model.p_out = pyo.Var(model.A, within=pyo.NonNegativeReals)  # Outlet pressure on arc a
    model.theta = pyo.Var(model.A, model.Z_theta, within=pyo.Binary)  # Binary variables for pressure bounds
    model.v = pyo.Var(model.N_s, model.Z_v, within=pyo.Binary)  # Binary variables for homogeneous splitting
    
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
            return sum(model.f[a, c] for a in model.A_n_minus[n] for c in model.C) <= model.G[n]
        return pyo.Constraint.Skip
    model.production_capacity = pyo.Constraint(model.N_hg, rule=production_capacity_rule)

    # Constraint 2: Component ratio constraint (byproduct/stoichiometric)
    def component_ratio_rule(model, n, c):
        if n in model.N_hg and c in model.C:
            return sum(model.f[a, c] for a in model.A_n_minus[n]) == model.alpha[n, c] * sum(model.f[a, cp] for a in model.A_n_minus[n] for cp in model.C)
        return pyo.Constraint.Skip
    model.component_ratio = pyo.Constraint(model.N_hg, model.C, rule=component_ratio_rule)

    # Constraint 3: Demand satisfaction constraint for production nodes
    def demand_satisfaction_production_rule(model, n, h):
        if n in model.N_hg:
            return sum(model.f[a, c] for a in model.A_n_minus[n] for c in model.C) >= model.D[h,n]
        return pyo.Constraint.Skip
    model.demand_satisfaction_production = pyo.Constraint(model.N_m, model.H, rule=demand_satisfaction_production_rule)

    # Constraint 4: Demand satisfaction constraint for market nodes
    def demand_satisfaction_market_rule(model, n):
        if n in model.N_m:
            return sum(model.f[a, c] for a in model.A_n_plus[n] for c in model.C) >= sum(model.D[h,n] for h in model.H)
        return pyo.Constraint.Skip
    model.demand_satisfaction_market = pyo.Constraint(model.N_m, rule=demand_satisfaction_market_rule)

    # Constraint 5: Flow balance constraint for intermediate nodes
    def flow_balance_rule(model, n, c):
        # Applies to nodes that are not production, market, or compression nodes
        intermediate_nodes = model.N - model.N_hg - model.N_m #- model.N_gamma
        if n in intermediate_nodes:
            inflow = sum(model.f[a, c] for a in model.A_n_plus[n])
            outflow = sum(model.f[a, c] for a in model.A_n_minus[n])
            return inflow == outflow
        return pyo.Constraint.Skip

    model.flow_balance = pyo.Constraint(model.N, model.C, rule=flow_balance_rule)

    # ________________
    # Pressure constraints
    # ________________


    


    return model

def solve_model(model):
    solver = pyo.SolverFactory('gurobi')  # You can choose a different solver if needed
    solver.options['OutputFlag'] = 1  # ensures full output
    solver.options['TimeLimit'] = 30  # optional
    results = solver.solve(model, tee=True)  # tee=True to display solver output
    return results



if __name__ == "__main__":
    G = example_graph()
    model = create_base_model(G)

    results = solve_model(model)
    print(results)