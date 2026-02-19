from network import example_graph, plot_network
import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use('Agg')  # Ensure it works in Codespaces terminal
import pyomo.environ as pyo
import networkx
import numpy as np
import itertools

folder = "Figures/"
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

def build_sets(model, network):

    model.N = pyo.Set(initialize=list(network.nodes))
    model.A = pyo.Set(initialize=list(network.edges), dimen=2)

    model.N_hg = pyo.Set(
        within=model.N,
        initialize=[n for n, d in network.nodes(data=True) if "supply_capacity" in d]
    )

    model.N_m = pyo.Set(
        within=model.N,
        initialize=[n for n, d in network.nodes(data=True) if "demand" in d]
    )

    model.N_gamma = pyo.Set(
        within=model.N,
        initialize=[n for n, d in network.nodes(data=True) if "compression_increase" in d]
    )

    model.N_s = pyo.Set(
        within=model.N,
        initialize=[n for n, d in network.nodes(data=True) if d.get("split_homogeneous", False)]
    )

    # Arc incidence sets
    def A_in_rule(m, n):
        return [a for a in m.A if a[1] == n]

    def A_out_rule(m, n):
        return [a for a in m.A if a[0] == n]

    model.A_n_plus = pyo.Set(model.N, initialize=A_in_rule)
    model.A_n_minus = pyo.Set(model.N, initialize=A_out_rule)

    model.C = pyo.Set(initialize=['NG', 'CO2', 'H2'])

    model.Z_theta = pyo.Set(initialize=range(NUMBER_OF_DENSITY_BOUNDS))
    model.L = pyo.Set(initialize=range(len(pairs)))

    # Suppliers
    model.H = pyo.Set(initialize=list(set().union(
        *(network.nodes[n].get('demand', {}).keys() for n in model.N_m)
    )))

    E_nz_init = {}

    # Homogeneous splitting parameters
    for n, arcs in model.A_n_minus.items():
        if n in model.N_s:
            order_arcs = len(arcs)
            E_nz_init[n] = [z for z in itertools.product(splits_per_arc, repeat=order_arcs) if sum(z) == 1]

    model.E_nz_init = E_nz_init

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

def build_parameters(model, network):

    # Prices and costs
    model.o_n_d = pyo.Param(model.N, initialize={n: network.nodes[n]['price'] for n in model.N_m})
    model.o_n_g = pyo.Param(model.N, initialize={n: network.nodes[n]['generation_cost'] for n in model.N_hg})
    model.o_p_a = pyo.Param(model.A, initialize={a: network.edges[a]['pressure_cost'] for a in model.A})

    # Production
    model.G = pyo.Param(model.N_hg, initialize={n: network.nodes[n]['supply_capacity'] for n in model.N_hg})
    model.alpha = pyo.Param(
        model.N_hg, model.C,
        initialize={(n, c): network.nodes[n].get('component_ratio', {}).get(c, 0)
                    for n in model.N_hg for c in model.C}
    )

    # Demand
    def demand_rule(m, h, n):
        return network.nodes[n].get('demand', {}).get(h, 0)

    model.D = pyo.Param(model.H, model.N_m, initialize=demand_rule)

    model.supplier = pyo.Param(
        model.N_hg,
        initialize={n: network.nodes[n].get('supplier', 'Unknown') for n in model.N_hg}
    )

    # Big-M
    model.M_n = pyo.Param(model.N, initialize={n: network.nodes[n]['max_flow'] for n in model.N})
    model.M_a = pyo.Param(model.A, initialize={a: network.edges[a]['max_flow'] for a in model.A})

    # Density
    model.rho_c = pyo.Param(model.C, initialize={"NG": 0.65, "CO2": 1.53, "H2": 0.07})

    rho_values = np.linspace(RHO_LOW, RHO_HIGH, NUMBER_OF_DENSITY_BOUNDS + 1)[1:]
    model.rho_Z = pyo.Param(model.Z_theta, initialize={z: rho_values[i] for i, z in enumerate(model.Z_theta)})

    # Weymouth
    model.K_a = pyo.Param(model.A, initialize={a: network.edges[a]['weymouth_constant'] for a in model.A})

    def K_az_rule(m, a1, a2, z):
        return m.K_a[a1, a2] / (m.rho_Z[z] ** 0.5)

    model.K_az = pyo.Param(model.A, model.Z_theta, initialize=K_az_rule)

    # Cutting planes
    model.P_in = pyo.Param(model.L, initialize={l: pairs[l][0] for l in range(len(pairs))})
    model.P_out = pyo.Param(model.L, initialize={l: pairs[l][1] for l in range(len(pairs))})

    # Pressure bounds
    model.P_max = pyo.Param(model.A, initialize={a: network.edges[a].get('max_inlet_pressure', 1000) for a in model.A})
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
        return m.E_nz_init[n][z][arc_idx]

    model.E_nza = pyo.Param(model.E_index, initialize=E_init)



def build_variables(model):

    model.f = pyo.Var(model.A, model.C, within=pyo.NonNegativeReals, initialize=0)
    model.p_in = pyo.Var(model.A, within=pyo.NonNegativeReals, initialize=0)
    model.p_out = pyo.Var(model.A, within=pyo.NonNegativeReals, initialize=0)

    model.theta = pyo.Var(model.A, model.Z_theta, within=pyo.Binary, initialize=0)

    # Compression
    model.w = pyo.Var(model.N_gamma, model.C, within=pyo.NonNegativeReals, initialize=0)

    # Splitting
    model.v = pyo.Var(model.v_index, within=pyo.Binary, initialize=0)
    model.e = pyo.Var(model.e_index, domain=pyo.NonNegativeReals, initialize=0)


def add_expressions(model):

    # Total flow
    def total_flow_rule(m, a1, a2):
        return sum(m.f[a1, a2, c] for c in m.C)

    model.total_flow = pyo.Expression(model.A, rule=total_flow_rule)

    # Relative density
    def rho_rule(m, a1, a2):
        num = sum(m.rho_c[c] * m.f[a1, a2, c] for c in m.C)
        den = sum(m.f[a1, a2, c] for c in m.C) + 1e-6
        return num / den

    model.rho = pyo.Expression(model.A, rule=rho_rule)

    # Weymouth flow
    def weymouth_expr(m, a1, a2):
        return sum(
            m.theta[a1, a2, z]
            * m.K_az[a1, a2, z]
            * (m.p_in[a1, a2]**2 - m.p_out[a1, a2]**2)**0.5
            for z in m.Z_theta
        )

    model.weymouth_flow = pyo.Expression(model.A, rule=weymouth_expr)


def add_objective(model):

    def objective_rule(m):
        revenue = sum(
            m.o_n_d[n] * sum(m.f[a, c] for a in m.A_n_plus[n] for c in m.C)
            for n in m.N_m
        )

        generation_cost = sum(
            m.o_n_g[n] * sum(m.f[a, c] for a in m.A_n_minus[n] for c in m.C)
            for n in m.N_hg
        )

        pressure_cost = sum(m.o_p_a[a] * m.p_in[a] for a in m.A)

        return revenue - generation_cost - pressure_cost

    model.objective = pyo.Objective(rule=objective_rule, sense=pyo.maximize)


def add_flow_constraints(model):

    # Production capacity
    def production_capacity_rule(m, n):
        if n in m.N_hg:
            return (
                sum(m.f[a, c] for a in m.A_n_minus[n] for c in m.C)
                - sum(m.f[a, c] for a in m.A_n_plus[n] for c in m.C)
                <= m.G[n]
            )
        return pyo.Constraint.Skip

    model.production_capacity = pyo.Constraint(model.N_hg, rule=production_capacity_rule)

    # Component ratio
    def component_ratio_rule(m, n, c):
        if n in m.N_hg:
            total = (
                sum(m.f[a, cp] for a in m.A_n_minus[n] for cp in m.C)
                - sum(m.f[a, cp] for a in m.A_n_plus[n] for cp in m.C)
            )
            return (
                sum(m.f[a, c] for a in m.A_n_minus[n])
                - sum(m.f[a, c] for a in m.A_n_plus[n])
                == m.alpha[n, c] * total
            )
        return pyo.Constraint.Skip

    model.component_ratio = pyo.Constraint(model.N_hg, model.C, rule=component_ratio_rule)

    # Supplier demand
    def demand_satisfaction_production_rule(m, h):
        lhs = sum(
            m.f[a, c]
            for c in m.C
            for n in m.N_hg if m.supplier[n] == h
            for a in m.A_n_minus[n]
        ) - sum(
            m.f[a, c]
            for c in m.C
            for n in m.N_hg if m.supplier[n] == h
            for a in m.A_n_plus[n]
        )
        rhs = sum(m.D[h, n] for n in m.N_m)
        return lhs >= rhs

    model.demand_satisfaction_production = pyo.Constraint(model.H, rule=demand_satisfaction_production_rule)

    # Market demand
    def demand_satisfaction_market_rule(m, n):
        if n in m.N_m:
            return sum(m.f[a, c] for a in m.A_n_plus[n] for c in m.C) >= sum(m.D[h, n] for h in m.H)
        return pyo.Constraint.Skip

    model.demand_satisfaction_market = pyo.Constraint(model.N_m, rule=demand_satisfaction_market_rule)

    # Flow balance
    def flow_balance_rule(m, n, c):
        intermediate = m.N - m.N_hg - m.N_m - m.N_gamma
        if n in intermediate:
            inflow = sum(m.f[a, c] for a in m.A_n_plus[n])
            outflow = sum(m.f[a, c] for a in m.A_n_minus[n])
            return inflow == outflow
        return pyo.Constraint.Skip

    model.flow_balance = pyo.Constraint(model.N, model.C, rule=flow_balance_rule)


def add_weymouth_constraints(model):

    # SOS1 density regime
    def SOS1_theta(m, a1, a2):
        return sum(m.theta[a1, a2, z] for z in m.Z_theta) == 1

    model.S1_theta = pyo.Constraint(model.A, rule=SOS1_theta)

    # Density upper bound
    def upperbound_rho_rule(m, a1, a2, z):
        total_flow = sum(m.f[a1, a2, c] for c in m.C)
        weighted_flow = sum(m.rho_c[c] * m.f[a1, a2, c] for c in m.C)

        return (
            m.rho_Z[z] * total_flow
            + m.M_a[a1, a2] * (1 - m.theta[a1, a2, z])
            >= weighted_flow
        )

    model.upperbound_rho = pyo.Constraint(model.A, model.Z_theta, rule=upperbound_rho_rule)

    # Cutting planes
    def weymouth_cutting_plane_rule(m, a1, a2, z, l):
        total_flow = sum(m.f[a1, a2, c] for c in m.C)

        p_in_l = m.P_in[l]
        p_out_l = m.P_out[l]
        denom = pyo.sqrt(p_in_l**2 - p_out_l**2)

        coeff_in = p_in_l / denom
        coeff_out = p_out_l / denom

        return (
            total_flow
            <= m.K_az[a1, a2, z] * (coeff_in * m.p_in[a1, a2] - coeff_out * m.p_out[a1, a2])
            + m.M_a[a1, a2] * (1 - m.theta[a1, a2, z])
        )

    model.weymouth_cutting_plane = pyo.Constraint(model.A, model.Z_theta, model.L, rule=weymouth_cutting_plane_rule)


def add_pressure_constraints(model):

    # Node pressure propagation
    def node_pressure_rule(m, n, a_in_1, a_in_2, a_out_1, a_out_2):
        return m.p_in[a_out_1, a_out_2] <= m.p_out[a_in_1,a_in_2]

    model.node_pressure = pyo.Constraint(
        ((n, a_in, a_out)
         for n in model.N if n not in model.N_gamma
         for a_in in model.A_n_plus[n]
         for a_out in model.A_n_minus[n]),
        rule=node_pressure_rule
    )

    # Pressure drop
    model.pressure_drop = pyo.Constraint(model.A, rule=lambda m, a1, a2: m.p_in[a1, a2] >= m.p_out[a1, a2])

    # Max pressure
    model.max_pressure = pyo.Constraint(model.A, rule=lambda m, a1, a2: m.p_in[a1, a2] <= m.P_max[a1, a2])

    # Min pressure at markets
    def min_pressure_demand_rule(m, n, a1,a2):
        return m.p_out[a1, a2] >= m.P_min[n]

    model.min_pressure_demand = pyo.Constraint(
        ((n, a) for n in model.N_m for a in model.A_n_plus[n]),
        rule=min_pressure_demand_rule
    )


def add_compression_constraints(model):

    def fuel_consumption_rule(m, n, c):
        if n in m.N_gamma:
            a = m.A_n_minus[n].first()
            a_prime = m.A_n_plus[n].first()
            return (
                m.w[n, c]
                == m.K_out_pipe[n, c] * m.p_in[a]
                - m.K_into_pipe[n, c] * m.p_out[a_prime]
                + m.K_flow[n, c] * m.f[a, c]
            )
        return pyo.Constraint.Skip

    model.fuel_consumption = pyo.Constraint(model.N_gamma, model.C, rule=fuel_consumption_rule)

    def fuel_flow_balance_rule(m, n, c):
        if n in m.N_gamma:
            return sum(m.f[a, c] for a in m.A_n_minus[n]) == sum(m.f[a, c] for a in m.A_n_plus[n]) - m.w[n, c]
        return pyo.Constraint.Skip

    model.fuel_flow_balance = pyo.Constraint(model.N_gamma, model.C, rule=fuel_flow_balance_rule)

    def compression_increase_rule(m, n):
        if n in m.N_gamma:
            a = m.A_n_minus[n].first()
            a_prime = m.A_n_plus[n].first()
            return m.p_in[a] == m.P_hat[n] * m.p_out[a_prime]
        return pyo.Constraint.Skip

    model.compression_increase = pyo.Constraint(model.N_gamma, rule=compression_increase_rule)


def add_quality_constraints(model):

    # Pipe composition
    def pipe_composition_max_rule(m, a1, a2, c):
        return m.f[a1, a2, c] <= m.q_plus_arc[a1, a2, c] * sum(m.f[a1, a2, cp] for cp in m.C)

    model.pipe_composition_max = pyo.Constraint(model.A, model.C, rule=pipe_composition_max_rule)

    # Market composition
    def market_node_max_rule(m, n, c):
        return (
            sum(m.f[a, c] for a in m.A_n_plus[n])
            <= m.q_plus_node[n, c] * sum(m.f[a, cp] for a in m.A_n_plus[n] for cp in m.C)
        )

    model.market_node_max = pyo.Constraint(model.N_m, model.C, rule=market_node_max_rule)


def add_homogeneous_splitting_constraints(model):

    def SOS1_v(m, n):
        return sum(m.v[n, z] for z in m.Z_v[n]) == 1

    model.S1_v = pyo.Constraint(model.N_s, rule=SOS1_v)

    def choice_v_flow_rule(m, n, z):
        return sum(m.e[n, c, z] for c in m.C) <= m.v[n, z] * m.M_n[n]

    model.choice_v_flow = pyo.Constraint(model.v_index, rule=choice_v_flow_rule)

    def homogeneous_split_rule(m, a1, a2, n, c):
        if (a1, a2) in m.A_n_minus[n]:
            return m.f[a1, a2, c] == sum(m.e[n, c, z] * m.E_nza[n, z, a1, a2] for z in m.Z_v[n])
        return pyo.Constraint.Skip

    model.homogeneous_split = pyo.Constraint(model.A, model.N_s, model.C, rule=homogeneous_split_rule)

def add_constraints(model):
    add_flow_constraints(model)
    add_weymouth_constraints(model)
    add_pressure_constraints(model)
    add_compression_constraints(model)
    add_quality_constraints(model)
    add_homogeneous_splitting_constraints(model)

def create_base_model(network: networkx.Graph):
    model = pyo.ConcreteModel()

    build_sets(model, network)
    build_parameters(model, network)
    build_variables(model)

    add_constraints(model)

    add_expressions(model)
    add_objective(model)

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
    plt.savefig(folder + "flow_per_arc_plot.png")  # Save the plot as a PNG file
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
    plt.savefig(folder + "inlet_outlet_pressure_plot.png")  # Save the plot as a PNG file
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
    plt.savefig(folder + "relative_density_plot.png")  # Save the plot as a PNG file
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

    ax.bar([i - width/2 for i in x], weymouth_flows, width, label='Weymouth Flow', color='skyblue')
    ax.bar([i + width/2 for i in x], total_flows, width, label='Total Flow', color='lightgreen')

    ax.set_xticks(x)
    ax.set_xticklabels(arcs)
    ax.set_xlabel("Arc")
    ax.set_ylabel("Flow")
    ax.set_title("Weymouth Flow vs Total Flow per Arc")
    ax.legend()
    plt.savefig(folder + "weymouth_vs_total_flow_plot.png")  # Save the plot as a PNG file
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
    plt.savefig(folder + "demand_and_composition_per_node_plot.png")
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