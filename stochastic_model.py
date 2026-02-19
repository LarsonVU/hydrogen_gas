from problem_file import build_base_graph, create_scenarios
import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use('Agg')  # Ensure it works in Codespaces terminal
import pyomo.environ as pyo
import networkx
import numpy as np
import itertools


NUMBER_OF_STAGES = 3
BRANCHES_PER_STAGE = {1 : 1, 2 : 4, 3: 4}
ALLOWED_DEVIATION = 0  # x% deviation from nominal values for scenarios

NUMBER_OF_DENSITY_BOUNDS = 10
RHO_LOW = 0.55
RHO_HIGH = 0.70

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
    cutting_plane_pairs = [
        (p_in, p_out)
        for p_out, p_in_list in zip(p_out_values, p_in_grid)
        for p_in in p_in_list
        if p_in > p_out
    ]
    
    return cutting_plane_pairs

# Usage
cutting_plane_pairs = generate_cutting_plane_pairs()

def build_sets(model, network, scenarios, cutting_plane_pairs, splits_per_arc):
    model.N = pyo.Set(initialize=list(network.nodes))
    model.A = pyo.Set(initialize=list(network.edges), dimen=2)

    model.N_hg = pyo.Set(
        within=model.N,
        initialize=[n for n, d in network.nodes(data=True) if "supply_capacity" in d]
    )

    model.N_m = pyo.Set(
        within=model.N,
        initialize=[n for n, d in network.nodes(data=True) if "max_fractions" in d]
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
    model.L = pyo.Set(initialize=range(len(cutting_plane_pairs)))

    # Suppliers
    model.H = pyo.Set(initialize=list(set().union(
        *(scenarios[NUMBER_OF_STAGES][0].G.nodes[n].get('demand', {}).keys() for n in model.N_m)
    )))

    E_nz_init = {}

    # Homogeneous splitting parameters
    for n, arcs in model.A_n_minus.items():
        if n in model.N_s:
            order_arcs = len(arcs)
            E_nz_init[n] = [z for z in itertools.product(splits_per_arc, repeat=order_arcs) if sum(z) == 1]

    model.E_nz_init = E_nz_init

    model.Z_v = pyo.Set(model.N_s, initialize = lambda model ,n: range(len(E_nz_init[n])) if n in E_nz_init else range(0))      # Type 1, homogeneous splitting

    # Set of stages for stochastic programming
    model.K = pyo.Set(initialize=scenarios.keys())

    def M_rule(model, k):
        return [m.index for m in scenarios[k]]

    # Set of scenarios per stage 
    model.M = pyo.Set(model.K, initialize=M_rule)

    # Auxiliary set for defining scenario-specific parameters and variables
    def KM_init(model):
        return ((k, m) for k in model.K for m in model.M[k])

    model.KM = pyo.Set(dimen=2, initialize=KM_init)

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


def build_parameters(model, network, scenarios, cutting_plane_pairs, allowed_deviation=ALLOWED_DEVIATION):
    # Prices and costs
    model.o_n_d = pyo.Param(model.N_m, model.M[3], initialize=lambda model, n, m: scenarios[3][m-1].G.nodes[n]['price'] if n in scenarios[3][m-1].G.nodes else 0)
    model.o_n_g = pyo.Param(model.N_hg, model.M[3], initialize=lambda model, n, m: scenarios[3][m-1].G.nodes[n]['generation_cost'] if n in scenarios[3][m-1].G.nodes else 0)
    model.o_p_a = pyo.Param(model.A, initialize={a: network.edges[a]['pressure_cost'] for a in model.A})
    model.o_n_x = pyo.Param(model.N, model.KM, initialize=lambda model, n, k, m: scenarios[k][m-1].G.nodes[n].get('booking_cost', 0) if n in scenarios[k][m-1].G.nodes else 0)
    

    # Production
    model.G = pyo.Param(model.N_hg, initialize={n: network.nodes[n]['supply_capacity'] for n in model.N_hg})
    model.alpha = pyo.Param(
        model.N_hg, model.C,
        initialize={(n, c): network.nodes[n].get('component_ratio', {}).get(c, 0)
                    for n in model.N_hg for c in model.C}
    )

    # Demand
    def demand_rule(m, h, n, m_index):
        return scenarios[NUMBER_OF_STAGES][m_index-1].G.nodes[n].get('demand', {}).get(h, 0)

    model.D = pyo.Param(model.H, model.N_m, model.M[3], initialize=demand_rule)

    model.supplier = pyo.Param(
        model.N_hg,
        initialize={n: network.nodes[n].get('supplier', 'Unknown') for n in model.N_hg},
        within=pyo.Any
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
    model.P_in = pyo.Param(model.L, initialize={l: cutting_plane_pairs[l][0] for l in range(len(cutting_plane_pairs))})
    model.P_out = pyo.Param(model.L, initialize={l: cutting_plane_pairs[l][1] for l in range(len(cutting_plane_pairs))})

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
    model.q_minus_arc = pyo.Param(model.A, model.C, initialize=lambda model, a1,a2, c: 0)

    # Node quality constraints
    model.q_plus_node = pyo.Param(model.N_m, model.C, initialize=lambda model, n, c: network.nodes[n].get('max_fractions', {}).get(c, 0))
    model.q_minus_node = pyo.Param(model.N_m, model.C, initialize=lambda model, n, c: 0)

    #  Homogeneous splits
    def E_init(m, n, z, a1,a2):
        arc_idx = list(m.A_n_minus[n]).index((a1,a2))
        return m.E_nz_init[n][z][arc_idx]

    model.E_nza = pyo.Param(model.E_index, initialize=E_init)


    # Predecessor scenario for each scenario (for non-anticipativity constraints)
    def predecessor_init(model, k, m):
        return scenarios[k][m-1].predecessor.index if scenarios[k][m-1].predecessor is not None else 1

    model.pred = pyo.Param(model.KM, initialize=predecessor_init, within=pyo.PositiveIntegers)

    def predecessor_chain_init(model, k, m):
        current_k, current_m = k, m
        chain = [(current_k, current_m)]
        while current_k > 1:
            current_m = model.pred[current_k, current_m]
            current_k -= 1
            chain.append((current_k, current_m))
        return chain

    model.pred_chain = pyo.Param(model.KM, initialize=predecessor_chain_init, within=pyo.Any)

    # Probability of each scenario
    model.sp = pyo.Param(model.KM, initialize=lambda model, k, m: scenarios[k][m-1].probability)

    # Allowed deviation for quality constraints
    model.C_deviation = int(allowed_deviation *len(model.M[3]))  # Max total deviation across all scenarios in final stage

def build_variables(model):
    model.f = pyo.Var(model.A, model.C, model.M[3], within=pyo.NonNegativeReals, initialize=0)

    # Weymouth
    model.p_in = pyo.Var(model.A, model.M[3], within=pyo.NonNegativeReals, initialize=0)
    model.p_out = pyo.Var(model.A, model.M[3], within=pyo.NonNegativeReals, initialize=0)
    model.theta = pyo.Var(model.A, model.Z_theta, model.M[3], within=pyo.Binary, initialize=0)

    # Compression
    model.w = pyo.Var(model.N_gamma, model.C, model.M[3], within=pyo.NonNegativeReals, initialize=0)

    # Splitting
    model.v = pyo.Var(model.v_index, model.M[3], within=pyo.Binary, initialize=0)
    model.e = pyo.Var(model.e_index, model.M[3], domain=pyo.NonNegativeReals, initialize=0)

    # Booking decisions 
    model.x_entry = pyo.Var(model.N, model.KM, within=pyo.NonNegativeReals, initialize=0)
    model.x_exit = pyo.Var(model.N, model.KM, within=pyo.NonNegativeReals, initialize=0)

    # Quality deviations
    model.delta = pyo.Var(model.M[3], within=pyo.Binary, initialize=0)

def add_expressions(model):

    # Total flow
    def total_flow_rule(m, a1, a2, m_3):
        return sum(m.f[a1, a2, c, m_3] for c in m.C)

    model.total_flow = pyo.Expression(model.A, model.M[3], rule=total_flow_rule)

    # Average total flow across scenarios for each arc
    def avg_total_flow_rule(m, a1,a2):
        return sum(m.sp[3,m_3] * m.total_flow[a1,a2,m_3] for m_3 in m.M[3]) 
    
    model.avg_total_flow = pyo.Expression(model.A, rule=avg_total_flow_rule)


def add_objective(model):

    def objective_rule(m):
        revenue = sum(
            m.sp[3,m_3] * m.o_n_d[n, m_3] * sum(m.f[a, c, m_3] for a in m.A_n_plus[n] for c in m.C)
            for n in m.N_m for m_3 in m.M[3]
        )

        generation_cost = sum(
            m.sp[3,m_3] * m.o_n_g[n, m_3] * sum(m.f[a, c, m_3] for a in m.A_n_minus[n] for c in m.C)
            for n in m.N_hg for m_3 in m.M[3]
        )

        pressure_cost =  sum(m.sp[3,m_3] * m.o_p_a[a] * m.p_in[a, m_3] for a in m.A for m_3 in m.M[3])

        booking_cost = sum(
            m.sp[k,m_k] * m.o_n_x[n, k, m_k] * (m.x_entry[n, k, m_k] + m.x_exit[n, k, m_k])
            for n in m.N for k in m.K for m_k in m.M[k])

        return revenue - generation_cost - pressure_cost - booking_cost

    model.objective = pyo.Objective(rule=objective_rule, sense=pyo.maximize)


def add_flow_constraints(model):

    # Production capacity
    def production_capacity_rule(m, n, m_3):
        if n in m.N_hg:
            return (
                sum(m.f[a, c, m_3] for a in m.A_n_minus[n] for c in m.C)
                - sum(m.f[a, c, m_3] for a in m.A_n_plus[n] for c in m.C)
                <= m.G[n]
            )
        return pyo.Constraint.Skip

    model.production_capacity = pyo.Constraint(model.N_hg, model.M[3], rule=production_capacity_rule)

    # Component ratio
    def component_ratio_rule(m, n, c, m_3):
        if n in m.N_hg:
            total = (
                sum(m.f[a, cp, m_3] for a in m.A_n_minus[n] for cp in m.C)
                - sum(m.f[a, cp, m_3] for a in m.A_n_plus[n] for cp in m.C)
            )
            return (
                sum(m.f[a, c, m_3] for a in m.A_n_minus[n])
                - sum(m.f[a, c, m_3] for a in m.A_n_plus[n])
                == m.alpha[n, c] * total
            )
        return pyo.Constraint.Skip

    model.component_ratio = pyo.Constraint(model.N_hg, model.C, model.M[3], rule=component_ratio_rule)

    # Supplier demand
    def demand_satisfaction_production_rule(m, h, m_3):
        lhs = sum(
            m.f[a, c, m_3]
            for c in m.C
            for n in m.N_hg if m.supplier[n] == h
            for a in m.A_n_minus[n]
        ) - sum(
            m.f[a, c, m_3]
            for c in m.C
            for n in m.N_hg if m.supplier[n] == h
            for a in m.A_n_plus[n]
        )
        rhs = sum(m.D[h, n, m_3] for n in m.N_m)
        return lhs >= rhs

    model.demand_satisfaction_production = pyo.Constraint(model.H, model.M[3], rule=demand_satisfaction_production_rule)

    # Market demand
    def demand_satisfaction_market_rule(m, n, m_3):
        if n in m.N_m:
            return sum(m.f[a, c, m_3] for a in m.A_n_plus[n] for c in m.C) >= sum(m.D[h, n, m_3] for h in m.H)

        return pyo.Constraint.Skip

    model.demand_satisfaction_market = pyo.Constraint(model.N_m, model.M[3], rule=demand_satisfaction_market_rule)

    # Flow balance
    def flow_balance_rule(m, n, c, m_3):
        intermediate = m.N - m.N_hg - m.N_m - m.N_gamma
        if n in intermediate:
            inflow = sum(m.f[a, c, m_3] for a in m.A_n_plus[n])
            outflow = sum(m.f[a, c, m_3] for a in m.A_n_minus[n])
            return inflow == outflow
        return pyo.Constraint.Skip

    model.flow_balance = pyo.Constraint(model.N, model.C, model.M[3], rule=flow_balance_rule)


def add_weymouth_constraints(model):

    # SOS1 density regime
    def SOS1_theta(m, a1, a2, m_3):
        return sum(m.theta[a1, a2, z, m_3] for z in m.Z_theta) == 1

    model.S1_theta = pyo.Constraint(model.A, model.M[3], rule=SOS1_theta)

    # Density upper bound
    def upperbound_rho_rule(m, a1, a2, z, m_3):
        total_flow = sum(m.f[a1, a2, c, m_3] for c in m.C)
        weighted_flow = sum(m.rho_c[c] * m.f[a1, a2, c, m_3] for c in m.C)

        return (
            m.rho_Z[z] * total_flow
            + m.M_a[a1, a2] * (1 - m.theta[a1, a2, z, m_3])
            >= weighted_flow
        )

    model.upperbound_rho = pyo.Constraint(model.A, model.Z_theta, model.M[3], rule=upperbound_rho_rule)

    # Cutting planes
    def weymouth_cutting_plane_rule(m, a1, a2, z, l, m_3):
        total_flow = sum(m.f[a1, a2, c, m_3] for c in m.C)

        p_in_l = m.P_in[l]
        p_out_l = m.P_out[l]
        denom = pyo.sqrt(p_in_l**2 - p_out_l**2)

        coeff_in = p_in_l / denom
        coeff_out = p_out_l / denom

        return (
            total_flow
            <= m.K_az[a1, a2, z] * (coeff_in * m.p_in[a1, a2, m_3] - coeff_out * m.p_out[a1, a2, m_3])
            + m.M_a[a1, a2] * (1 - m.theta[a1, a2, z, m_3])
        )

    model.weymouth_cutting_plane = pyo.Constraint(model.A, model.Z_theta, model.L, model.M[3], rule=weymouth_cutting_plane_rule)


def add_pressure_constraints(model):

    # Node pressure propagation
    def node_pressure_rule(m, n, a_in_1, a_in_2, a_out_1, a_out_2, m_3):
        return m.p_in[a_out_1, a_out_2, m_3] <= m.p_out[a_in_1,a_in_2, m_3]

    model.node_pressure = pyo.Constraint(
        ((n, a_in, a_out, m_3)
         for n in model.N if n not in model.N_gamma
         for a_in in model.A_n_plus[n]
         for a_out in model.A_n_minus[n]
         for m_3 in model.M[3]),
        rule=node_pressure_rule
    )

    # Pressure drop
    model.pressure_drop = pyo.Constraint(model.A, model.M[3], rule=lambda m, a1, a2, m_3: m.p_in[a1, a2, m_3] >= m.p_out[a1, a2, m_3])

    # Max pressure
    model.max_pressure = pyo.Constraint(model.A, model.M[3], rule=lambda m, a1, a2, m_3: m.p_in[a1, a2, m_3] <= m.P_max[a1, a2])

    # Min pressure at markets
    def min_pressure_demand_rule(m, n, a1,a2, m_3):
        return m.p_out[a1, a2, m_3] >= m.P_min[n]

    model.min_pressure_demand = pyo.Constraint(
        ((n, a, m_3) for n in model.N_m for a in model.A_n_plus[n] for m_3 in model.M[3]),
        rule=min_pressure_demand_rule
    )


def add_compression_constraints(model):

    def fuel_consumption_rule(m, n, c, m_3):
        if n in m.N_gamma:
            a = m.A_n_minus[n].first()
            a_prime = m.A_n_plus[n].first()
            return (
                m.w[n, c, m_3]
                == m.K_out_pipe[n, c] * m.p_in[a, m_3]
                - m.K_into_pipe[n, c] * m.p_out[a_prime, m_3]
                + m.K_flow[n, c] * m.f[a, c, m_3]
            )
        return pyo.Constraint.Skip

    model.fuel_consumption = pyo.Constraint(model.N_gamma, model.C, model.M[3], rule=fuel_consumption_rule)

    def fuel_flow_balance_rule(m, n, c, m_3):
        if n in m.N_gamma:
            return sum(m.f[a, c, m_3] for a in m.A_n_minus[n]) == sum(m.f[a, c, m_3] for a in m.A_n_plus[n]) - m.w[n, c, m_3]
        return pyo.Constraint.Skip

    model.fuel_flow_balance = pyo.Constraint(model.N_gamma, model.C, model.M[3], rule=fuel_flow_balance_rule)

    def compression_increase_rule(m, n, m_3):
        if n in m.N_gamma:
            a = m.A_n_minus[n].first()
            a_prime = m.A_n_plus[n].first()
            return m.p_in[a, m_3] == m.P_hat[n] * m.p_out[a_prime, m_3]
        return pyo.Constraint.Skip

    model.compression_increase = pyo.Constraint(model.N_gamma, model.M[3], rule=compression_increase_rule)


def add_quality_constraints(model):

    # Pipe composition
    def pipe_composition_max_rule(m, a1, a2, c, m_3):
        return m.f[a1, a2, c, m_3] <= m.q_plus_arc[a1, a2, c] * sum(m.f[a1, a2, cp, m_3] for cp in m.C)

    model.pipe_composition_max = pyo.Constraint(model.A, model.C, model.M[3], rule=pipe_composition_max_rule)

    def pipe_composition_min_rule(m, a1, a2, c, m_3):
        return m.f[a1, a2, c, m_3] >= m.q_minus_arc[a1, a2, c] * sum(m.f[a1, a2, cp, m_3] for cp in m.C)

    model.pipe_composition_min = pyo.Constraint(model.A, model.C, model.M[3], rule=pipe_composition_min_rule)

    # Market composition
    def market_node_max_rule(m, n, c, m_3):
        return (
            sum(m.f[a, c, m_3] for a in m.A_n_plus[n])
            <= m.q_plus_node[n, c] * sum(m.f[a, cp, m_3] for a in m.A_n_plus[n] for cp in m.C) + m.delta[m_3] * model.M_n[n]
        )

    model.market_node_max = pyo.Constraint(model.N_m, model.C, model.M[3], rule=market_node_max_rule)

    def market_node_min_rule(m, n, c, m_3):
        return (
            sum(m.f[a, c, m_3] for a in m.A_n_plus[n])
            >= m.q_minus_node[n, c] * sum(m.f[a, cp, m_3] for a in m.A_n_plus[n] for cp in m.C) - m.delta[m_3] * model.M_n[n]
        )
    
    model.market_node_min = pyo.Constraint(model.N_m, model.C, model.M[3], rule=market_node_min_rule)


def add_homogeneous_splitting_constraints(model):

    def SOS1_v(m, n, m_3):
        return sum(m.v[n, z, m_3] for z in m.Z_v[n]) == 1

    model.S1_v = pyo.Constraint(model.N_s, model.M[3], rule=SOS1_v)

    def choice_v_flow_rule(m, n, z, m_3):
        return sum(m.e[n, c, z, m_3] for c in m.C) <= m.v[n, z, m_3] * m.M_n[n]

    model.choice_v_flow = pyo.Constraint(model.v_index, model.M[3], rule=choice_v_flow_rule)

    def homogeneous_split_rule(m, a1, a2, n, c, m_3):
        if (a1, a2) in m.A_n_minus[n]:
            return m.f[a1, a2, c, m_3] == sum(m.e[n, c, z, m_3] * m.E_nza[n, z, a1, a2] for z in m.Z_v[n])
        return pyo.Constraint.Skip

    model.homogeneous_split = pyo.Constraint(model.A, model.N_s, model.C, model.M[3], rule=homogeneous_split_rule)

def add_booking_constraints(model):
    # Booking-flow consistency
    def sufficient_entry_booking_rule(m, n, m_3):
        return sum(m.x_entry[n, k_prime, m_prime] for k_prime, m_prime in m.pred_chain[3, m_3]) >= sum(m.f[a, c, m_3] for a in m.A_n_plus[n] for c in m.C) - sum(m.f[a, c, m_3] for a in m.A_n_minus[n] for c in m.C)

    def sufficient_exit_booking_rule(m, n, m_3):
        return sum(m.x_exit[n, k_prime, m_prime] for k_prime, m_prime in m.pred_chain[3, m_3]) >= sum(m.f[a, c, m_3] for a in m.A_n_minus[n] for c in m.C) - sum(m.f[a, c, m_3] for a in m.A_n_plus[n] for c in m.C)

    model.sufficient_entry_booking = pyo.Constraint(model.N, model.M[3], rule=sufficient_entry_booking_rule)
    model.sufficient_exit_booking = pyo.Constraint(model.N, model.M[3], rule=sufficient_exit_booking_rule) 

    # Booking balance
    def booking_balance_rule(m, k, m_index):
        return sum(m.x_entry[n, k, m_index] for n in m.N) == sum(m.x_exit[n, k, m_index] for n in m.N)

    model.booking_balance = pyo.Constraint(model.KM, rule=booking_balance_rule)

def add_quality_deviation_constraints(model):

    # Limit total number of scenarios that can deviate from quality constraints
    def total_deviation_rule(m):
        return sum(m.delta[m_3] for m_3 in m.M[3]) <= m.C_deviation

    model.total_deviation = pyo.Constraint(rule=total_deviation_rule)

def add_constraints(model):
    add_flow_constraints(model)
    add_weymouth_constraints(model)
    add_pressure_constraints(model)
    add_compression_constraints(model)
    add_quality_constraints(model)
    add_homogeneous_splitting_constraints(model)
    add_booking_constraints(model)
    add_quality_deviation_constraints(model)

def create_model(network: networkx.Graph, scenarios=None,
                  cutting_plane_pairs=generate_cutting_plane_pairs(), splits_per_arc=splits_per_arc, allowed_deviation=ALLOWED_DEVIATION):
    if scenarios is None:
        raise ValueError("Scenarios must be provided to create the stochastic model. For a deterministic model, create a single scenario, or consult the deterministic_model folder.")

    model = pyo.ConcreteModel()

    build_sets(model, network, scenarios=scenarios, cutting_plane_pairs=cutting_plane_pairs, splits_per_arc=splits_per_arc)
    build_parameters(model, network, scenarios=scenarios, cutting_plane_pairs=cutting_plane_pairs, allowed_deviation=allowed_deviation)
    build_variables(model)

    add_constraints(model)

    add_expressions(model)
    add_objective(model)

    return model

def solve_model(model, verbose=True, time_limit=None, precision = 0.0001):
    solver = pyo.SolverFactory('gurobi')  # You can choose a different solver if needed
    solver.options['OutputFlag'] = 1  # ensures full output
    solver.options['MIPGap'] = precision  # set precision for MIP gap
    if time_limit is not None:
        solver.options['TimeLimit'] = time_limit  # optional
    results = solver.solve(model, tee=verbose)  # tee=True to display solver output
    return results

def plot_average_flows(model):
    """
    Plot average flows per arc and per component with error bars showing range across scenarios
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    arcs = list(model.A)
    components = list(model.C)
    
    # Calculate average flows per component
    avg_flows_per_component = {c: [] for c in components}
    
    for c in components:
        for a in arcs:
            avg_flow = sum(pyo.value(model.f[a, c, m_3]) * pyo.value(model.sp[3, m_3]) for m_3 in model.M[3])
            avg_flows_per_component[c].append(avg_flow)

    # Calculate min and max flows for each arc and component across scenarios
    min_flows_per_component = {c: [] for c in components}
    max_flows_per_component = {c: [] for c in components}
    
    for c in components:
        for a in arcs:
            flows = [pyo.value(model.f[a, c, m_3]) for m_3 in model.M[3]]
            min_flows_per_component[c].append(min(flows))
            max_flows_per_component[c].append(max(flows))

    # Calculate error bars
    arc_labels = [f"{a[0]}-{a[1]}" for a in arcs]
    x_pos = range(len(arcs))
    bar_width = 0.25
    
    for i, c in enumerate(components):
        lower_error = [avg_flows_per_component[c][j] - min_flows_per_component[c][j] for j in range(len(arcs))]
        upper_error = [max_flows_per_component[c][j] - avg_flows_per_component[c][j] for j in range(len(arcs))]
        
        offset = (i - len(components) / 2 + 0.5) * bar_width
        ax.bar([x + offset for x in x_pos], avg_flows_per_component[c], 
                yerr=[lower_error, upper_error], capsize=5, alpha=0.7, 
                label=c, width=bar_width, error_kw={'elinewidth': 2})

    ax.set_xlabel('Arc', fontsize=12)
    ax.set_ylabel('Flow (units)', fontsize=12)
    ax.set_title('Average Flow per Arc per Component with Range Across Scenarios', fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(arc_labels, rotation=45, ha='right')
    ax.legend(title='Component')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_component_ratios(model):
    """
    Plot component ratios summed across all market nodes for each component
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    components = list(model.C)

    for c in components:
        ratios = []
        for m_3 in model.M[3]:
            total_flow = sum(pyo.value(model.f[a, cp, m_3]) for n in model.N_m for a in model.A_n_plus[n] for cp in model.C)
            if total_flow > 0:
                ratio = sum(pyo.value(model.f[a, c, m_3]) for n in model.N_m for a in model.A_n_plus[n]) / total_flow
                ratios.append(ratio)
        if ratios:
            ax.plot(ratios, label=c, marker='o')

    ax.set_xlabel('Scenario Index', fontsize=12)
    ax.set_ylabel('Component Ratio', fontsize=12)
    ax.set_title('Component Ratios Summed Across All Market Nodes', fontsize=14)
    ax.legend(title='Component')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_results(model):
    plot_average_flows(model)
    plot_component_ratios(model)

if __name__ == "__main__":
    G = build_base_graph()
    scenarios = create_scenarios(NUMBER_OF_STAGES, BRANCHES_PER_STAGE, G)

    model = create_model(G, scenarios, cutting_plane_pairs=generate_cutting_plane_pairs())

    results = solve_model(model, time_limit=20)
    print(results)
    plot_results(model)