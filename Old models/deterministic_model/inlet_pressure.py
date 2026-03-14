import numpy as np
import matplotlib.pyplot as plt

# Parameters
K = 1.0
P_out = 30.0
M = 45.0

# Inlet pressure domain
p_min = P_out + 1.0
p_max = 60.0
p_in = np.linspace(p_min, p_max, 800)

# Weymouth function
def weymouth(p):
    return K * np.sqrt(p**2 - P_out**2)

# Activated Weymouth constraint
wey_active = weymouth(p_in)

# Non-activated Weymouth constraints (+M)
wey_inactive_1 = weymouth(p_in) + M
wey_inactive_2 = weymouth(p_in) + 1.4 * M

# Well-separated linearization points
tangent_points = [33.0, 40.0, 50.0]

# Local visibility window for cutting planes
window = 7.50

plt.figure()

# Activated Weymouth (solid)
plt.plot(p_in, wey_active, label="Activated Weymouth constraint")

# Active cutting planes (solid, local only)
for i, p0 in enumerate(tangent_points):
    slope = K * p0 / np.sqrt(p0**2 - P_out**2)
    intercept = weymouth(p0) - slope * p0

    p_local = np.linspace(max(p_min, p0 - window),
                          min(p_max, p0 + window), 200)
    cp_local = slope * p_local + intercept

    plt.plot(
        p_local,
        cp_local,
        color="green",
        label="Active cutting plane" if i == 0 else None
    )

# Non-activated Weymouth constraints (dotted)
# Non-activated Weymouth constraints (dotted)
plt.plot(p_in, wey_inactive_1, linestyle=":", color="orange", label="Non-activated Weymouth (+M)")
plt.plot(p_in, wey_inactive_2, linestyle=":", color="orange")

plt.xlabel(r"Inlet pressure $p_{a,\mathrm{in}}$")
plt.ylabel(r"Max total flow $\sum_{c \in C} f_a^c$")
plt.title("Localized cutting planes for the Weymouth constraint")
plt.legend()
plt.show()
