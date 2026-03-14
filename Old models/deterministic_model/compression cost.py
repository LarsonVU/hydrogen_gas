import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Parameters (reasonable values)
# -----------------------------
Q_std = 1.0e6          # Standard flow [Sm3/day]
p_i = 40.0             # Inlet pressure [bar]
p_d = 70.0             # Discharge pressure [bar]
p_std = 1.01325        # Standard pressure [bar]
T_i = 288.15           # Inlet temperature [K]
T_std = 288.15         # Standard temperature [K]
Z = 0.9                # Compressibility factor [-]
eta = 0.75             # Compressor efficiency [-]

k_ng = 1.29            # Specific heat ratio natural gas
k_h2 = 1.40            # Specific heat ratio hydrogen

# -----------------------------
# Hydrogen share (0–50 vol%)
# -----------------------------
h2_share = np.linspace(0.0, 0.5, 50)

# Linear mixing rule for k
k_mix = (1 - h2_share) * k_ng + h2_share * k_h2

# -----------------------------
# Compressor power formula
# -----------------------------
pressure_term = (p_d / p_i) ** ((k_mix - 1) / k_mix) - 1

P = (
    (p_std * Z * T_i) / (T_std * eta * (k_mix - 1))
    * Q_std
    * pressure_term
    / (24 * 3600)      # convert day -> seconds
) / 1e6                # convert W -> MW

# -----------------------------
# Plot
# -----------------------------
plt.figure()
plt.plot(h2_share * 100, P)
plt.xlabel("Hydrogen share [% vol]")
plt.ylabel("Compressor power [MW]")
plt.title("Effect of Hydrogen Blending on Compressor Power")
plt.grid(True)
plt.show()
