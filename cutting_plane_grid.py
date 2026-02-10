import numpy as np
from base_model import generate_cutting_plane_pairs
import matplotlib.pyplot as plt

# Assuming your constants and generate_cutting_plane_pairs() function are already defined
pairs = generate_cutting_plane_pairs(
    n_p_out=10,
    p_out_low=0,
    p_out_high=10,
    n_p_in=10,
    p_in_low=0,
    p_in_high=10
)

# Extract p_in and p_out values
p_in_values = [pair[0] for pair in pairs]
p_out_values = [pair[1] for pair in pairs]

# Calculate Weymouth cutting constant for each pair
weymouth_constants = [np.sqrt(pair[0]**2 - pair[1]**2) for pair in pairs]

# Create the plot
plt.figure(figsize=(10, 6))
scatter = plt.scatter(p_out_values, p_in_values, c=weymouth_constants, marker='.', 
                      s=200, cmap='viridis')
plt.xlabel(r'$P_{\text{out},l}$')
plt.ylabel(r'$P_{\text{in},l}$')
plt.title('Cutting Plane Points')
plt.grid(True, alpha=0.3)
colorbar = plt.colorbar(scatter)
colorbar.set_label('Linearize approximation values')

# # Add diagonal line where p_in = p_out
# max_val = max(max(p_out_values), max(p_in_values))
# plt.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='p_in = p_out')
# plt.legend()

plt.tight_layout()
plt.savefig("cutting_plane_grid.png", dpi=300)
plt.show()
