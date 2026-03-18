import matplotlib.pyplot as plt
import numpy as np

# --- Physical parameters ---
d_min = 0.01      # microns
d_max = 2.0       # microns
N_bins = 40

N0 = 1e9          # total number concentration [#/m^3]
d_g = 0.3         # geometric mean diameter [microns]
sigma_g = 1.8     # geometric standard deviation

K = 0.02          # drying constant [micron^2 / s]

# --- Time parameters ---
dt = 0.001         # s
t_end = 10.0      # s

# Bin edges (log-spaced)
edges = np.logspace(np.log10(d_min), np.log10(d_max), N_bins + 1)

# Bin midpoints (geometric mean)
d = np.sqrt(edges[:-1] * edges[1:])

# Bin widths
delta_d = edges[1:] - edges[:-1]

def lognormal_ndf(d, N0, d_g, sigma_g):
    return (N0 /
            (d * np.sqrt(2 * np.pi) * np.log(sigma_g)) *
            np.exp(-(np.log(d / d_g))**2 /
                   (2 * (np.log(sigma_g))**2)))
# Initial number in each bin
n0 = lognormal_ndf(d, N0, d_g, sigma_g)
N = n0 * delta_d     # number per bin
print("Total number concentration:", np.sum(N))

#Droplet drying law
def d_dot(d, K):
    return -K / (2 * d)

# This is a 1D advection equation in size space, solved using upwind differencing (since droplets shrink).
time = 0.0
history = []

while time < t_end:
    N_new = N.copy()

    # Fluxes at bin interfaces
    for i in range(1, N_bins):
        d_face = edges[i]
        v = d_dot(d_face, K)

        # Upwind: shrinking droplets move from larger to smaller bins
        if v < 0:
            flux = v * N[i]
        else:
            flux = v * N[i - 1]

        N_new[i - 1] -= flux * dt / delta_d[i - 1]
        N_new[i]     += flux * dt / delta_d[i]

    # Boundary condition at smallest size (evaporation loss)
    N_new[0] = max(N_new[0], 0.0)

    N = N_new
    history.append(N.copy())
    time += dt

print(np.min(history[-1]), np.max(history[-1]))

plt.figure(figsize=(7,5))

for i, idx in enumerate([0, int(len(history)/3), int(2*len(history)/3), -1]):
    plt.plot(d, history[idx] / delta_d, label=f"t = {idx*dt:.1f} s")

plt.xscale("log")
plt.yscale("linear")
plt.xlabel("Droplet diameter [µm]")
plt.ylabel("Number density [#/m³/µm]")
plt.legend()
plt.title("Population balance: drying droplets")
plt.tight_layout()
plt.show()

