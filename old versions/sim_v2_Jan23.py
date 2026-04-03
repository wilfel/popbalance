import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import ScalarFormatter

# ---- Parameters ----
d_min = 1    # nanometers
d_max = 2000     # nanometers
N_bins = 50

N0 = 1e9        # total number concentration [#/m^3]
d_g = 300       # geometric mean diameter [nanometers]
sigma_g = 1.8   # geometric standard deviation

# ---- Time parameters ----
dt = 0.01              # model timestep (seconds)
total_time = 100       # simulation time (seconds)
plot_interval_seconds = 2 # update plot every n seconds
n_steps = int(total_time/dt)
plot_interval = int(plot_interval_seconds/dt)    # update plot every N timesteps

# ---- Log-spaced bins ----
edges = np.logspace(np.log10(d_min), np.log10(d_max), N_bins + 1)
d = np.sqrt(edges[:-1] * edges[1:])
delta_d = edges[1:] - edges[:-1]

# ---- Lognormal number distribution ----
def lognormal_ndf(d, N0, d_g, sigma_g):
    return (N0 /
            (d * np.sqrt(2*np.pi) * np.log(sigma_g)) *
            np.exp(-(np.log(d / d_g))**2 /
                   (2 * (np.log(sigma_g))**2)))

# ---- Initial population ----
n0 = lognormal_ndf(d, N0, d_g, sigma_g)
N = n0 * delta_d
N *= N0 / np.sum(N)

# ---- Transform function (EDIT THIS LATER) ----
def transform_population(N, d, dt, K=20):
    """
    Evaporation-driven droplet drying using the d^2-law:
        d(d^2)/dt = -K

    Parameters
    ----------
    N : array
        Number of droplets per bin
    d : array
        Bin-center diameters [microns]
    dt : float
        Time step [s]
    K : float
        Evaporation constant [micron^2 / s]
    """

    # Current squared diameters
    d2 = d**2

    # Evolve according to d^2-law
    d2_new = d2 - K * dt

    # Enforce disappearance when fully dried
    d2_new[d2_new < 0.0] = 0.0 # Track based on what particle size was at the beginning how small it could get before it becomes a particle
    d_new = np.sqrt(d2_new)

    # Interpolate population back onto fixed grid
    # Droplets move from larger to smaller diameters
    N_new = np.interp(d, d_new, N, left=0.0, right=0.0)

    return N_new

# ---- Precompute time evolution ----
history = []
N_current = N.copy()

for step in range(n_steps):
    N_current = transform_population(N_current, d, dt)
    N_current[N_current < 0] = 0.0
    #N_current *= N0 / np.sum(N_current)

    if step % plot_interval == 0:
        history.append(N_current.copy())
    print(np.sum(N_current))

# ---- Animation (histogram-style bins) ----
fig, ax = plt.subplots()

# Initial histogram
bars = ax.bar(
    edges[:-1],
    history[0] / delta_d,
    width=delta_d,
    align="edge",
    edgecolor="black",
    linewidth=0.8
)

ax.set_xscale("log")
ax.set_xlim(1, d_max)
ax.set_ylim(0, np.max(history[0] / delta_d) * 1.2)

ax.set_xlabel("Droplet diameter [nm]")
ax.set_ylabel("Number density [#/m³/nm]")
ax.set_title("Evolving droplet size distribution")

def update(frame):
    heights = history[frame] / delta_d
    for bar, h in zip(bars, heights):
        bar.set_height(h)
    seconds = frame * plot_interval_seconds
    ax.set_title(f"Evolving droplet size distribution (t = {seconds} seconds)")
    return bars

ani = FuncAnimation(
    fig,
    update,
    frames=len(history),
    interval=100,   # ms between frames
    blit=False      # blitting does not work reliably with bars
)

plt.show()
