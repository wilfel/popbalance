"""
AS/SA Population Balance Model Code
Will Feldscher

New version to clean up structure (not tracking origin bin, etc.) and implement nucleation
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import ScalarFormatter

# --- Define values ---
d_min = 1           # minimum diameter plotted (nm)
d_max = 8000        # maximum diameter plotted (nm)
N_bins = 100         # number of bins to create
dt = 0.00000001           # length of one time step in simulation (s)
total_time = 0.002   # total simulation time (s)
plot_interval_seconds = 0.00001       # interval to plot (time length between screenshots plotted)
plot_interval = int(plot_interval_seconds / dt) # plot interval in terms of time steps
init_total_num = 1e9    # total number of particles initially
init_mean = 650     # initial geometric mean diameter of droplets (nm)
init_sd = 2.5       # initial geometric standard deviation of the lognormal dist (nm)

#init_mean = 2000         # initial geometric mean diameter of droplets (nm)
#init_sd = 2.5           # initial geometric standard deviation of the lognormal dist (nm)

# --- Classes ---
class SizeGrid:
    """
    Represents the static properties of the grid where the size distribution is plotted.
    
    Attributes:
        edges (np.ndarray): Edges of each size bin (nm)
        centers (np.ndarray): Geometric mean (center) of each bin (nm)
        widths (np.ndarray): Linear diameter width of each bin (nm)
        n_bins (int): Number of bins
        n_steps (int): Number of time steps
        delta_log_d (float): Width of each bin in log space
    
    Methods:
        init_lognormal(self, N0, d_g, sigma_g): Returns an array containing the number of particles in each bin
    """
    def __init__(self, d_min, d_max, n_bins, dt, total_time):
        self.edges = np.logspace(np.log10(d_min), np.log10(d_max), n_bins + 1)  # edges of each size bin (nm)
        self.centers = np.sqrt(self.edges[:-1] * self.edges[1:])    # center diameter of each bin (nm)
        self.widths = self.edges[1:] - self.edges[:-1]  # length of each bin (nm)
        self.n_bins = n_bins    # number of bins
        self.n_steps = int(total_time / dt) # number of time steps
        self.delta_log_d = np.log10(self.edges[1]) - np.log10(self.edges[0])    # width of each bin in log space
    
    def init_lognormal(self, N0, d_g, sigma_g):
        """
        Creates the initial particle size distribution.

        Args:
            N0 (float): Initial total number of particles
            d_g (float): Initial geometric mean diameter of droplets (nm)
            sigma_g (float): Initial geometric standard deviation of the lognormal dist
            
        Returns:
            N_dens_dist_init (np.ndarray): Array containing number density distribution evaluated at bin centers
        """
        N_dens_dist_init = (N0 /
                (self.centers * np.sqrt(2*np.pi) * np.log(sigma_g)) *
                np.exp(-(np.log(self.centers / d_g))**2 /
                    (2 * (np.log(sigma_g))**2)))
        return N_dens_dist_init
    
class Population:
    """
    Represents the entire population of particles
    
    Attributes:
        grid (class): Contains all static information on grid where population is plotted
        N_density (np.ndarray): Array containing number density distribution evaluated at bin centers
        N_dist (np.ndarray): Array containing number distribution of particles in each bin
        
    Methods:
    """
    def __init__(self, grid, N_dens_dist_init, wet):
        self.grid = grid    
        self.N_density = N_dens_dist_init
        self.d_dist = grid.centers.copy() * 1e-9   # bin centers (m)
        self.N_dist = self.N_density * grid.delta_log_d   # initial total number in each bin
        self.V = np.pi/6 * self.d_dist**3   # volume of each center of a SINGLE particle (m^3)
        self.wet = wet
        if wet:
            self.wet_init()
        else:
            self.dry_init()
    
    def wet_init(self):
        # Initialize solute concentration of every bin
        n = self.grid.n_bins
        self.frac_SA = np.full(n, 0.75)
        self.frac_AS = np.full(n, 0.25)
        self.x_solute = np.full(n, 0.001)
        
        rho_water = 1000
        self.m = rho_water * self.V   # total mass of particles
        self.m_solute = self.x_solute * self.m
        self.m_water = (1-self.x_solute) * self.m

    def dry_init(self):
        n = self.grid.n_bins
        self.m = np.zeros_like(n)
        self.V = np.zeros_like(n)
        
    def wet_update_diameter(self, dm_dt, dt):
        """
        Updates diameter and other variables associated with the wet population based on a mass loss rate.

        Args:
            dm_dt (float): Mass loss rate of water [kg/s]
            dt (float): Time step length [s]
        
        Returns:
            None
        """

        self.m_water = np.maximum(self.m_water + dm_dt * dt, 0) # Ensure mass will not go negative
        self.m = self.m_solute + self.m_water
        self.x_solute = np.zeros_like(self.m)
        mask = self.m > 0
        self.x_solute[mask] = self.m_solute[mask] / self.m[mask]
        
        rho_water = 1000 # density in kg/m^3
        rho_solute = 1613
        rho_mix = np.full_like(self.m, rho_water)

        mask = self.m > 0
        rho_mix[mask] = self.m[mask] / (self.m_solute[mask]/rho_solute + self.m_water[mask]/rho_water)
        self.V = self.m / rho_mix
        self.d_dist = (self.V * 6/np.pi)**(1/3)
            
class GasPhase:
    """
    Represents the gas phase of the system and its composition.
    
    Attributes:
        the
    
    Methods:
        the
    """
    def __init__(self):
        self.conc_SA_gas = 0    # concentration in kg/m^3
        self.mass_SA_gas = 0
        self.conc_AS_gas = 0   # concentration in kg/m^3, I think this will effectively stay 0

class DryingModel:
    def __init__(self):
            pass
        
    def evaporate_water(self, wet_pop):
        d = wet_pop.d_dist   # diameter (m)
        A_d = 4*np.pi*(d/2)**2 # surface area of droplet, m^2
        rho_g = 2.416 # density of air at P = 30 PSI, T = 25 C, kg/m^3
        #Tg = 298.15 # temperature of the gas, K (assume 25 C)
        #Dv = 1.175E-9 * (Tg)**(1.75)/(rho_g)    # mass diffusion coeff of vapor in gas, m^2/s
        Dv = 1.2E-5
        mu_g = 18.37E-6       # viscosity of air in Pa*s
        RH_sat = 1      # assume air immediately above droplet is fully saturated
        RH_bulk = 0.3
        P_sat = 0.0313 # saturated vapor pressure of water at 25 C is 0.0313 atm
        P_tot = 2.04138 # total pressure in atm; 30 PSI
        y_sat = RH_sat*P_sat*18.016/(RH_sat*P_sat*18.016+(P_tot-RH_sat*P_sat)*28.97)
        y_bulk = RH_bulk*P_sat*18.016/(RH_bulk*P_sat*18.016+(P_tot-RH_bulk*P_sat)*28.97)
        v_g = 0.63662 # velocity of gas, m/s
        Re = rho_g * v_g * d / mu_g # reynolds number
        Sc = mu_g/(rho_g * Dv)  # schmidt number
        Sh = 2 + 0.552*Re**(1/2)*Sc**(1/3)  # sherwood number
        
        
        dm_dt = np.zeros_like(d)

        mask = d > 0

        dm_dt[mask] = -(A_d[mask] * rho_g * Dv *
                        (y_sat - y_bulk) *
                        Sh[mask]) / d[mask]        
        return(dm_dt)   # kg/s
    
    def nucleation(self, wet_pop, dry_pop, dt):
        C_sat_SA = 83   # kg/m^3 max concentration of SUCCINIC ACID
        C_SA_current = wet_pop.x_solute * wet_pop.frac_SA * 1000
        wet_d_dist_old = wet_pop.d_dist.copy()
        # Create boolean masks to identify dried particles
        not_empty_mask = wet_pop.m > 0     # bins that actually contain particles
        no_water_mask = wet_pop.m_water <= 1e-25 #<= 1e-15    # particles already fully dry
        above_sat_mask = C_SA_current >= C_sat_SA   # supersaturated particles
        dry_mask = (no_water_mask | above_sat_mask) & not_empty_mask    # dried particles have no water OR are supersaturated AND are not already empty

        # Determine dry particle size
        rho_SA = 1560   # kg/m^3
        rho_AS = 1770   # kg/m^3
        mass_SA = wet_pop.m_solute * wet_pop.frac_SA
        mass_AS = wet_pop.m_solute * wet_pop.frac_AS
        dry_vol = mass_SA/rho_SA + mass_AS/rho_AS
        dry_d_dist = (dry_vol * 6/np.pi)**(1/3)
        
        wet_pop.m_water[dry_mask] = 0   # remove reamining water from nucleating bins
        #dry_pop.N_dist[dry_mask] += wet_pop.N_dist[dry_mask]
        #wet_pop.N_dist[dry_mask] = 0
        
        
        new_particles = wet_pop.N_dist[dry_mask]

        dry_bins = np.searchsorted(wet_pop.grid.edges, dry_d_dist*1e9) - 1
        dry_bins = np.clip(dry_bins, 0, wet_pop.grid.n_bins-1)
        
        for idx, N in zip(np.where(dry_mask)[0], new_particles):

            j = dry_bins[idx]
            dry_pop.N_dist[j] += N

        wet_pop.N_dist[dry_mask] = 0
    
    def wet_SA_massloss(self, wet_pop):
        d_m = wet_pop.d_dist
        T = 298.15      # K
        R = 8.314   # J/molK
        M_sa = 0.11809 #kg/mol
        M_w = 0.01806   # kg/mol
        A_d = 4*np.pi*(d_m/2)**2 # surface area of droplet, m^2
        gamma_SA = 1        # activity coefficient = 1 for now
        p_sat_SA = 2.55E-5      # Pa
        conc_SA_liq = wet_pop.x_solute * wet_pop.frac_SA * 1000
        molefrac_SA_liq = (conc_SA_liq/M_sa) / ((conc_SA_liq/M_sa) + ((1000 - conc_SA_liq)/M_w))
        p_SA_surface = molefrac_SA_liq * gamma_SA * p_sat_SA
        p_SA_bulk = (gas.conc_SA_gas * R * T)/M_sa
        rho_g = 2.416 # density of air at P = 30 PSI, T = 25 C, kg/m^3
        mu_g = 18.37E-6       # viscosity of air in Pa*s
        D_SA = 2E-6         # check this value later
        v_g = 0.63662 # velocity of gas, m/s
        Re_SA = rho_g * v_g * d_m / mu_g
        Sc_SA = mu_g/(rho_g * D_SA)
        Sh_SA = 2 + 0.552*Re_SA**(1/2)*Sc_SA**(1/3)
        k_mass = Sh_SA * D_SA / d_m
        C_SA_surface = p_SA_surface * M_sa / (R * T)
        C_SA_bulk    = p_SA_bulk    * M_sa / (R * T)
        dm_SA_dt = k_mass * A_d * (C_SA_surface-C_SA_bulk)  # kg/s
        return dm_SA_dt
    
    def condensation(self, gas):
        d_dry = dry_pop.grid.centers * 1E-9
        T = 298.15      # K
        R = 8.314   # J/molK
        N = 1e9    # total number of particles initially
        M_sa = 0.11809 #kg/mol
        M_w = 0.01806   # kg/mol
        D_SA = 2E-6         # check this value later
        n_density = 1E15    # check this idk
        vol = N / n_density
        gas.conc_SA_gas = gas.mass_SA_gas / vol
        
        gamma_SA = 1        # activity coefficient = 1 for now
        p_sat_SA = 2.55E-5      # Pa
        
        conc_SA_liq = wet_pop.x_solute * wet_pop.frac_SA * 1000
        molefrac_SA_liq = (conc_SA_liq/M_sa) / ((conc_SA_liq/M_sa) + ((1000 - conc_SA_liq)/M_w))
        p_SA_surface = molefrac_SA_liq * gamma_SA * p_sat_SA
        p_SA_bulk = (gas.conc_SA_gas * R * T)/M_sa
        C_SA_surface = p_SA_surface * M_sa / (R * T)
        C_SA_bulk    = p_SA_bulk    * M_sa / (R * T)

        alpha_sa = 1    # source says accom coeff for SA should be about unity
        vmolec_SA = (8 * R * T / (np.pi * M_sa))**(1/2)
        lambda_sa = 3 * D_SA / vmolec_SA
        Kn_sa = 2 * lambda_sa / d_dry
        rho_sa = 1560 # kg/m^3
        dd_DRY_dt = 1/d_dry * (4 * D_SA  / rho_sa) * 0.75*alpha_sa*(1+Kn_sa)/(1 + Kn_sa**2 + Kn_sa + 0.283 * Kn_sa * alpha_sa + 0.75*alpha_sa) * (C_SA_bulk - C_SA_surface)        
        return dd_DRY_dt
    
    def advance(self, wet_pop, dry_pop, gas, dt):
        d_old = wet_pop.d_dist.copy()          # 1. capture current centers (snapped)
        dm_dt = self.evaporate_water(wet_pop)
        wet_pop.wet_update_diameter(dm_dt, dt) # 2. compute new physical diameters
        d_new = wet_pop.d_dist.copy()          # 3. capture new physical diameters
        self.nucleation(wet_pop, dry_pop, dt)
        #wet_pop.redistribute_bins(d_old, d_new) # 4. remap → snaps d_dist at end
        self.condensation(gas)
        """
        # DEBUG
        d_old = wet_pop.d_dist.copy()
        dm_dt = self.evaporate_water(wet_pop)
        wet_pop.wet_update_diameter(dm_dt, dt)
        d_new = wet_pop.d_dist.copy()
        print(f"d_old range: {d_old.min()*1e9:.1f} – {d_old.max()*1e9:.1f} nm")
        print(f"d_new range: {d_new.min()*1e9:.1f} – {d_new.max()*1e9:.1f} nm")
        print(f"dd range: {((d_new-d_old)*1e9).min():.4f} – {((d_new-d_old)*1e9).max():.4f} nm")
        print(f"N_dist nonzero bins: {np.count_nonzero(wet_pop.N_dist)}")
        print(f"m_water range (nonzero): {wet_pop.m_water[wet_pop.m_water>0].min():.3e} – {wet_pop.m_water[wet_pop.m_water>0].max():.3e}")
        print("---")
        """
# --- Set up Grid ---
grid = SizeGrid(d_min, d_max, N_bins, dt, total_time)
N_dens_dist_init = grid.init_lognormal(init_total_num, init_mean, init_sd)

# --- Set up Populations ---
wet_pop = Population(grid, N_dens_dist_init, wet=True)
dry_pop = Population(grid, np.zeros_like(N_dens_dist_init), wet=False)
gas = GasPhase()
drying = DryingModel()

monitor_bins = np.arange(wet_pop.grid.n_bins - 30, wet_pop.grid.n_bins-20)  # indices of 5 largest bins

# --- Precompute time evolution ---
wet_history = []
dry_history = []
gas_history = []
time_history = []
dist_history =[]

for step in range(grid.n_steps):
    drying.advance(wet_pop, dry_pop, gas, dt)
    total = wet_pop.N_dist.sum()
    if step % plot_interval == 0:
        wet_history.append(wet_pop.N_dist.copy())
        dry_history.append(dry_pop.N_dist.copy())

        gas_history.append(gas.conc_SA_gas)   # store gas concentration
        time_history.append(step * dt)
        
        dist_history.append(wet_pop.d_dist.copy())
        # --- DEBUG: print m_water for monitored bins ---
        t_ms = step * dt * 1000
        print(f"\nt = {t_ms:.3f} ms")
        print(f"{'Bin':>5} {'Center (nm)':>12} {'N_dist':>12} {'m_water (kg)':>15} {'x_solute':>10}")
        for b in monitor_bins:
            print(f"{b:>5} {grid.centers[b]:>12.1f} {wet_pop.N_dist[b]:>12.3e} "
                  f"{wet_pop.m_water[b]:>15.3e} {wet_pop.x_solute[b]:>10.4f}")


# --- Animation ---
fig, ax = plt.subplots()

def update(frame):
    ax.cla()

    d_phys = dist_history[frame] * 1e9
    n_wet = wet_history[frame]
    n_dry = dry_history[frame]

    mask = n_wet > 0

    # Wet bars — black outline, white fill, same as original
    ax.bar(
        d_phys[mask],
        n_wet[mask] / grid.delta_log_d,
        width=grid.widths[mask] * (d_phys[mask] / grid.centers[mask]),
        align='center',
        edgecolor='black',
        linewidth=0.8,
        label='Wet'
    )

    # Dry bars — red outline, fixed grid positions
    ax.bar(
        grid.edges[:-1],
        n_dry / grid.delta_log_d,
        width=grid.widths,
        align='edge',
        edgecolor='red',
        linewidth=0.8,
        alpha=0.6,
        label='Dry'
    )

    ax.set_xscale('log')
    ax.set_xlim(d_min, d_max)
    ax.set_ylim(0, np.max([w.max() for w in wet_history]) / grid.delta_log_d * 1.1)
    ax.set_xlabel('Particle diameter [nm]')
    ax.set_ylabel('Number density [#/cm³]')
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.ticklabel_format(axis='x', style='plain')
    ax.legend()

    t = frame * plot_interval_seconds * 1000
    ax.set_title(f"Simulated Drying of 1 g/L 25 AS:75 SA Droplets (t = {t:.2f} ms)")

ani = FuncAnimation(fig, update, frames=len(wet_history), interval=150, blit=False) # type: ignore
plt.show()

plt.figure()

plt.plot(time_history, gas_history)

plt.xlabel("Time (s)")
plt.ylabel("SA Gas Concentration (kg/m³)")
plt.title("Succinic Acid Gas Concentration vs Time")

plt.grid(True)

plt.show()