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
d_max = 4000        # maximum diameter plotted (nm)
N_bins = 150         # number of bins to create
dt = 0.0000001           # length of one time step in simulation (s)
total_time = 0.005   # total simulation time (s)
plot_interval_seconds = 0.00001       # interval to plot (time length between screenshots plotted)
plot_interval = int(plot_interval_seconds / dt) # plot interval in terms of time steps
init_total_num = 5.63e7    # total number of particles initially
#init_mean = 630
init_mean = 825  # initial geometric mean diameter of droplets (nm)
init_sd = 1.68       # initial geometric standard deviation of the lognormal dist (nm)

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
        self.delta_log_d = np.log(self.edges[1]) - np.log(self.edges[0])    # width of each bin in log space
    
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
        self.m_SA = self.x_solute * self.m * self.frac_SA

    def dry_init(self):
        n = self.grid.n_bins
        self.m = np.zeros(n, dtype=float)  # float array with n elements
        self.V = np.zeros(n, dtype=float)  # float array with n elements
        self.m_solute = np.zeros(n, dtype=float)  # float array with n elements
        self.m_SA = np.zeros(n, dtype=float)  # float array with n elements
        self.frac_AS = np.zeros(n, dtype=float)  # float array with n elements
        self.frac_SA = np.zeros(n, dtype=float)  # float array with n elements
        
    def wet_update_diameter(self, dm_dt,dm_SA_dt, dt):
        """
        Updates diameter and other variables associated with the wet population based on a mass loss rate.

        Args:
            dm_dt (float): Mass loss rate of water [kg/s]
            dt (float): Time step length [s]
        
        Returns:
            None
        """
        self.m_water = np.maximum(self.m_water + dm_dt * dt, 0) # Ensure mass will not go negative
        self.m_solute = np.maximum(self.m_solute + dm_SA_dt * dt, 0) # Ensure mass will not go negative
        self.m_SA = np.maximum(self.m_SA + dm_SA_dt * dt, 0)

        self.m = self.m_solute + self.m_water
        self.x_solute = np.zeros_like(self.m)
        mask = self.m > 0
        self.x_solute[mask] = self.m_solute[mask] / self.m[mask]
        
        rho_water = 1000 # density in kg/m^3
        rho_solute = 1613
        rho_mix = np.full_like(self.m, rho_water)

        self.frac_SA = np.zeros_like(self.m_solute)
        self.frac_AS = np.zeros_like(self.m_solute)
        self.frac_SA[mask] = self.m_SA[mask] / self.m_solute[mask]
        self.frac_AS[mask] = 1 - self.frac_SA[mask]
        rho_mix[mask] = self.m[mask] / (self.m_solute[mask]/rho_solute + self.m_water[mask]/rho_water)
        self.V = self.m / rho_mix
        self.d_dist = (self.V * 6/np.pi)**(1/3)
    
    def dry_update_diameter(self, dm_cond_dt, dt):
        self.m_SA += dm_cond_dt * dt        
        self.m_solute += dm_cond_dt * dt    
        self.m += dm_cond_dt * dt          
               
        mask = self.m > 0
        self.frac_SA[mask] = self.m_SA[mask] / self.m_solute[mask]
        self.frac_AS[mask] = 1 - self.frac_SA[mask]
        
        self.frac_SA[~mask] = 0.0
        self.frac_AS[~mask] = 0.0
        rho_SA = 1560  # kg/m³
        rho_AS = 1770  # kg/m³ 
        rho_total = self.frac_SA * rho_SA + self.frac_AS * rho_AS
        self.V[mask] = self.m_solute[mask] / rho_total[mask]
        self.d_dist[mask] = (self.V[mask] * 6/np.pi)**(1/3)
        

    def get_summary_stats(self):
        pass
            
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
        self.mass_SA_gas = 1e-11
        self.conc_AS_gas = 0  # concentration in kg/m^3, I think this will effectively stay 0
    
    def update_gas(self, dm_SA_dt, dm_cond_dt, wet_pop, dt):
        pass
        gas.mass_SA_gas += -np.sum(dm_SA_dt*wet_pop.N_dist) * dt   # kg
        #N = 5.63e7    # total number of particles initially
        #n_density = 1E15    # check this idk
        #vol = N / n_density
        vol = 1E-6
        gas.conc_SA_gas = gas.mass_SA_gas / vol
        

class DryingModel:
    def __init__(self):
            pass
        
    def evaporate_water(self, wet_pop):
        d = wet_pop.d_dist   # diameter (m)
        A_d = 4*np.pi*(d/2)**2 # surface area of droplet, m^2
        rho_g = 2.416 # density of air at P = 30 PSI, T = 25 C, kg/m^3
        #Tg = 298.15 # temperature of the gas, K (assume 25 C)
        #Dv = 1.175E-9 * (Tg)**(1.75)/(rho_g)    # mass diffusion coeff of vapor in gas, m^2/s
        Dv = 2.12E-5
        mu_g = 18.37E-6       # viscosity of air in Pa*s
        RH_sat = 1      # assume air immediately above droplet is fully saturated
        RH_bulk = 0.2
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
    
    def wet_to_dry(self, wet_pop, dry_pop, dt):
        C_sat_SA = 83   # kg/m^3 max concentration of SUCCINIC ACID
        C_SA_current = wet_pop.x_solute * wet_pop.frac_SA * 1000
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
            dry_pop.m[j] += wet_pop.m[idx]
            dry_pop.m_solute[j] += wet_pop.m_solute[idx]
            dry_pop.m_SA[j]+= wet_pop.m_SA[idx]

        nonempty = dry_pop.m_solute > 0
        dry_pop.frac_SA[nonempty] = dry_pop.m_SA[nonempty] / dry_pop.m_solute[nonempty]
        dry_pop.frac_AS[nonempty] = 1 - dry_pop.frac_SA[nonempty]
        wet_pop.N_dist[dry_mask] = 0
        wet_pop.m[dry_mask] = 0
        wet_pop.m_solute[dry_mask] = 0
        wet_pop.m_SA[dry_mask] = 0
        
    def wet_SA_massloss(self, wet_pop):
        not_empty_mask = wet_pop.N_dist > 0
        d_m = np.zeros_like(wet_pop.d_dist)
        d_m[not_empty_mask] = wet_pop.d_dist[not_empty_mask]

        T = 298.15      # K
        R = 8.314   # J/molK
        M_sa = 0.11809 # kg/mol
        M_w = 0.01806   # kg/mol
        A_d = 4*np.pi*(d_m/2)**2 # surface area of droplet, m^2
        gamma_SA = 1        # activity coefficient = 1 for now
        p_sat_SA = 1.95E-3      # Pa
        #C_sat_kgm3    = p_sat_SA    * M_sa / (R * T)
        conc_SA_liq = wet_pop.x_solute * wet_pop.frac_SA * 1000
        molefrac_SA_liq = (conc_SA_liq/M_sa) / ((conc_SA_liq/M_sa) + ((1000 - conc_SA_liq)/M_w))
        p_SA_surface = molefrac_SA_liq * gamma_SA * p_sat_SA
        p_SA_bulk = (gas.conc_SA_gas * R * T)/M_sa
        rho_g = 2.416 # density of air at P = 30 PSI, T = 25 C, kg/m^3
        mu_g = 18.37E-6       # viscosity of air in Pa*s
        D_SA = 9.52E-6         # check this value later
        v_g = 0.63662 # velocity of gas, m/s
        Re_SA = rho_g * v_g * d_m / mu_g
        Sc_SA = mu_g/(rho_g * D_SA)
        Sh_SA = 2 + 0.552*Re_SA**(1/2)*Sc_SA**(1/3)
        k_mass = np.zeros_like(d_m)
        k_mass[not_empty_mask] = Sh_SA[not_empty_mask] * D_SA / d_m[not_empty_mask]
        C_SA_surface = p_SA_surface * M_sa / (R * T)
        C_SA_bulk    = p_SA_bulk    * M_sa / (R * T)
        dm_SA_dt = -k_mass * A_d * (C_SA_surface-C_SA_bulk)  # kg/s
        print(p_SA_bulk)
        return dm_SA_dt
    
    def condensation(self, gas):
        not_empty_mask = dry_pop.N_dist > 0
        d_dry = np.zeros_like(dry_pop.N_dist)
        d_dry[not_empty_mask] = dry_pop.d_dist[not_empty_mask]
        
        T = 298.15      # K
        R = 8.314   # J/molK
        M_sa = 0.11809 #kg/mol
        M_w = 0.01806   # kg/mol
        D_SA = 2E-6         # check this value later
        
        gamma_SA = 1        # activity coefficient = 1 for now
        p_sat_SA = 2.55E-5      # Pa
        

        conc_SA_liq = wet_pop.x_solute * wet_pop.frac_SA * 1000
        molefrac_SA_liq = (conc_SA_liq/M_sa) / ((conc_SA_liq/M_sa) + ((1000 - conc_SA_liq)/M_w))
        p_SA_surface = molefrac_SA_liq * gamma_SA * p_sat_SA
        p_SA_bulk = (gas.conc_SA_gas * R * T)/M_sa
        C_SA_surface = np.zeros_like(d_dry)
        C_SA_surface[not_empty_mask] = p_SA_surface[not_empty_mask] * M_sa / (R * T)

        C_SA_bulk    = p_SA_bulk    * M_sa / (R * T)

        alpha_sa = 1    # source says accom coeff for SA should be about unity
        vmolec_SA = (8 * R * T / (np.pi * M_sa))**(1/2)
        lambda_sa = 3 * D_SA / vmolec_SA
        
        Kn_sa = np.zeros_like(d_dry)
        
        Kn_sa[not_empty_mask] = 2 * lambda_sa / d_dry[not_empty_mask]
        rho_sa = 1560 # kg/m^3
        
        
        dd_cond_dt = np.zeros_like(d_dry)
        dd_cond_dt[not_empty_mask] = (1/d_dry[not_empty_mask] * (4 * D_SA  / rho_sa) * 
        0.75*alpha_sa*(1+Kn_sa[not_empty_mask])/(1 + Kn_sa[not_empty_mask]**2 + Kn_sa[not_empty_mask] 
        + 0.283 * Kn_sa[not_empty_mask] * alpha_sa + 0.75*alpha_sa) * (C_SA_bulk - C_SA_surface[not_empty_mask]))
        
        dd_cond_dt[np.isnan(dd_cond_dt)] = 0
        dm_cond_dt = (np.pi * rho_sa * wet_pop.d_dist**2 / 2) * dd_cond_dt
        return dd_cond_dt, dm_cond_dt

    def nucleation(self,dry_pop,gas):
        T = 298.15  # Temp in K
        kB = 1.380649E-23   # Boltzmann constant; m2 kg s-2 K-1
        Na = 6.022E23       # Avogrados number
        M_sa = 0.11809      # molar mass of succinic acid, kg/m^3
        rho_SA_solid = 1560     # kg/m^3
        R = 8.3145          # J/mol*K
        rho_SA_vapor = gas.conc_SA_gas
        rho_v = rho_SA_vapor * Na / M_sa   # Number density of vapor
        rho_l = rho_SA_solid * Na / M_sa   # Number density of liquid (Solid)?
        gamma_SA = 67.8 /1000   # surface tension (SFE) of succinic acid in N/m
        m_molec = M_sa / Na     # Mass of 1 molecule of SA (kg)
        P_v = rho_SA_vapor * R * T /M_sa
        P_sat = 1.95E-3      # Saturation vapor pressure of SA in Pa
        v1 = P_v / np.sqrt(2*np.pi*m_molec*kB*T)
        S = P_v / P_sat
        d_mu = kB * T * np.log(S)
        Jo = (rho_v)**2/(rho_l) * np.sqrt(2*gamma_SA/(np.pi*m_molec))
        dG_star = (16 * np.pi/3) * (v1)**2*gamma_SA**3/((d_mu)**2)
        J = Jo * np.exp(-dG_star/(kB*T))   # nucleation rate; number of events per volume times time [#/m^3*s]
        J = J


    def advance(self, wet_pop, dry_pop, gas, dt):
        dm_dt = self.evaporate_water(wet_pop)
        dm_SA_dt = self.wet_SA_massloss(wet_pop)
        dd_cond_dt, dm_cond_dt = self.condensation(gas)
        wet_pop.wet_update_diameter(dm_dt, dm_SA_dt, dt)
        dry_pop.dry_update_diameter(dm_cond_dt,dt)
        gas.update_gas(dm_SA_dt, dm_cond_dt, wet_pop, dt)
        self.wet_to_dry(wet_pop, dry_pop, dt)

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

d_wet_cmd = wet_pop.d_dist[np.searchsorted(
    np.cumsum(wet_pop.N_dist), np.sum(wet_pop.N_dist) * 0.50)]

for step in range(grid.n_steps):
    drying.advance(wet_pop, dry_pop, gas, dt)
    total = wet_pop.N_dist.sum()
    if step % plot_interval == 0:
        wet_history.append(wet_pop.N_dist.copy())
        dry_history.append(dry_pop.N_dist.copy())

        gas_history.append(gas.conc_SA_gas)   # store gas concentration
        time_history.append(step * dt)
        
        dist_history.append(wet_pop.d_dist.copy())

dry_pop.get_summary_stats()
d_dry_cmd = dry_pop.d_dist[np.searchsorted(
    np.cumsum(dry_pop.N_dist), np.sum(dry_pop.N_dist) * 0.50)]
print(f"Wet CMD: {d_wet_cmd*1e9:.1f} nm")
print(f"Dry CMD : {d_dry_cmd*1e9:.1f} nm")
print(f"Shrinkage factor : {d_dry_cmd/d_wet_cmd:.4f}")
print(f"Solute vol frac  : {(d_dry_cmd/d_wet_cmd)**3:.5f}")
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

print(dry_pop.frac_SA)
print(dry_pop.frac_AS)
plt.show()

plt.figure()

plt.plot(time_history, gas_history)

plt.xlabel("Time (s)")
plt.ylabel("SA Gas Concentration (kg/m³)")
plt.title("Succinic Acid Gas Concentration vs Time")

plt.grid(True)

plt.show()

# Plot fraction of SA vs diameter - simple bar plot
plt.figure(figsize=(10, 6))

# Get final state data
final_frac_SA = dry_pop.frac_SA
final_dry_N = dry_history[-1]  # Final dry particle distribution

# Only plot bins with particles
mask_with_particles = final_dry_N > 0

if np.any(mask_with_particles):
    plt.bar(grid.centers[mask_with_particles], 
             final_frac_SA[mask_with_particles],
             width=grid.widths[mask_with_particles],
             alpha=0.7,
             edgecolor='black',
             linewidth=0.5)
    
    # Add horizontal line at 0.75 (original wet composition)
    plt.axhline(y=0.75, color='red', linestyle='--', alpha=0.7, 
                label='Original wet SA fraction (0.75)')

plt.xlabel('Particle Diameter (nm)')
plt.ylabel('SA Fraction')
plt.title('SA Fraction vs Particle Diameter (Final State)')
plt.xscale('log')
plt.ylim(0, 1.1)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


def plot_final_distribution():
    """
    Plot the final particle size distribution (both wet and dry) as a static image.
    This matches the appearance of the animation frames.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get final state data
    final_frame = 400  # Last frame
    d_phys = dist_history[final_frame] * 1e9  # Convert to nm
    n_wet = wet_history[final_frame]
    n_dry = dry_history[final_frame]
    
    mask = n_wet > 0
    
    # Wet bars — black outline, white fill
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
    
    # Final time in milliseconds
    final_time_ms = time_history[final_frame] * 1000
    plt.text(1.5,40000,f"t={final_time_ms:.2f} ms", size=12)

    plt.tight_layout()
    plt.show()
    
plot_final_distribution()