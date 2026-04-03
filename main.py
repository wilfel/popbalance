"""
AS/SA Population Balance Model Code
Will Feldscher

New version to clean up structure (not tracking origin bin, etc.) and implement nucleation
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from dataclasses import dataclass, field
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
    
@dataclass
class Cohort:
    """
    Represents a group of particles that nucleated (or dried) at the same time.
    All particles in a cohort share the same diameter and grow together.
    
    Attributes:
        N: number of particles in this cohort
        d: diameter of a single particle (m)
        m: mass of a single particle (kg)
        m_SA: mass of succinic acid in a single particle (kg)
        m_AS: mass of ammonium sulfate in a single particle (kg)
        t_born: simulation time when cohort was created (s)
    """
    N: float
    d: float
    m: float
    m_SA: float
    m_AS: float
    t_born: float = 0.0
    
class WetPopulation:
    """
    Represents the entire population of particles
    
    Attributes:
        grid (class): Contains all static information on grid where population is plotted
        N_density (np.ndarray): Array containing number density distribution evaluated at bin centers
        N_dist (np.ndarray): Array containing number distribution of particles in each bin
        
    Methods:
    """
    def __init__(self, grid, N_dens_dist_init):
        rho_water = 1000
        conc_SA = 0.75
        conc_AS = 0.25
        concentration = 0.001
        
        self.grid = grid    
        self.N_density = N_dens_dist_init
        self.d_dist = grid.centers.copy() * 1e-9   # bin centers (m)
        self.N_dist = self.N_density * grid.delta_log_d   # initial total number in each bin
        self.V = np.pi/6 * self.d_dist**3   # volume of each center of a SINGLE particle (m^3)
        n = self.grid.n_bins
        self.frac_SA = np.full(n, conc_SA)
        self.frac_AS = np.full(n, conc_AS)
        self.x_solute = np.full(n, concentration)
        self.m = rho_water * self.V   # mass of a single particle
        self.m_solute = self.x_solute * self.m
        self.m_water = (1-self.x_solute) * self.m
        self.m_SA = self.x_solute * self.m * self.frac_SA

    def wet_update_diameter(self, dm_dt,dm_SA_dt, dt):
        """
        Updates diameter and other variables associated with the wet population based on a mass loss rate.

        Args:
            dm_dt (float): Mass loss rate of water [kg/s]
            dm_SA_dt (float): Mass loss rate of SA [kg/s]
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
        
class DryPopulation:
    """
    Lagrangian representation of dry/nucleated particles as discrete cohorts.
    Each cohort was created at a specific time; all particles within it share
    the same diameter and grow together via condensation.
    """
    def __init__(self):
        self.cohorts: list[Cohort] = []
        
    def add_cohort(self, N, d, m, m_SA, m_AS, t):
        """Add a new cohort (from nucleation or wet→dry transfer)."""
        if N > 0:
            self.cohorts.append(Cohort(N=N, d=d, m=m, m_SA=m_SA, m_AS=m_AS, t_born=t))
            
    def project_to_grid(self, grid):
        """
        Project cohort diameters onto the bin grid for plotting.
        Returns N_dist array (number per bin, not density).
        """
        N_dist = np.zeros(grid.n_bins)
        for c in self.cohorts:
            if c.N <= 0:
                continue
            d_nm = c.d * 1e9
            j = np.searchsorted(grid.edges, d_nm) - 1
            j = np.clip(j, 0, grid.n_bins - 1)
            N_dist[j] += c.N
        return N_dist
    
    def merge_similar_cohorts(self, tol=0.01):
        """
        Merge cohorts within fractional diameter of each other to keep
        the cohort list length manageable. Conserves total number and mass.
        """
        if len(self.cohorts) < 2:   # nothing to merge if there's less than 2 cohorts
            return
        self.cohorts = [c for c in self.cohorts if c.N > 0] # count cohorts as any with dry particles
        self.cohorts.sort(key=lambda c: c.d) # sort all cohorts by diameter, smallest to largest
        merged = [self.cohorts[0]] # initiialized empty merged list
        for c in self.cohorts[1:]: # iterate over every cohort
            prev = merged[-1]   # store neighbor cohort
            # check if difference between neighboring bins is within tolerance used, 1e-30 to prevent div 0
            if abs(c.d - prev.d) / max(prev.d, 1e-30) < tol:    
                total_N = prev.N + c.N
                # Number-weighted average of per-particle quantities if they're close enough
                prev.d   = (prev.d   * prev.N + c.d   * c.N) / total_N
                prev.m   = (prev.m   * prev.N + c.m   * c.N) / total_N
                prev.m_SA= (prev.m_SA* prev.N + c.m_SA* c.N) / total_N
                prev.m_AS= (prev.m_AS* prev.N + c.m_AS* c.N) / total_N
                prev.N   = total_N
            else:
                merged.append(c)
        self.cohorts = merged
        
class GasPhase:
    def __init__(self):
        self.vol          = 1e-6                        # m³
        self.conc_SA_gas  = 9.289e-08 * 210           # kg/m³
        self.mass_SA_gas  = self.conc_SA_gas * self.vol  # kg
        self.conc_AS_gas  = 0.0

    def remove_mass(self, delta_mass_kg):
        """Subtract mass from gas phase and update concentration."""
        self.mass_SA_gas  = max(self.mass_SA_gas - delta_mass_kg, 0.0)
        self.conc_SA_gas  = self.mass_SA_gas / self.vol

    def update_from_wet(self, dm_SA_dt, wet_N_dist, dt):
        """Update gas based on wet-particle SA flux."""
        self.mass_SA_gas += -np.sum(dm_SA_dt * wet_N_dist) * dt
        self.mass_SA_gas  = max(self.mass_SA_gas, 0.0)
        self.conc_SA_gas  = self.mass_SA_gas / self.vol
        
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
    
    def wet_SA_massloss(self, wet_pop,gas):
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
        return dm_SA_dt
    
    def condensation(self, cohort_pop, gas, dt):
        """
        Apply condensation to every cohort (dry particle group)
        Returns total mass (kg) transferred from gas to cohorts this timestep.
        """
        if not cohort_pop.cohorts:
            return 0.0

        T = 298.15
        R = 8.314
        M_sa = 0.11809
        D_SA   = 2e-6
        alpha  = 1.0
        rho_sa = 1560.0
        p_sat_SA = 2.55e-5      # Pa — solid-phase saturation vapour pressure

        # Vectorise over all cohorts
        d_arr = np.array([c.d for c in cohort_pop.cohorts])  # m

        p_SA_bulk    = gas.conc_SA_gas * R * T / M_sa
        C_SA_bulk    = p_SA_bulk * M_sa / (R * T)
        C_SA_surface = p_sat_SA * M_sa / (R * T)   # pure solid surface

        vmolec = (8 * R * T / (np.pi * M_sa))**(1/2)
        lam    = 3 * D_SA / vmolec
        Kn     = 2 * lam / d_arr

        # Fuchs-Sutugin transitional correction
        beta   = (0.75 * alpha * (1 + Kn) /
                  (Kn**2 + Kn + 0.283 * Kn * alpha + 0.75 * alpha))

        dd_dt  = (4 * D_SA / (rho_sa * d_arr)) * beta * (C_SA_bulk - C_SA_surface)
        dm_dt  = np.pi * rho_sa * d_arr**2 / 2 * dd_dt   # kg/s per particle

        total_mass_removed = 0.0
        for i, c in enumerate(cohort_pop.cohorts):
            delta_d = dd_dt[i] * dt
            delta_m = dm_dt[i] * dt
            c.d   = max(c.d + delta_d, 0.0)
            c.m   = max(c.m + delta_m, 0.0)
            c.m_SA= max(c.m_SA + delta_m, 0.0)   # all condensate is SA
            # Recompute diameter from mass for consistency
            c.d   = (c.m / rho_sa * 6 / np.pi)**(1/3) if c.m > 0 else 0.0
            total_mass_removed += delta_m * c.N

        return total_mass_removed
    
    def nucleation(self, cohort_pop, gas, t):
        """
        Compute nucleation rate via CNT. If J_vol >= 1, create a new cohort
        at the critical cluster size and subtract mass from gas.
        """
        T        = 298.15
        kB       = 1.380649e-23
        Na       = 6.022e23
        M_sa     = 0.11809
        rho_sa   = 1560.0
        vol      = 1e-6
        R        = 8.3145
        P_sat    = 1.95e-3

        rho_v    = gas.conc_SA_gas
        P_v      = rho_v * R * T / M_sa
        S        = P_v / P_sat

        if S <= 1.0:
            return

        # CNT quantities
        gamma    = 67.8e-3          # N/m
        m_molec  = M_sa / Na        # kg/molecule
        v_l      = M_sa / (Na * rho_sa)    # m³/molecule  (eq. 3.23)
        rho_v_num= rho_v * Na / M_sa       # number density, #/m³
        rho_l_num= rho_sa * Na / M_sa

        d_mu     = kB * T * np.log(S)

        # Critical cluster size (eq. 3.27)
        s1       = (36 * np.pi)**(1/3) * v_l**(2/3)
        theta_inf= gamma * s1 / (kB * T)
        nc       = (2 * theta_inf / (3 * np.log(S)))**3

        # Nucleation barrier (eq. 3.28)
        dG_star  = (16 * np.pi / 3) * v_l**2 * gamma**3 / d_mu**2

        # Kinetic prefactor
        Jo       = (rho_v_num**2 / rho_l_num) * np.sqrt(2 * gamma / (np.pi * m_molec))

        J        = Jo * np.exp(-dG_star / (kB * T))    # #/m³/s
        J_vol    = int(np.round(J * vol))               # whole nucleation events

        if J_vol <= 0:
            return

        # --- Critical cluster properties ---
        nc_int       = max(int(np.round(nc)), 1)
        m_cluster    = nc_int * m_molec                  # kg per cluster
        d_cluster    = (m_cluster / rho_sa * 6 / np.pi)**(1/3)  # m

        # --- Create new cohort ---
        cohort_pop.add_cohort(
            N    = J_vol,
            d    = d_cluster,
            m    = m_cluster,
            m_SA = m_cluster,
            m_AS = 0.0,
            t    = t
        )

        # --- Remove mass from gas ---
        mass_transferred = J_vol * m_cluster
        gas.remove_mass(mass_transferred)
        
    def wet_to_dry(self, wet_pop, cohort_pop, t):
        """
        Detect fully dried wet-pop bins and transfer them to new cohorts.
        Each dried bin becomes its own cohort (then merge_similar_cohorts
        consolidates nearby ones).
        """
        C_sat_SA       = 83.0
        C_SA_current   = wet_pop.x_solute * wet_pop.frac_SA * 1000
        not_empty_mask = wet_pop.m > 0
        no_water_mask  = wet_pop.m_water <= 1e-25
        above_sat_mask = C_SA_current >= C_sat_SA
        dry_mask       = (no_water_mask | above_sat_mask) & not_empty_mask

        if not np.any(dry_mask):
            return

        rho_SA = 1560.0;  rho_AS = 1770.0

        for i in np.where(dry_mask)[0]:
            N_bin   = wet_pop.N_dist[i]
            if N_bin <= 0:
                continue
            m_sol   = wet_pop.m_solute[i]
            m_SA_p  = wet_pop.m_SA[i]
            m_AS_p  = m_sol - m_SA_p
            fSA     = m_SA_p / m_sol if m_sol > 0 else 0.0
            fAS     = 1.0 - fSA
            rho_tot = fSA * rho_SA + fAS * rho_AS
            V_p     = m_sol / rho_tot if rho_tot > 0 else 0.0
            d_p     = (V_p * 6 / np.pi)**(1/3) if V_p > 0 else 0.0

            cohort_pop.add_cohort(
                N    = N_bin,
                d    = d_p,
                m    = m_sol,
                m_SA = m_SA_p,
                m_AS = m_AS_p,
                t    = t
            )

        # Zero out the dried bins in wet population
        wet_pop.N_dist[dry_mask]    = 0
        wet_pop.m[dry_mask]         = 0
        wet_pop.m_water[dry_mask]   = 0
        wet_pop.m_solute[dry_mask]  = 0
        wet_pop.m_SA[dry_mask]      = 0
    

    def advance(self, wet_pop, cohort_pop, gas, dt, t):
        # --- Wet population ---
        dm_water_dt = self.evaporate_water(wet_pop)
        dm_SA_dt    = self.wet_SA_massloss(wet_pop, gas)
        wet_pop.wet_update_diameter(dm_water_dt, dm_SA_dt, dt)

        # --- Dry cohorts: grow via condensation ---
        mass_cond = self.condensation(cohort_pop, gas, dt)
        gas.remove_mass(mass_cond)

        # --- Nucleation: may add new cohort ---
        self.nucleation(cohort_pop, gas, t)

        # --- Update gas from wet-particle SA flux ---
        gas.update_from_wet(dm_SA_dt, wet_pop.N_dist, dt)

        # --- Transfer fully dried wet particles to cohort population ---
        self.wet_to_dry(wet_pop, cohort_pop, t)

        # --- Merge near-identical cohorts to control list length ---
        cohort_pop.merge_similar_cohorts(tol=0.01)


# =============================================================================
# SETUP
# =============================================================================

grid            = SizeGrid(d_min, d_max, N_bins, dt, total_time)
N_dens_init     = grid.init_lognormal(init_total_num, init_mean, init_sd)
wet_pop         = WetPopulation(grid, N_dens_init)
cohort_pop      = DryPopulation()
gas             = GasPhase()
drying          = DryingModel()

# =============================================================================
# TIME LOOP
# =============================================================================

wet_history     = []
dry_history     = []    # stored as projected N_dist arrays for plotting
gas_history     = []
time_history    = []
dist_history    = []

for step in range(grid.n_steps):
    t = step * dt
    drying.advance(wet_pop, cohort_pop, gas, dt, t)

    if step % plot_interval == 0:
        wet_history.append(wet_pop.N_dist.copy())
        dry_history.append(cohort_pop.project_to_grid(grid))
        gas_history.append(gas.conc_SA_gas)
        time_history.append(t)
        dist_history.append(wet_pop.d_dist.copy())

        n_cohorts = len(cohort_pop.cohorts)

# =============================================================================
# SUMMARY STATS
# =============================================================================

all_d   = np.array([c.d for c in cohort_pop.cohorts]) * 1e9    # nm
all_N   = np.array([c.N for c in cohort_pop.cohorts])

if all_N.sum() > 0:
    cumN    = np.cumsum(all_N[np.argsort(all_d)])
    d_sort  = np.sort(all_d)
    d_dry_cmd = d_sort[np.searchsorted(cumN, cumN[-1] * 0.5)]
else:
    d_dry_cmd = 0.0

d_wet_cmd = wet_pop.d_dist[np.searchsorted(
    np.cumsum(wet_pop.N_dist), np.sum(wet_pop.N_dist) * 0.50)] * 1e9

print(f"\nWet CMD : {d_wet_cmd:.1f} nm")
print(f"Dry CMD : {d_dry_cmd:.1f} nm")
if d_wet_cmd > 0:
    sf = d_dry_cmd / d_wet_cmd
    print(f"Shrinkage factor : {sf:.4f}")
    print(f"Solute vol frac  : {sf**3:.5f}")

# =============================================================================
# ANIMATION
# =============================================================================

fig, ax = plt.subplots()

def update(frame):
    ax.cla()
    d_phys  = dist_history[frame] * 1e9
    n_wet   = wet_history[frame]
    n_dry   = dry_history[frame]
    mask    = n_wet > 0

    ax.bar(d_phys[mask], n_wet[mask] / grid.delta_log_d,
           width=grid.widths[mask] * (d_phys[mask] / grid.centers[mask]),
           align='center', edgecolor='black', linewidth=0.8, label='Wet')

    ax.bar(grid.centers, n_dry / grid.delta_log_d,
           width=grid.widths, align='center',
           edgecolor='red', linewidth=0.8, alpha=0.6, label='Dry')

    ax.set_xscale('log')
    ax.set_xlim(d_min, d_max)
    ax.set_ylim(0, max((w.max() for w in wet_history), default=1) / grid.delta_log_d * 1.1)
    ax.set_xlabel('Particle diameter [nm]')
    ax.set_ylabel('dN/d(log d)  [#]')
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.ticklabel_format(axis='x', style='plain')
    ax.legend()
    t_ms = time_history[frame] * 1000
    ax.set_title(f"Simulated Drying — Cohort Tracking  (t = {t_ms:.2f} ms)")

ani = FuncAnimation(fig, update, frames=len(wet_history), interval=150, blit=False)
plt.show()

# =============================================================================
# GAS CONCENTRATION PLOT
# =============================================================================

plt.figure()
plt.plot(time_history, gas_history)
plt.xlabel("Time (s)")
plt.ylabel("SA Gas Concentration (kg/m³)")
plt.title("Succinic Acid Gas Concentration vs Time")
plt.grid(True)
plt.tight_layout()
plt.show()

# =============================================================================
# COHORT DIAMETER HISTORY (shows individual cohort growth)
# =============================================================================

plt.figure()
for c in cohort_pop.cohorts[::max(1, len(cohort_pop.cohorts)//50)]:   # sample ~50 cohorts
    plt.scatter(c.t_born * 1000, c.d * 1e9, s=5, color='steelblue', alpha=0.6)
plt.xlabel("Time nucleated / dried  (ms)")
plt.ylabel("Final diameter  (nm)")
plt.title("Final Diameter of Each Cohort vs Birth Time")
plt.grid(True)
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