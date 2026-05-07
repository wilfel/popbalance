"""
AS/SA Population Balance Model Code
Will Feldscher

Simulation of ammonium sulfate/succinic acid aerosol droplet drying and dry aerosol formation. Used
with the goal of demonstrating that a bimodal particle size distribution can be formed via 
evaporation and deposition/nucleation of succinic acid.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from dataclasses import dataclass
from matplotlib.ticker import ScalarFormatter
import os
import shutil

# --- Define parameters for simulation (these are some of the ones you'll likely want to change) ---
p_sat_SA_kgm = 9.289e-08   # saturated vapor pressure of succinic acid in kg/m^3 (constant)
conc_SA_gas_init = 0 * p_sat_SA_kgm  # concentration of succinic acid in vapor phase initially in the system in kg/m^3. 
simulated_volume = 1e-6 # simulated volume moving through system as 1 cm^3, default unit in SMPS
RH_bulk = 0.2   # relative humidity of the bulk air in the system
gamma_SA    = (67.8e-3) * 0.48  # Surface free energy of SA N/m

# --- Define general plotting / simulation values (these are some of the ones you'll likely want to change) ---
d_min = 1           # minimum diameter plotted (nm)
d_max = 4000        # maximum diameter plotted (nm)
N_bins = 150         # number of bins to create
dt = 0.000001           # length of one time step in simulation (s)
total_time = 0.5   # total simulation time (s)
plot_interval_seconds = 0.00001       # interval to plot (time length between screenshots plotted)
plot_interval = int(plot_interval_seconds / dt) # plot interval in terms of time steps
init_total_num = 5.63e7    # total number of particles initially
init_mean = 630 #825 initial geometric mean diameter of droplets (nm)
init_sd = 1.68       # initial geometric standard deviation of the lognormal dist (nm)


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
        N: number of particles in this group
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
    N_added: float = 0.0      # cumulative particles added via condensation/merging

class WetPopulation:
    """
    Represents the entire population of particles
    
    Attributes:
        grid (class): Contains all static information on grid where population is plotted
        N_density (np.ndarray): Array containing number density distribution evaluated at bin centers
        N_dist (np.ndarray): Array containing number distribution of particles in each bin
        d_dist (np.ndarray): Bin center diameters in meters (converted from nm)
        V (np.ndarray): Volume of a single particle at each bin center (m^3)
        frac_SA (np.ndarray): Per-bin Succinic acid mass fraction (default: 0.75)
        frac_AS (np.ndarray): Per-bin ammonium sulfate mass fraction (default: 0.25)
        x_solute (np.ndarray): Per-bin total solute concentration (g/L), uniform at initialization
        m (np.ndarray): Mass of a single water particle at each bin center (kg)
        m_solute (np.ndarray): Solute mass per single particle at each bin center (kg)
        m_water (np.ndarray): Water mass per single particle at each bin center (kg)
        m_SA (np.ndarray): Succinic acid mass per single particle at each bin center (kg)
        
    Methods:
        wet_update_diameter(dm_dt,dm_SA_dt,dt): pdates diameter and other variables associated with the wet population based on a mass loss rate.
    """
    def __init__(self, grid, N_dens_dist_init):
        rho_water = 1000    # density of water [kg/m^3]
        conc_SA = 0.75      # solute fraction (75 wt.%) or conc in g/L
        conc_AS = 0.25      # solute fraction (25 wt.%) or conc in g/L
        concentration = 0.001   # concentration of total solute in g/L
        
        self.grid = grid    
        self.N_density = N_dens_dist_init
        self.d_dist = grid.centers.copy() * 1e-9   # bin centers (m)
        self.N_dist = self.N_density * grid.delta_log_d   # initial total number in each bin
        self.V = np.pi/6 * self.d_dist**3   # volume of each center of a SINGLE particle (m^3)
        n = self.grid.n_bins
        self.frac_SA = np.full(n, conc_SA)  # array full of initial value for every droplet
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
        rho_solute = 1613 # combined density of solid SA/SA, density in kg/m^3
        rho_mix = np.full_like(self.m, rho_water) # Create array of density of water as default

        self.frac_SA = np.zeros_like(self.m_solute) # Create arrays of 0's as default
        self.frac_AS = np.zeros_like(self.m_solute) 
        
        self.frac_SA[mask] = self.m_SA[mask] / self.m_solute[mask]  # Update values for only bins with particles
        self.frac_AS[mask] = 1 - self.frac_SA[mask]
        
        rho_mix[mask] = self.m[mask] / (self.m_solute[mask]/rho_solute + self.m_water[mask]/rho_water) # Update only bins with particles
        self.V = self.m / rho_mix
        self.d_dist = (self.V * 6/np.pi)**(1/3)
        
class DryPopulation:
    """
    Representation of dry/nucleated particles as discrete groups (cohorts). Each group was created at a specific time. 
    All particles within it share the same diameter and grow together via condensation.
    
    Attributes:
        cohorts (list[Cohort]): List of all active cohorts, where each cohort represents a group of particles

    Methods:
        add_cohort: Register a new cohort from nucleation or wet-to-dry transfer.
        project_to_grid: Map cohort diameters onto a logarithmic bin grid for plotting.
        merge_similar_cohorts: Consolidate cohorts with similar diameters to limit computational complexity (otherwise it'd be too much)
    """
    
    def __init__(self):
        self.cohorts: list[Cohort] = [] # Create empty list of all groups of particles
        
    def add_cohort(self, N, d, m, m_SA, m_AS, t):
        """
        Add a new cohort from nucleation or wet-to-dry transfer.

        Does nothing if N <= 0.

        Args:
            N (float): Number of particles in this cohort.
            d (float): Initial particle diameter (m).
            m (float): Per-particle total dry mass (kg).
            m_SA (float): Per-particle succinic acid mass (kg).
            m_AS (float): Per-particle ammonium sulfate mass (kg).
            t (float): Birth time of the cohort (s).
        """
        
        if N > 0:
            self.cohorts.append(Cohort(N=N, d=d, m=m, m_SA=m_SA, m_AS=m_AS, t_born=t, N_added = N))
            
    def project_to_grid(self, grid):
        """
        Project cohort diameters onto a logarithmic bin grid for plotting.

        Each cohort's particle count is linearly interpolated between the two
        nearest bin edges (in log-diameter space), so that total number is conserved.
        Cohorts with N <= 0 are skipped.

        Args:
            grid: A bin-grid object containing all static info on grid where data is plotted.

        Returns:
            np.ndarray: Shape (n_bins) array of particle counts per bin (not a number density).
        """
        
        N_dist = np.zeros(grid.n_bins)     # Initialize empty distribution
        log_edges = np.log(grid.edges)
        
        for c in self.cohorts:
            if c.N <= 0:    # pass if that cohort of particles is empty
                continue
            d_nm = c.d * 1e9    # convert avg particle diameter to nm
            log_d = np.log(max(d_nm, grid.edges[0]))    # calculate log diameter
            j = np.searchsorted(log_edges, log_d) - 1   # find which size bin to put that cohort in
            j = np.clip(j, 0, grid.n_bins - 1)          # ensure the bin the cohort is placed in is in a valid range
            N_dist[j] += c.N    # add all the particles in a cohort to that bin for plotting
            
            # Linearly interpolate particle count across the two neighboring bins
            # to conserve total number particles while sharing across bins to make distribution smoother
            # *** THIS ADDRESSES PROBLEMS WITH NOISE IN THE DATA WITHOUT SIGNIFICANTLY CHANGING THE DISTRIBUTION
            if j < grid.n_bins - 1: # if not the last bin
                f = (log_d - log_edges[j]) / (log_edges[j+1] - log_edges[j])    # fractional position within bin
                f = np.clip(f, 0, 1)    # ensure valid value range
                N_dist[j]   += c.N * (1 - f)    # share across bins based on fraction
                N_dist[j+1] += c.N * f
            else:   # if the last bin
                N_dist[j] += c.N    # dont distribute, just place all particles in one bin
        return N_dist
        
    def merge_similar_cohorts(self, tol):
        """
        Merge cohorts within fractional diameter of each other to keep
        the number of cohorts short enough that runtime is reasonable. Conserves total number and mass.
        
        Args:
            tol: Maximum ratio of one cohort to another of similar size before they are considered separate.
                If ratio is less than tol then cohorts are merged.
                
        Returns:
            None
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
                prev.N_added = prev.N_added + c.N_added    # accumulate additions
                prev.d   = (prev.d   * prev.N + c.d   * c.N) / total_N
                prev.m   = (prev.m   * prev.N + c.m   * c.N) / total_N
                prev.m_SA= (prev.m_SA* prev.N + c.m_SA* c.N) / total_N
                prev.m_AS= (prev.m_AS* prev.N + c.m_AS* c.N) / total_N
                prev.N   = total_N
            else:
                merged.append(c)
        self.cohorts = merged
class GasPhase:
    """
    Represents the gas-phase succinic acid (SA) reservoir in the simulation volume.

    Attributes:
        vol (float): Simulation volume in m^3.
        conc_SA_gas (float): Current gas-phase SA concentration in kg/m^3.
        mass_SA_gas (float): Current gas-phase SA mass in kg.
        conc_AS_gas (float): Gas-phase ammonium sulfate concentration in kg/m^3 (unused).
    
    Methods:
        remove_mass(): Subtract a fixed mass of SA from the gas phase and update concentration.
        update_from_wet(): Update gas-phase SA mass based on condensation flux onto wet particles.
    """
    def __init__(self):
        self.vol          = simulated_volume
        self.conc_SA_gas  = conc_SA_gas_init
        self.mass_SA_gas  = self.conc_SA_gas * self.vol
        self.conc_AS_gas  = 0   # AS is nonvolatile

    def remove_mass(self, delta_mass_kg):
        """
        Subtract a fixed mass of SA from the gas phase and update concentration.

        Args:
            delta_mass_kg (float): Mass of SA to remove in kg.
        """
        self.mass_SA_gas  = max(self.mass_SA_gas - delta_mass_kg, 0)
        self.conc_SA_gas  = self.mass_SA_gas / self.vol

    def update_from_wet(self, dm_SA_dt, wet_N_dist, dt):
        """
        Update gas-phase SA mass based on condensation flux onto wet particles.

        Computes the total mass condensed across all size bins during timestep dt
        and subtracts it from the gas phase.

        Args:
            dm_SA_dt (np.ndarray): Per-cohort SA condensation rate in kg/s, shape (n_cohorts,).
            wet_N_dist (np.ndarray): Particle count per bin, shape (n_bins,).
            dt (float): Timestep duration in seconds.
        """
        self.mass_SA_gas += -np.sum(dm_SA_dt * wet_N_dist) * dt
        self.mass_SA_gas  = max(self.mass_SA_gas, 0)
        self.conc_SA_gas  = self.mass_SA_gas / self.vol
class DryingModel:
    """
    Represents all physical processes that evolve the aerosol population each timestep.

    Handles evaporation of water from wet droplets, succinic acid (SA) condensation
    flux to/from wet particles, homogeneous nucleation of new dry cohorts via Classical
    Nucleation Theory (CNT), transfer of fully dried wet bins into the cohort population,
    and condensational growth of existing dry cohorts. All processes are advanced together
    through the 'advance' method.
    
    """
    def __init__(self):
            pass
        
    def evaporate_water(self, wet_pop, RH_bulk):
        """
        Compute the water evaporation rate ,for each bin in the wet population.

        Uses the Froessling correlation for the Sherwood number to compute
        convective mass transfer from each droplet surface to the bulk gas.

        Args:
            wet_pop: WetPopulation object with a d_dist attribute (diameter array, m).
            RH_bulk (float): Relative humidity of bulk air

        Returns:
            np.ndarray: Per-bin water mass loss rate in kg/s, shape (n_bins,).
                        Values are negative (mass leaving the droplet).
        """
        # --- Constants ---
        Dv = 2.12E-5
        mu_g = 18.37E-6       # viscosity of air in Pa*s
        RH_sat = 1      # assume air immediately above droplet is fully saturated
        P_sat = 0.0313 # saturated vapor pressure of water at 25 C is 0.0313 atm
        P_tot = 2.04138 # total pressure in atm; 30 PSI
        v_g = 0.63662 # velocity of gas, m/s
        MW_water = 18.016   # molecular weight of water in g/mol
        MW_SA = 28.97       # molecular weight of SA in g/mol
        rho_g = 2.416 # density of air at P = 30 PSI, T = 25 C, kg/m^3

        # --- Calculations ---
        d = wet_pop.d_dist   # diameter (m)
        A_d = 4*np.pi*(d/2)**2 # surface area of droplet, m^2
        y_sat = RH_sat*P_sat*MW_water/(RH_sat*P_sat*MW_water+(P_tot-RH_sat*P_sat)*MW_SA)
        y_bulk = RH_bulk*P_sat*MW_water/(RH_bulk*P_sat*MW_water+(P_tot-RH_bulk*P_sat)*MW_SA)
        Re = rho_g * v_g * d / mu_g # reynolds number
        Sc = mu_g/(rho_g * Dv)  # schmidt number
        Sh = 2 + 0.552*Re**(1/2)*Sc**(1/3)  # sherwood number; Froessling eqn
        dm_dt = np.zeros_like(d)
        mask = d > 0    # only calculate for particles with d > 0
        dm_dt[mask] = -(A_d[mask] * rho_g * Dv *        # From Poozesh et al., 2022
                        (y_sat - y_bulk) *
                        Sh[mask]) / d[mask]        
        return(dm_dt)   # kg/s
    
    def wet_SA_massloss(self, wet_pop,gas):
        """
        Compute the SA condensation/evaporation rate for each bin in the wet population.

        Calculates the vapor pressure of SA at the droplet surface from the liquid-phase
        mole fraction (Raoult's law, activity coefficient = 1), then drives mass transfer
        against the bulk gas-phase SA concentration via the Froessling Sherwood equation.

        Args:
            wet_pop: WetPopulation object with d_dist, N_dist, x_solute, and frac_SA arrays.
            gas: GasPhase object providing the current bulk SA concentration (conc_SA_gas, kg/m^3).

        Returns:
            dm_SA_dt(np.ndarray): Per-bin SA mass transfer rate in kg/s, shape (n_bins,).
                                    Negative means net evaporation from droplet to gas.
        """
        # --- Constants ---
        T = 298.15      # temp in K
        R = 8.314   # J/molK
        M_sa = 0.11809 # kg/mol
        M_w = 0.01806   # kg/mol
        rho_g = 2.416 # density of air at P = 30 PSI, T = 25 C, kg/m^3
        mu_g = 18.37E-6       # viscosity of air in Pa*s
        D_SA = 9.52E-6         # check this value later
        v_g = 0.63662 # velocity of gas, m/s
        p_sat_SA = 1.95E-3      # Pa
        
        # --- Equations ---
        not_empty_mask = wet_pop.N_dist > 0
        d_m = np.zeros_like(wet_pop.d_dist)
        d_m[not_empty_mask] = wet_pop.d_dist[not_empty_mask]    # zero out empty bins to avoid divide by 0
        
        A_d = 4*np.pi*(d_m/2)**2 # surface area of droplet, m^2
        gamma_SA = 1        # activity coefficient (ideal solution assumption)
        conc_SA_liq = wet_pop.x_solute * wet_pop.frac_SA * 1000     # SA concentration in droplet, g/L
        molefrac_SA_liq = (conc_SA_liq/M_sa) / ((conc_SA_liq/M_sa) + ((1000 - conc_SA_liq)/M_w))    # mole fraction of SA in droplet
        p_SA_surface = molefrac_SA_liq * gamma_SA * p_sat_SA    # vapor pressure of SA at the droplet surface
        p_SA_bulk = (gas.conc_SA_gas * R * T)/M_sa              # vapor pressure of SA in the bulk droplet
        Re_SA = rho_g * v_g * d_m / mu_g        # Reynolds number
        Sc_SA = mu_g/(rho_g * D_SA)             # Schmidt number
        Sh_SA = 2 + 0.552*Re_SA**(1/2)*Sc_SA**(1/3) # Sherwood number
        k_mass = np.zeros_like(d_m) # initialize as 0
        k_mass[not_empty_mask] = Sh_SA[not_empty_mask] * D_SA / d_m[not_empty_mask] # mass transfer coefficient, m/s
        C_SA_surface = p_SA_surface * M_sa / (R * T)    # SA concentration at droplet surface, kg/m^3
        C_SA_bulk    = p_SA_bulk    * M_sa / (R * T)    # SA concentration at bulk, kg/m^3
        dm_SA_dt = -k_mass * A_d * (C_SA_surface-C_SA_bulk)  # SA mass loss rate, kg/s
        return dm_SA_dt
    
    def condensation(self, dry_pop, gas, dt):
        """
        Apply condensational growth to every dry cohort over one timestep (to dry particle group)

        Computes the diameter and mass growth rate for each cohort using the
        Fuchs-Sutugin transitional correction to account for non-continuum effects
        at small particle sizes (Knudsen number correction). Cohort diameters and
        masses are updated in-place, with diameter recomputed from mass for
        consistency. Returns the total SA mass removed from the gas phase.

        Args:
            dry_pop: CohortPopulation object whose cohorts list will be updated in-place.
            gas: GasPhase object providing the current bulk SA concentration (conc_SA_gas, kg/m^3).
            dt (float): Timestep duration in seconds.

        Returns:
            total_mass_removed (float): Total mass of SA transferred from gas to cohorts this timestep, in kg.
        """
        if not dry_pop.cohorts:
            return 0      # if empty stop

        # --- Constants ---
        T = 298.15  # temperature (K)
        R = 8.314   # gas constant (J/molK)
        M_sa = 0.11809  # MW of SA (kg/mol)
        D_SA = 9.52E-6  # mass diffusivity of SA vapor in air, m^2/s
        alpha  = 1    # mass accomodation coefficient (1, no surface resistance)
        rho_sa = 1560.0 # density of solid SA, kg/m^3
        p_sat_SA = 2.55e-5  # SA saturated vapor pressure, Pa

        # --- Calculations ---
        d_arr = np.array([c.d for c in dry_pop.cohorts])  # cohort diameters, m
        p_SA_bulk    = gas.conc_SA_gas * R * T / M_sa   # vapor pressure of SA in bulk, Pa
        C_SA_bulk    = p_SA_bulk * M_sa / (R * T)       # concentration of SA in bulk, kg/m^3
        C_SA_surface = p_sat_SA * M_sa / (R * T)   # concentration of SA at pure solid surface, kg/m^3

        vmolec = (8 * R * T / (np.pi * M_sa))**(1/2)    # mean molecular velocity of SA vapor, m/s
        lam    = 3 * D_SA / vmolec      # mean free path of SA molecules in gas, m
        Kn     = 2 * lam / d_arr        # Knudsen number

        # Fuchs-Sutugin transitional correction
        beta   = (0.75 * alpha * (1 + Kn) /
                  (Kn**2 + Kn + 0.283 * Kn * alpha + 0.75 * alpha))

        dd_dt  = (4 * D_SA / (rho_sa * d_arr)) * beta * (C_SA_bulk - C_SA_surface)  # growth rate in m/s
        dm_dt  = np.pi * rho_sa * d_arr**2 / 2 * dd_dt   # mass gain rate in kg/s per particle

        total_mass_removed = 0.0    # initialize total mass removed as 0
        for i, c in enumerate(dry_pop.cohorts):
            delta_d = dd_dt[i] * dt # diameter change this timestep
            delta_m = dm_dt[i] * dt # mass change this timestep
            c.d   = max(c.d + delta_d, 0.0) # update all values for mass/diameter for that cohort, ensure they dont decrease
            c.m   = max(c.m + delta_m, 0.0)
            c.m_SA= max(c.m_SA + delta_m, 0.0)   # all condensate is SA
            # Recompute diameter from mass for consistency
            c.d   = (c.m / rho_sa * 6 / np.pi)**(1/3) if c.m > 0 else 0.0
            total_mass_removed += delta_m * c.N # total mass removed from the gas phase by particle count in a cohort
        return total_mass_removed   # kg
    
    def nucleation(self, dry_pop, gas, t, gamma):
        """
        Compute the homogeneous nucleation rate via Classical Nucleation Theory (CNT)
        and, if nucleation occurs, create a new dry cohort at the critical cluster size.

        Evaluates the supersaturation ratio S = P_v / P_sat. If S <= 1, nucleation
        is thermodynamically impossible and the method returns immediately. Otherwise,
        computes the critical cluster size nc, the nucleation barrier dG*, and the
        steady-state nucleation rate J. The expected number of nucleation events in the
        simulation volume is drawn from a Poisson distribution. A new cohort is added
        to dry_pop and the corresponding mass is removed from the gas phase.

        Args:
            dry_pop: CohortPopulation object; a new cohort is appended if nucleation occurs.
            gas: GasPhase object providing the current bulk SA concentration (conc_SA_gas, kg/m^3).
            t (float): Current simulation time in seconds (recorded as cohort birth time).
            gamma_SA (float): Surface free energy of SA (N/m)
        
        Returns:
            None
        """
        # --- Constants ---
        T        = 298.15       # Temp in K
        kB       = 1.380649e-23 # Boltzmann constant in m^2*kg*s^-2*K^-1
        Na       = 6.022e23     # Avogadros number
        M_sa     = 0.11809      # Molecular weight of succinic acid kg/mol
        rho_sa   = 1560.0       # Density of solid succinic acid in kg/m^3
        vol      = 1e-6         # Volume simulated in m^3 (pocket of 1 cm^3)
        R        = 8.3145       # Gas constant in J/molK
        P_sat    = 1.95e-3      # Vapor pressure of succinic acid in Pa
        rho_v    = gas.conc_SA_gas  # Density of gas currently in kg/m^3
        P_v      = rho_v * R * T / M_sa # Vapor pressure of gas
        S        = P_v / P_sat

        if S <= 1.0:    # if supersaturation ratio is below 1, nucleation is never thermodynamically favorable
            return

        # CNT quantities, all eqns from Kalikmanov (2012)
        m_molec  = M_sa / Na        # kg/molecule
        v_l      = M_sa / (Na * rho_sa)    # m^3 /molecule  
        rho_v_num= rho_v * Na / M_sa       # molecular number density (vapor), #/m^3
        rho_l_num= rho_sa * Na / M_sa      # molecular number density (liquid), #/m^3

        d_mu     = kB * T * np.log(S)   # chemical potential difference driving force (J)

        # Critical cluster size for nucleation
        s1       = (36 * np.pi)**(1/3) * v_l**(2/3) # surface area prefactor for a sphere
        theta_inf= gamma * s1 / (kB * T)    # dimensionless surface area parameter
        nc       = (2 * theta_inf / (3 * np.log(S)))**3 # number of molecules in a critical cluster
        
        # Nucleation barrier
        dG_star  = (16 * np.pi / 3) * v_l**2 * gamma**3 / d_mu**2   # Free energy barrier, J

        # Kinetic prefactor
        Jo       = (rho_v_num**2 / rho_l_num) * np.sqrt(2 * gamma / (np.pi * m_molec))

        # Calculation
        J        = Jo * np.exp(-dG_star / (kB * T))    # nucleation rate in #/m³/s
        J_vol    = J*vol*dt              # whole nucleation events
        if J_vol > 0:
            N_new = np.random.poisson(J_vol)    # draw actual event count from a Poisson distribution so its a whole #
        else:
            N_new = 0
        if J_vol <= 0:
            return

        # --- Critical cluster properties ---
        nc_int       = max(int(np.round(nc)), 1)
        m_cluster    = nc_int * m_molec                  # kg per cluster
        d_cluster    = (m_cluster / rho_sa * 6 / np.pi)**(1/3)  # m

        # --- Create new cohort ---
        dry_pop.add_cohort(
            N    = N_new,
            d    = d_cluster,
            m    = m_cluster,
            m_SA = m_cluster,
            m_AS = 0.0,
            t    = t
        )

        # --- Remove mass from gas ---
        mass_transferred = N_new * m_cluster    # total mass transferred out of SA gas phase, kg
        gas.remove_mass(mass_transferred)
        
    def wet_to_dry(self, wet_pop, dry_pop, t):
        """
        Detect fully dried wet-population bins and transfer them into new dry cohorts.

        A bin is considered dry if it has lost all water (m_water <= 1e-25 kg) or if
        its SA concentration has reached or exceeded saturation (83 g/L). Each qualifying
        bin becomes its own cohort; the corresponding bins are then zeroed out in wet_pop.
        Cohort consolidation is left to merge_similar_cohorts in the advance loop.

        Args:
            wet_pop: WetPopulation object whose dried bins will be zeroed in-place.
            dry_pop: DryPopulation object; one new cohort is appended per dried bin.
            t (float): Current simulation time in seconds (recorded as cohort birth time).
        """
        C_sat_SA       = 83.0       # solubility limit of SA (saturation) in g/L
        C_SA_current   = wet_pop.x_solute * wet_pop.frac_SA * 1000  # current conc of SA in g/L
        not_empty_mask = wet_pop.m > 0
        no_water_mask  = wet_pop.m_water <= 1e-25   # less than this threshold considered empty
        above_sat_mask = C_SA_current >= C_sat_SA   # # SA concentration at or above solubility limit
        dry_mask       = (no_water_mask | above_sat_mask) & not_empty_mask # dry bins have no water, are more concentrated than saturation, and arent empty

        if not np.any(dry_mask):    # if none are dry stop
            return

        rho_SA = 1560.0 # Solid densities of AS and SA in kg/m^3
        rho_AS = 1770.0

        for i in np.where(dry_mask)[0]: # for every bin
            N_bin   = wet_pop.N_dist[i] # Number of particles
            if N_bin <= 0:
                continue    # if empty skip
            m_sol   = wet_pop.m_solute[i]       # total solute mass per particle, kg
            m_SA_p  = wet_pop.m_SA[i]           # SA mass per particle, kg
            m_AS_p  = m_sol - m_SA_p            # ammonium sulfate mass per particle, kg
            fSA     = m_SA_p / m_sol if m_sol > 0 else 0.0     # SA mass fraction
            fAS     = 1.0 - fSA                                 # AS mass fraction
            rho_tot = fSA * rho_SA + fAS * rho_AS               # mixture density, kg/m^3
            V_p     = m_sol / rho_tot if rho_tot > 0 else 0.0   # particle volume, m^3
            d_p     = (V_p * 6 / np.pi)**(1/3) if V_p > 0 else 0.0    # particle diameter, m

            # Create new cohort for the newly dried particles
            dry_pop.add_cohort( 
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
    

    def advance(self, wet_pop, dry_pop, gas, dt, t):
        """
        Advance all aerosol populations and the gas phase by one timestep.

        Applies each physical process in sequence: water evaporation and SA flux
        for wet droplets, condensational growth of dry cohorts, homogeneous nucleation,
        gas-phase update from wet-particle SA flux, drying and transfer of saturated
        wet bins, and cohort merging to control list length.

        Args:
            wet_pop: Wet population object updated in-place.
            cohort_pop: CohortPopulation object updated in-place.
            gas: GasPhase object updated in-place.
            dt (float): Timestep duration in seconds.
            t (float): Current simulation time in seconds.
        
        Returns:
            None
        """
        
        # --- Wet population: evaporate water and compute SA flux ---
        dm_water_dt = self.evaporate_water(wet_pop, RH_bulk)
        dm_SA_dt    = self.wet_SA_massloss(wet_pop, gas)
        wet_pop.wet_update_diameter(dm_water_dt, dm_SA_dt, dt)

        # --- Dry population: grow via SA condensation from gas phase ---
        mass_cond = self.condensation(dry_pop, gas, dt)
        gas.remove_mass(mass_cond)

        # --- Nucleation: may add a new cohort if gas is supersaturated ---
        self.nucleation(dry_pop, gas, t, gamma_SA)

        # --- Update gas-phase SA concentration from wet-particle flux ---
        gas.update_from_wet(dm_SA_dt, wet_pop.N_dist, dt)

        # --- Transfer fully dried wet bins into the cohort population ---
        self.wet_to_dry(wet_pop, dry_pop, t)

        # --- Merge near-identical cohorts to keep list length manageable ---
        dry_pop.merge_similar_cohorts(tol=0.02)

# =============================================================================
# SETUP
# =============================================================================

grid            = SizeGrid(d_min, d_max, N_bins, dt, total_time)
N_dens_init     = grid.init_lognormal(init_total_num, init_mean, init_sd)
wet_pop         = WetPopulation(grid, N_dens_init)
dry_pop         = DryPopulation()
gas             = GasPhase()
drying          = DryingModel()

# =============================================================================
# TIME LOOP
# =============================================================================

wet_history     = []
dry_history     = []    # stored as projected N_dist arrays for plotting
gas_history     = []
relative_gas_history = []
time_history    = []
dist_history    = []

for step in range(grid.n_steps):
    t = step * dt
    drying.advance(wet_pop, dry_pop, gas, dt, t)

    if step % plot_interval == 0:
        wet_history.append(wet_pop.N_dist.copy())
        dry_history.append(dry_pop.project_to_grid(grid))
        T = 298.15      # K
        R = 8.314   # J/molK
        M_sa = 0.11809 # kg/mol
        p_sat_SA = 1.95E-3      # Pa
        gas_history.append(gas.conc_SA_gas * R * T/M_sa)
        relative_gas_history.append((gas.conc_SA_gas * R * T/M_sa)/p_sat_SA)
        #         #C_sat_kgm3    = p_sat_SA    * M_sa / (R * T)
        time_history.append(t)
        dist_history.append(wet_pop.d_dist.copy())

        n_cohorts = len(dry_pop.cohorts)
        
        all_d_sorted = sorted([c.d * 1e9 for c in dry_pop.cohorts])
        if len(all_d_sorted) > 1:
            gaps = [(all_d_sorted[i+1] - all_d_sorted[i]) / all_d_sorted[i] 
                    for i in range(len(all_d_sorted)-1)]
            idx = int(np.argmax(gaps))
            print(f"t={t*1000:.2f} ms")


# =============================================================================
# SUMMARY STATS
# =============================================================================

all_d   = np.array([c.d for c in dry_pop.cohorts]) * 1e9    # nm
all_N   = np.array([c.N for c in dry_pop.cohorts])

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
    ax.set_xlim(d_min, 500)
    ax.set_ylim(0, max((w.max() for w in dry_history), default=1) / grid.delta_log_d * 1.1)
    ax.set_xlabel('Particle diameter [nm]', fontsize="12")
    ax.set_ylabel('dN/d(log d)  [#]',fontsize="12")
    ax.tick_params(axis='both', labelsize=11)
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
plt.plot(time_history, relative_gas_history)
plt.xlabel("Time (s)")
plt.ylabel(r"SA Supersaturation Ratio ($P_v / P_v^*$)", fontsize=12)
plt.xticks(fontsize = 11)
plt.yticks(fontsize = 11)
plt.grid(True)
plt.tight_layout()
plt.show()



def save_snapshots():
    """
    Save a snapshot of the particle size distribution every 0.5 ms into 'snapshots/'.
    Clears the folder at the start of every run.
    """
    snapshot_dir = "snapshots"
    if os.path.exists(snapshot_dir):
        shutil.rmtree(snapshot_dir)
    os.makedirs(snapshot_dir)

    snapshot_interval_ms = 1
    y_max = np.max([w.max() for w in dry_history]) / grid.delta_log_d * 1.1
    last_saved_ms = -np.inf

    for frame, t in enumerate(time_history):
        t_ms = t * 1000
        if t_ms - last_saved_ms < snapshot_interval_ms:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))

        d_phys = dist_history[frame] * 1e9
        n_wet  = wet_history[frame]
        n_dry  = dry_history[frame]
        mask   = n_wet > 0

        ax.bar(
            d_phys[mask],
            n_wet[mask] / grid.delta_log_d,
            width=grid.widths[mask] * (d_phys[mask] / grid.centers[mask]),
            align='center',
            edgecolor='black',
            linewidth=0.8,
            label='Wet'
        )
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
        ax.set_ylim(0, y_max)
        ax.set_xlabel('Particle diameter [nm]', fontsize=16)
        ax.set_ylabel('dN/d(log d)  [#]', fontsize=16)
        ax.tick_params(axis = "both", labelsize = 13)
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.ticklabel_format(axis='x', style='plain')
        ax.legend()
        ax.set_title(f"Simulated Drying  (t = {t_ms:.2f} ms)")

        filename = os.path.join(snapshot_dir, f"snapshot_{t_ms:07.3f}ms.png")
        fig.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)

        last_saved_ms = t_ms

    print(f"Saved {len(os.listdir(snapshot_dir))} snapshots to '{snapshot_dir}/'")

save_snapshots()

# =============================================================================
# FINAL COMPOSITION vs SIZE PLOT (wt.% SA vs diameter)
# =============================================================================

# --- Collect dry cohort data ---
d_dry = []
wtpct_SA_dry = []
weights_dry = []

for c in dry_pop.cohorts:
    if c.N <= 0 or c.m <= 0:
        continue
    d_nm = c.d * 1e9
    wt_pct = 100 * c.m_SA / c.m   # wt.% SA

    d_dry.append(d_nm)
    wtpct_SA_dry.append(wt_pct)
    weights_dry.append(c.N)

d_dry = np.array(d_dry)
wtpct_SA_dry = np.array(wtpct_SA_dry)
weights_dry = np.array(weights_dry)

# --- Optional: include remaining wet particles ---
mask = wet_pop.N_dist > 0
d_wet = wet_pop.d_dist[mask] * 1e9
wtpct_SA_wet = 100 * (wet_pop.m_SA[mask] / wet_pop.m[mask])
weights_wet = wet_pop.N_dist[mask]

# --- Combine (optional, comment out if you only want dry) ---
d_all = np.concatenate([d_dry, d_wet])
wtpct_all = np.concatenate([wtpct_SA_dry, wtpct_SA_wet])
weights_all = np.concatenate([weights_dry, weights_wet])

# =============================================================================
# PLOT
# =============================================================================

plt.figure(figsize=(8,6))

# Scatter (size vs composition)
plt.scatter(d_all, wtpct_all, s=10, alpha=0.6)

# Optional: size-weighted visualization (better physically)
# plt.scatter(d_all, wtpct_all, s=weights_all / weights_all.max() * 50, alpha=0.6)

plt.xscale('log')
plt.xlabel('Particle diameter [nm]', fontsize=12)
plt.ylabel('wt.% SA', fontsize=12)
plt.title('Final Particle Composition vs Size', fontsize=13)
plt.tight_layout()
plt.show()