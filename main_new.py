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
    def __init__(self, grid, N_dens_dist_init, wet):
        rho_water = 1000
        conc_SA = 0.75
        conc_AS = 0.25
        concentration = 0.001
        
        self.grid = grid    
        self.N_density = N_dens_dist_init
        self.d_dist = grid.centers.copy() * 1e-9   # bin centers (m)
        self.N_dist = self.N_density * grid.delta_log_d   # initial total number in each bin
        self.V = np.pi/6 * self.d_dist**3   # volume of each center of a SINGLE particle (m^3)
        self.wet = wet
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