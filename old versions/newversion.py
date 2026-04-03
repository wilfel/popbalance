"""
SA Population Balance Model - WITH TARGETED SA LOSS FIXES
Focus: Fix aggressive wet-to-dry transfer and add mass balance tracking
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import ScalarFormatter

# --- Define values ---
d_min = 1           
d_max = 4000        
N_bins = 150         
dt = 0.00000001           
total_time = 0.001   
plot_interval_seconds = 0.00001       
plot_interval = int(plot_interval_seconds / dt) 
init_total_num = 1e9    
init_mean = 825  
init_sd = 1.68       

# --- Classes remain mostly the same, with key fixes ---
class SizeGrid:
    def __init__(self, d_min, d_max, n_bins, dt, total_time):
        self.edges = np.logspace(np.log10(d_min), np.log10(d_max), n_bins + 1)
        self.centers = np.sqrt(self.edges[:-1] * self.edges[1:])
        self.widths = self.edges[1:] - self.edges[:-1]
        self.n_bins = n_bins
        self.n_steps = int(total_time / dt)
        self.delta_log_d = np.log(self.edges[1]) - np.log(self.edges[0])
    
    def init_lognormal(self, N0, d_g, sigma_g):
        N_dens_dist_init = (N0 /
                (self.centers * np.sqrt(2*np.pi) * np.log(sigma_g)) *
                np.exp(-(np.log(self.centers / d_g))**2 /
                    (2 * (np.log(sigma_g))**2)))
        return N_dens_dist_init
    
class Population:
    def __init__(self, grid, N_dens_dist_init, wet):
        self.grid = grid    
        self.N_density = N_dens_dist_init
        self.d_dist = grid.centers.copy() * 1e-9   
        self.N_dist = self.N_density * grid.delta_log_d   
        self.V = np.pi/6 * self.d_dist**3   
        self.wet = wet
        if wet:
            self.wet_init()
        else:
            self.dry_init()
    
    def wet_init(self):
        n = self.grid.n_bins
        self.frac_SA = np.full(n, 0.75)
        self.frac_AS = np.full(n, 0.25)
        self.x_solute = np.full(n, 0.001)
        
        rho_water = 1000
        self.m = rho_water * self.V   
        self.m_solute = self.x_solute * self.m
        self.m_water = (1-self.x_solute) * self.m
        self.m_SA = self.x_solute * self.m * self.frac_SA

    def dry_init(self):
        n = self.grid.n_bins    
        self.m = np.zeros(n, dtype=float)
        self.V = np.zeros(n, dtype=float)
        self.frac_SA = np.zeros(n, dtype=float) 
        self.frac_AS = np.zeros(n, dtype=float)  
        self.x_solute = np.zeros(n, dtype=float)  
        self.m_SA = np.zeros(n, dtype=float)  
        
    def wet_update_diameter(self, dm_dt, dm_SA_dt, dt):
        """FIXED: Better mass balance and SA loss limiting"""
        # Update water mass
        self.m_water = np.maximum(self.m_water + dm_dt * dt, 0)
        
        # FIXED: More conservative SA mass loss limiting
        max_SA_loss = -self.m_SA / dt  
        dm_SA_dt_limited = np.maximum(dm_SA_dt, max_SA_loss)
        
        # Update SA mass first
        self.m_SA = np.maximum(self.m_SA + dm_SA_dt_limited * dt, 0)
        
        # FIXED: Ensure solute mass consistency
        # Only decrease solute mass if SA mass decreased
        dm_solute = np.where(dm_SA_dt_limited < 0, dm_SA_dt_limited, 0)
        self.m_solute = np.maximum(self.m_solute + dm_solute * dt, self.m_SA)
        
        self.m = self.m_solute + self.m_water
        
        # FIXED: Better fraction calculations with consistency checks
        self.x_solute = np.zeros_like(self.m)
        self.frac_SA = np.zeros_like(self.m)
        self.frac_AS = np.zeros_like(self.m)
        
        mask = self.m > 1e-20  
        self.x_solute[mask] = self.m_solute[mask] / self.m[mask]
        
        solute_mask = self.m_solute > 1e-20
        self.frac_SA[solute_mask] = self.m_SA[solute_mask] / self.m_solute[solute_mask]
        self.frac_AS[solute_mask] = 1 - self.frac_SA[solute_mask]
        
        # Ensure fractions are physical
        self.frac_SA = np.clip(self.frac_SA, 0, 1)
        self.frac_AS = np.clip(self.frac_AS, 0, 1)
        
        rho_water = 1000
        rho_solute = 1613
        rho_mix = np.full_like(self.m, rho_water)

        mask = self.m > 1e-20
        denominator = self.m_solute[mask]/rho_solute + self.m_water[mask]/rho_water
        valid_denom = denominator > 1e-30
        rho_mix[mask] = np.where(valid_denom, 
                                self.m[mask] / denominator, 
                                rho_water)
        
        self.V = np.where(rho_mix > 0, self.m / rho_mix, 0)
        self.d_dist = (self.V * 6/np.pi)**(1/3)
    
    def dry_update_diameter(self, dm_cond_dt, dt):
        """FIXED: Proper array operations for dry particle condensation"""
        self.m_SA += dm_cond_dt * dt  
        
        total_solute_mass = self.m_SA  
        self.m = total_solute_mass.copy()  
        
        mask = self.m > 1e-20
        self.frac_SA[mask] = self.m_SA[mask] / self.m[mask]
        self.frac_AS[mask] = 1 - self.frac_SA[mask]
        
        self.frac_SA[~mask] = 0.0
        self.frac_AS[~mask] = 0.0
        
    def get_summary_stats(self):
        pass
            
class GasPhase:
    def __init__(self):
        self.conc_SA_gas = 0    
        self.mass_SA_gas = 0
        self.conc_AS_gas = 0   
        
    def gas_update(self, dm_SA_dt, dm_cond_dt, dt, wet_N_dist, dry_N_dist):
        """FIXED: Proper mass balance for gas phase"""
        # SA evaporated from wet particles (mass loss rate * particles * time)
        bin_SA_evap = dm_SA_dt * wet_N_dist * dt
        self.mass_SA_gas += -np.sum(bin_SA_evap)
        
        # SA condensed onto dry particles
        bin_SA_cond = dm_cond_dt * dry_N_dist * dt
        self.mass_SA_gas += -np.sum(bin_SA_cond)

class DryingModel:
    def __init__(self):
        pass
        
    def evaporate_water(self, wet_pop):
        d = wet_pop.d_dist   
        A_d = 4*np.pi*(d/2)**2 
        rho_g = 2.416 
        Dv = 1.2E-5
        mu_g = 18.37E-6       
        RH_sat = 1      
        RH_bulk = 0.3
        P_sat = 0.0313 
        P_tot = 2.04138 
        y_sat = RH_sat*P_sat*18.016/(RH_sat*P_sat*18.016+(P_tot-RH_sat*P_sat)*28.97)
        y_bulk = RH_bulk*P_sat*18.016/(RH_bulk*P_sat*18.016+(P_tot-RH_bulk*P_sat)*28.97)
        v_g = 0.63662 
        Re = rho_g * v_g * d / mu_g 
        Sc = mu_g/(rho_g * Dv)  
        Sh = 2 + 0.552*Re**(1/2)*Sc**(1/3)  
        
        dm_dt = np.zeros_like(d)
        mask = d > 0
        dm_dt[mask] = -(A_d[mask] * rho_g * Dv *
                        (y_sat - y_bulk) *
                        Sh[mask]) / d[mask]        
        return(dm_dt)   
    
    def wet_to_dry(self, wet_pop, dry_pop):
        """FIXED: Much less aggressive drying conditions"""
        # FIXED: Higher saturation concentration for SA
        C_sat_SA = 830   # Increased from 83 to 830 kg/m³
        C_SA_current = wet_pop.x_solute * wet_pop.frac_SA * 1000
        
        not_empty_mask = wet_pop.m > 0     
        
        # FIXED: Much higher water threshold and water fraction requirement
        no_water_mask = wet_pop.m_water <= 1e-19  # Increased from 1e-25
        
        # FIXED: Add water fraction requirement - must lose 95% of water
        water_frac = np.where(wet_pop.m > 1e-20, 
                             wet_pop.m_water / wet_pop.m, 
                             0)
        low_water_frac_mask = water_frac <= 0.05  # Less than 5% water remaining
        
        above_sat_mask = C_SA_current >= C_sat_SA   
        
        # FIXED: More stringent drying conditions
        dry_mask = ((no_water_mask & low_water_frac_mask) | above_sat_mask) & not_empty_mask
 
        # Rest of wet_to_dry remains the same...
        rho_SA = 1560   
        rho_AS = 1770   
        mass_SA = wet_pop.m_solute * wet_pop.frac_SA
        mass_AS = wet_pop.m_solute * wet_pop.frac_AS
        dry_vol = mass_SA/rho_SA + mass_AS/rho_AS
        dry_d_dist = (dry_vol * 6/np.pi)**(1/3)
        
        wet_indices = np.where(dry_mask)[0]
        if len(wet_indices) > 0:
            stored_N_dist = wet_pop.N_dist[dry_mask].copy()
            stored_m = wet_pop.m[dry_mask].copy()
            stored_x_solute = wet_pop.x_solute[dry_mask].copy()
            stored_frac_SA = wet_pop.frac_SA[dry_mask].copy()
            stored_frac_AS = wet_pop.frac_AS[dry_mask].copy()
            stored_m_SA = wet_pop.m_SA[dry_mask].copy()
            
            dry_bins = np.searchsorted(wet_pop.grid.edges, dry_d_dist[dry_mask]*1e9) - 1
            dry_bins = np.clip(dry_bins, 0, wet_pop.grid.n_bins-1)
            
            for i, idx in enumerate(wet_indices):
                j = dry_bins[i]
                
                if dry_pop.N_dist[j] > 0:
                    total_mass = dry_pop.m[j] + stored_m[i]
                    if total_mass > 0:
                        dry_pop.x_solute[j] = (dry_pop.x_solute[j] * dry_pop.m[j] + 
                                              stored_x_solute[i] * stored_m[i]) / total_mass
                        dry_pop.frac_SA[j] = (dry_pop.frac_SA[j] * dry_pop.m[j] + 
                                             stored_frac_SA[i] * stored_m[i]) / total_mass
                        dry_pop.frac_AS[j] = (dry_pop.frac_AS[j] * dry_pop.m[j] + 
                                             stored_frac_AS[i] * stored_m[i]) / total_mass
                        dry_pop.m_SA[j] = (dry_pop.m_SA[j] + stored_m_SA[i])
                else:
                    dry_pop.x_solute[j] = stored_x_solute[i]
                    dry_pop.frac_SA[j] = stored_frac_SA[i]
                    dry_pop.frac_AS[j] = stored_frac_AS[i]
                    dry_pop.m_SA[j] = stored_m_SA[i]
                
                dry_pop.N_dist[j] += stored_N_dist[i]
                dry_pop.m[j] += stored_m[i]
            
            mask_dry_particles = dry_pop.N_dist > 0
            if np.any(mask_dry_particles):
                dry_pop.d_dist[mask_dry_particles] = dry_pop.grid.centers[mask_dry_particles] * 1e-9
        
        wet_pop.m_water[dry_mask] = 0   
        wet_pop.N_dist[dry_mask] = 0    
        wet_pop.m[dry_mask] = 0
        wet_pop.x_solute[dry_mask] = 0
        wet_pop.frac_SA[dry_mask] = 0
        wet_pop.frac_AS[dry_mask] = 0
        wet_pop.m_SA[dry_mask] = 0

    def wet_SA_massloss(self, wet_pop, gas):
        """Same as corrected version with proper mass limiting"""
        d_m = wet_pop.d_dist
        T = 298.15      
        R = 8.314   
        M_sa = 0.11809 
        M_w = 0.01806   
        A_d = 4*np.pi*(d_m/2)**2 
        gamma_SA = 1        
        p_sat_SA = 2.55E-5      
        
        conc_SA_liq = wet_pop.x_solute * wet_pop.frac_SA * 1000
        total_moles = (conc_SA_liq/M_sa) + ((1000 - conc_SA_liq)/M_w)
        molefrac_SA_liq = np.where(total_moles > 0, 
                                  (conc_SA_liq/M_sa) / total_moles, 
                                  0)
        
        p_SA_surface = molefrac_SA_liq * gamma_SA * p_sat_SA
        p_SA_bulk = (gas.conc_SA_gas * R * T)/M_sa
        rho_g = 2.416 
        mu_g = 18.37E-6       
        D_SA = 2E-6         
        v_g = 0.63662 
        Re_SA = rho_g * v_g * d_m / mu_g
        Sc_SA = mu_g/(rho_g * D_SA)
        Sh_SA = 2 + 0.552*Re_SA**(1/2)*Sc_SA**(1/3)
        k_mass = Sh_SA * D_SA / d_m
        C_SA_surface = p_SA_surface * M_sa / (R * T)
        C_SA_bulk    = p_SA_bulk    * M_sa / (R * T)
        
        dm_SA_dt_theoretical = -k_mass * A_d * (C_SA_surface-C_SA_bulk)  
        
        # Limit mass transfer to available SA mass
        max_SA_available = wet_pop.m_SA  
        max_loss_rate = -max_SA_available / dt  
        dm_SA_dt = np.maximum(dm_SA_dt_theoretical, max_loss_rate)
        
        return dm_SA_dt
    
    def condensation(self, gas, wet_pop, dry_pop):
        """Fixed version without divide by zero warnings"""
        not_empty_mask = dry_pop.N_dist > 0
        d_dry = np.zeros_like(dry_pop.N_dist)
        d_dry[not_empty_mask] = dry_pop.d_dist[not_empty_mask]
        
        T = 298.15      
        R = 8.314   
        N = 1e9    
        M_sa = 0.11809 
        M_w = 0.01806   
        D_SA = 2E-6         
        n_density = 1E15    
        vol = N / n_density
        gas.conc_SA_gas = gas.mass_SA_gas / vol
        
        gamma_SA = 1        
        p_sat_SA = 2.55E-5      
        
        conc_SA_liq = wet_pop.x_solute * wet_pop.frac_SA * 1000
        total_moles = (conc_SA_liq/M_sa) + ((1000 - conc_SA_liq)/M_w)
        molefrac_SA_liq = np.where(total_moles > 0, 
                                  (conc_SA_liq/M_sa) / total_moles, 
                                  0)
        
        p_SA_surface = molefrac_SA_liq * gamma_SA * p_sat_SA
        p_SA_bulk = (gas.conc_SA_gas * R * T)/M_sa
        C_SA_surface = p_SA_surface * M_sa / (R * T)
        C_SA_bulk    = p_SA_bulk    * M_sa / (R * T)

        alpha_sa = 1    
        vmolec_SA = (8 * R * T / (np.pi * M_sa))**(1/2)
        lambda_sa = 3 * D_SA / vmolec_SA
        
        # FIXED: No divide by zero warning
        Kn_sa = np.zeros_like(d_dry)
        mask_nonzero = d_dry > 0
        Kn_sa[mask_nonzero] = 2 * lambda_sa / d_dry[mask_nonzero]
        
        rho_sa = 1560 
        
        denominator = (1 + Kn_sa**2 + Kn_sa + 0.283 * Kn_sa * alpha_sa + 0.75*alpha_sa)
        condensation_factor = np.zeros_like(d_dry)
        valid_mask = mask_nonzero & (denominator > 0)
        condensation_factor[valid_mask] = (
            (4 * D_SA / rho_sa) * 0.75*alpha_sa*(1+Kn_sa[valid_mask]) / denominator[valid_mask]
        )
        
        dd_cond_dt = np.zeros_like(d_dry)
        dd_cond_dt[valid_mask] = (condensation_factor[valid_mask] / d_dry[valid_mask] * 
                                 (C_SA_bulk - C_SA_surface[valid_mask]))
        
        dm_cond_dt = (np.pi * rho_sa * wet_pop.d_dist**2 / 2) * dd_cond_dt
        return dd_cond_dt, dm_cond_dt
    
    def advance(self, wet_pop, dry_pop, gas, dt):
        dm_dt = self.evaporate_water(wet_pop)
        dm_SA_dt = self.wet_SA_massloss(wet_pop, gas)
        dd_cond_dt, dm_cond_dt = self.condensation(gas, wet_pop, dry_pop)
        wet_pop.wet_update_diameter(dm_dt, dm_SA_dt, dt)
        dry_pop.dry_update_diameter(dm_cond_dt, dt)
        gas.gas_update(dm_SA_dt, dm_cond_dt, dt, wet_pop.N_dist, dry_pop.N_dist)
        self.wet_to_dry(wet_pop, dry_pop)

# --- Set up Grid ---
grid = SizeGrid(d_min, d_max, N_bins, dt, total_time)
N_dens_dist_init = grid.init_lognormal(init_total_num, init_mean, init_sd)

# --- Set up Populations ---
wet_pop = Population(grid, N_dens_dist_init, wet=True)
dry_pop = Population(grid, np.zeros_like(N_dens_dist_init), wet=False)
gas = GasPhase()
drying = DryingModel()

# ADDED: Track initial SA mass for mass balance checking
initial_total_SA = np.sum(wet_pop.N_dist * wet_pop.m_SA)
print(f"Initial total SA mass: {initial_total_SA:.2e} kg")

monitor_bins = np.arange(wet_pop.grid.n_bins - 30, wet_pop.grid.n_bins-20)  

# --- Precompute time evolution WITH MASS BALANCE TRACKING ---
wet_history = []
dry_history = []
gas_history = []
time_history = []
dist_history = []

d_wet_cmd = wet_pop.d_dist[np.searchsorted(
    np.cumsum(wet_pop.N_dist), np.sum(wet_pop.N_dist) * 0.50)]

for step in range(grid.n_steps):
    drying.advance(wet_pop, dry_pop, gas, dt)
    total = wet_pop.N_dist.sum()
    if step % plot_interval == 0:
        wet_history.append(wet_pop.N_dist.copy())
        dry_history.append(dry_pop.N_dist.copy())
        gas_history.append(gas.conc_SA_gas)   
        time_history.append(step * dt)
        dist_history.append(wet_pop.d_dist.copy())
        
    # ADDED: Mass balance tracking every 10 intervals
    if step % (plot_interval * 10) == 0:
        # Track total SA in all phases
        total_SA_wet = np.sum(wet_pop.N_dist * wet_pop.m_SA)
        total_SA_dry = np.sum(dry_pop.N_dist * dry_pop.m_SA)
        total_SA_gas = gas.mass_SA_gas
        total_SA = total_SA_wet + total_SA_dry + total_SA_gas
        
        print(f"Step {step}: SA masses - Wet: {total_SA_wet:.2e}, "
              f"Dry: {total_SA_dry:.2e}, Gas: {total_SA_gas:.2e}, "
              f"Total: {total_SA:.2e}")
        
        # Check for mass balance violation
        if step > 0:
            mass_balance_error = abs(total_SA - initial_total_SA) / initial_total_SA
            if mass_balance_error > 0.1:  # 10% error
                print(f"❌ MASS BALANCE ERROR: {mass_balance_error*100:.1f}% loss")
            
        # Check average SA fractions
        if np.sum(wet_pop.N_dist) > 0:
            avg_frac_SA_wet = np.average(wet_pop.frac_SA, weights=wet_pop.N_dist)
            print(f"Average SA fraction in wet particles: {avg_frac_SA_wet:.3f}")
        
        if np.sum(dry_pop.N_dist) > 0:
            avg_frac_SA_dry = np.average(dry_pop.frac_SA, weights=dry_pop.N_dist)
            print(f"Average SA fraction in dry particles: {avg_frac_SA_dry:.3f}")
        print("-" * 60)

dry_pop.get_summary_stats()
d_dry_cmd = dry_pop.d_dist[np.searchsorted(
    np.cumsum(dry_pop.N_dist), np.sum(dry_pop.N_dist) * 0.50)]
print(f"Wet CMD: {d_wet_cmd*1e9:.1f} nm")
print(f"Dry CMD : {d_dry_cmd*1e9:.1f} nm")
print(f"Shrinkage factor : {d_dry_cmd/d_wet_cmd:.4f}")
print(f"Solute vol frac  : {(d_dry_cmd/d_wet_cmd)**3:.5f}")

# Final SA analysis
final_SA_wet = np.sum(wet_pop.N_dist * wet_pop.m_SA)
final_SA_dry = np.sum(dry_pop.N_dist * dry_pop.m_SA)
final_SA_gas = gas.mass_SA_gas
final_SA_total = final_SA_wet + final_SA_dry + final_SA_gas

print(f"\nFINAL SA MASS BALANCE:")
print(f"Initial SA: {initial_total_SA:.2e} kg")
print(f"Final SA:   {final_SA_total:.2e} kg")
print(f"Net loss:   {(initial_total_SA - final_SA_total):.2e} kg")
print(f"% retained: {(final_SA_total/initial_total_SA)*100:.1f}%")

# Check final SA fractions
print(f"\nFinal dry SA fractions - min: {np.min(dry_pop.frac_SA):.3f}, max: {np.max(dry_pop.frac_SA):.3f}")
valid_dry = dry_pop.N_dist > 0
if np.any(valid_dry):
    avg_final_SA_frac = np.average(dry_pop.frac_SA[valid_dry], weights=dry_pop.N_dist[valid_dry])
    print(f"Average final dry SA fraction: {avg_final_SA_frac:.3f}")

# --- Animation (same as before) ---
fig, ax = plt.subplots()

def update(frame):
    ax.cla()
    d_phys = dist_history[frame] * 1e9
    n_wet = wet_history[frame]
    n_dry = dry_history[frame]
    mask = n_wet > 0

    ax.bar(d_phys[mask], n_wet[mask] / grid.delta_log_d,
           width=grid.widths[mask] * (d_phys[mask] / grid.centers[mask]),
           align='center', edgecolor='black', linewidth=0.8, label='Wet')

    ax.bar(grid.edges[:-1], n_dry / grid.delta_log_d, width=grid.widths,
           align='edge', edgecolor='red', linewidth=0.8, alpha=0.6, label='Dry')

    ax.set_xscale('log')
    ax.set_xlim(d_min, d_max)
    ax.set_ylim(0, np.max([w.max() for w in wet_history]) / grid.delta_log_d * 1.1)
    ax.set_xlabel('Particle diameter [nm]')
    ax.set_ylabel('Number density [#/cm³]')
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.ticklabel_format(axis='x', style='plain')
    ax.legend()

    t = frame * plot_interval_seconds * 1000
    ax.set_title(f"Simulated Drying - FIXED VERSION (t = {t:.2f} ms)")

ani = FuncAnimation(fig, update, frames=len(wet_history), interval=150, blit=False)
print(dry_pop.frac_SA)
plt.show()

plt.figure()
plt.plot(time_history, gas_history)
plt.xlabel("Time (s)")
plt.ylabel("SA Gas Concentration (kg/m³)")
plt.title("Succinic Acid Gas Concentration vs Time - FIXED")
plt.grid(True)
plt.show()