import torch
from sim_config import SimulationConfig, build_derived_quantities
from Bi_OU_core import run_simulation
from plot import plot_surface, plot_isotropic_psd

params = SimulationConfig()          
rho = params.rho_s

cfg = SimulationConfig()
derived = build_derived_quantities(cfg)

# Outputs
V_x, V_y, stream, vort, epsi = run_simulation(cfg, derived)

# Plots

# 2D + 3D
#plot_surface(stream[0,:,:,0], title="stream at t=0")
#plot_surface(V_x[0,:,:,0], title="Vx at t=0")
#plot_surface(V_y[0,:,:,0], title="Vy at t=0")
#plot_surface(vort[0,:,:,0], title="vort at t=0")
#plot_surface(epsi[0,:,:,0], title="epsi at t=0")

#plot_surface(stream[-1,:,:,0], title="stream at t=tf")
#plot_surface(V_x[-1,:,:,0], title="Vx at t=tf")
#plot_surface(V_y[-1,:,:,0], title="Vy at t=tf")
#plot_surface(vort[-1,:,:,0], title="vort at t=tf")
#plot_surface(epsi[-1,:,:,0], title="epsi at t=tf")

# Spectra (Spatial)
krOU, psdOU = plot_isotropic_psd(
    V_x[0,:,:,0],
    scale=1.6e23,
    label="Bi-OU",
    slope=-rho
)


