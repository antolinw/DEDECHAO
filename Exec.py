import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew
from sim_config import SimulationConfig, build_derived_quantities
from Bi_OU_core import run_simulation
from plot import plot_img, plot_surface, plot_isotropic_psd, compute_correlations_notime, plot_radial_profile

params = SimulationConfig()          
c2 = params.c2
rho = params.rho_s
dim = params.dim

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
    label="Bi-OU",
    slope=-rho
)

S2x, Fx, S3x = compute_correlations_notime(V_x[0, :, :,0])
#plot_img(S2x[0,:,:])
#print(np.shape(S2x))

r, radial_means = plot_radial_profile(
    S2x[0, :, :],
    slope=rho + dim - 2,
    title="S2 Radial Intensity"
)

r, radial_means = plot_radial_profile(
    S3x[0, :, :],
    slope= c2 + (rho + dim - 2) / 2,
    title="S3 Radial Intensity"
)