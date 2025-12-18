import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew
from sim_config import SimulationConfig, build_derived_quantities
from Bi_OU_core import run_simulation
from plot import plot_img, plot_surface, plot_isotropic_psd, compute_correlations_notime, plot_radial_profile, save_field_gif_matplotlib

       
cfg = SimulationConfig()
c2 = cfg.c2
rho = cfg.rho_s
dim = cfg.dim
derived = build_derived_quantities(cfg)

# Outputs
V_x, V_y, stream, vort, epsi = run_simulation(cfg, derived)

########################### Plots ###########################

# 2D + 3D instant plots :
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

# Spectra (Spatial) :
krOU, psdOU = plot_isotropic_psd(
    V_x[0,:,:,0],
    label="Bi-OU",
    slope=-rho
)

# Structure functions computation :
S2x, Fx, S3x = compute_correlations_notime(V_x[0, :, :,0])

# Plot of structure functions as images :
#plot_img(S2x[0,:,:])
#print(np.shape(S2x))

# Plot of radial S2 :
r, radial_means = plot_radial_profile(
    S2x[0, :, :],
    slope=rho + dim - 2,
    title="S2 Radial Intensity"
)

# Plot of radial S3 :
r, radial_means = plot_radial_profile(
    S3x[0, :, :],
    slope= c2 + (rho + dim - 2) / 2,
    title="S3 Radial Intensity"
)

# Animation of vorticity over time (I took the first member of the ensemble here) :
save_field_gif_matplotlib(vort[:,:,:,0], filename="Vorticity.gif", fps=15)