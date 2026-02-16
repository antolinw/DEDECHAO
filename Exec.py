import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew
from sim_config import Parameters
from Bi_OU_core import run_simulation
from Bi_OU_core_Wavelet import run_simulation_W
from plot import plot_img, plot_surface, plot_isotropic_psd, \
                 compute_correlations_notime, plot_radial_profile, \
                 save_field_gif_matplotlib, compute_spectrum, plot_spectrum, \
                 plot_spectrum_mean_std, zeta_n, compute_structure_function_polar, \
                    plot_structure_function_polar

       
cfg = Parameters()
c1 = cfg.c1
c2 = cfg.c2
rho = cfg.rho_s
dim = cfg.dim
gamma = cfg.gamma_t

if not cfg.wavelet:
    V_x, V_y, stream, vort, epsi = run_simulation(cfg)
else:
    V_x, V_y, stream, vort, epsi = run_simulation_W(cfg)


###################################### Plots ######################################
##### The outputs are 4D arrays with dimensions (time, x, y, ensemble member) #####
###################################################################################

### 2D + 3D instant plots :
#plot_surface(stream[0,:,:,0], title="stream at t=0")
#plot_surface(V_x[0,:,:,0], title="Vx at t=0")
#plot_surface(V_y[0,:,:,0], title="Vy at t=0")
plot_surface(vort[0,:,:,0], title="vort at t=0")
#plot_surface(epsi[0,:,:,0], title="epsi at t=0")

#plot_surface(stream[-1,:,:,0], title="stream at t=tf")
#plot_surface(V_x[-1,:,:,0], title="Vx at t=tf")
#plot_surface(V_y[-1,:,:,0], title="Vy at t=tf")
#plot_surface(vort[-1,:,:,0], title="vort at t=tf")
#plot_surface(epsi[-1,:,:,0], title="epsi at t=tf")

## Spectra (Spatial) :
krOU, psdOU = plot_isotropic_psd(
   V_x[0,:,:,0],
   label="Bi-OU",
   slope=-rho
)

## Structure functions computation :
#S2x, Fx, S3x = compute_correlations_notime(V_x[0, :, :,0])

## Plot of structure functions as images :
#plot_img(S2x[0,:,:],title="S2")

## Plot of radial S2 :
#r, radial_means = plot_radial_profile(
#   S2x[0, :, :],
#   slope= zeta_n(c1, c2, 2),
#   title="S2 Radial Intensity"
#)

## Plot of radial S3 :
#r, radial_means = plot_radial_profile(
#   S3x[0, :, :],
#   slope= zeta_n(c1, c2, 3),
#   title="S3 Radial Intensity"
#)

## Plot of radial S3 :
#r, radial_means = plot_radial_profile(
#   Fx[0, :, :],
#   slope= zeta_n(c1, c2, 4) - 2*zeta_n(c1, c2, 2),
#   title="Flatness"
#)

# Animation of vorticity over time (I took the first member of the ensemble here) :
#save_field_gif_matplotlib(vort[:,:,:,0], filename="Vort_r-6_c2-1_zeps-1.gif", fps=15)
#save_field_gif_matplotlib(epsi[:,:,:,0], filename="Epsi_r-6_c2-1_zeps-1.gif", fps=15)

###Temporal Spectrum take at a fixed point (here aprx the center of the domain)
signal = V_x[:, :, :, 0]

### Compute spectrum
w, ke_hat = compute_spectrum(signal, cfg.Delta_t)

### Plot spectrum averaged over space
# ke_mean = ke_hat.mean(axis=(1,2))
# ke_std  = ke_hat.std(axis=(1,2))
#plot_spectrum(w, ke_mean, slope=-gamma)

### Plot spectrum with mean and std
plot_spectrum_mean_std(
    w,
    ke_hat,
    slope=-gamma,
    std_factor=1.0
)

