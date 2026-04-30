import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew
from sim_config import Parameters
from Bi_OU_core import run_simulation
from Bi_OU_core_Wavelet import run_simulation_W
from plot import plot_img, plot_surface, plot_isotropic_psd, \
                 save_field_gif_matplotlib, compute_spectrum, plot_spectrum, \
                 plot_spectrum_mean_std, zeta_n, plot_structure_functions, plot_increment_pdfs, \
                 plot_structure_functions_3D

       
cfg = Parameters()
c1 = cfg.c1
c2 = cfg.c2
rho = cfg.rho_s
dim = cfg.dim
gamma = cfg.gamma_t
L = cfg.Lx
L_LS = cfg.L_LS
nu = cfg.nu_m
ell = (L_LS * np.sqrt(2 * nu)) / (2 * np.pi)
dx = cfg.dx
dt = cfg.Delta_t
T = cfg.Big_t

if not cfg.wavelet:
    V_x, V_y, stream, vort, epsi = run_simulation(cfg)
else:
    V_x, V_y, stream, vort, epsi = run_simulation_W(cfg)


print("Simulation completed. Now plotting results...")

###################################### Plots ######################################
##### The outputs are 4D arrays with dimensions (time, x, y, ensemble member) #####
###################################################################################

### 2D + 3D instant plots :
#plot_surface(stream[0,:,:,0], title="stream at t=0")
plot_surface(V_x[0,:,:,0], title="Vx at t=0")
#plot_surface(V_y[0,:,:,0], title="Vy at t=0")
#plot_surface(vort[0,:,:,0], title="vort at t=0")
#plot_surface(epsi[0,:,:,0], title="epsi at t=0")

#plot_surface(stream[-1,:,:,0], title="stream at t=tf")
#plot_surface(V_x[-1,:,:,0], title="Vx at t=tf")
#plot_surface(V_y[-1,:,:,0], title="Vy at t=tf")
#plot_surface(vort[-1,:,:,0], title="vort at t=tf")
#plot_surface(epsi[-1,:,:,0], title="epsi at t=tf")

## Spectra (Spatial) :
krOU, psdOU = plot_isotropic_psd(
   V_x[0,:,:,0],
   label="Bi-OU_t0",
   slope=-rho
)

# krOU, psdOU = plot_isotropic_psd(
#    V_x[900,:,:,0],
#    label="Bi-OU_t900",
#    slope=-rho
# )


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

tcurrent = 0
img_norm = np.sqrt(V_x[tcurrent, :, :, 0]**2 + V_y[tcurrent, :, :, 0]**2) 
img_norm_t_evolve = np.sqrt(V_x**2 + V_y**2) 
save_field_gif_matplotlib(img_norm_t_evolve, filename="normV.gif", fps=6)

# S_s, x_s, SR_s, SC_s, KR_s, Kc_s = plot_structure_functions(V_x[tcurrent, :, :, 0],
#                             small_scale=dx,
#                             large_scale=L_LS,
#                             nv=2**6,
#                             Nreal=101,
#                             # --- options affichage ---
#                             normalize=False,
#                             scale_factor_R=0.25,
#                             scale_factor_C=0.17,
#                             # --- options théorie Matérn ---
#                             plot_theory=True,
#                             dim=2,
#                             nu=c1,
#                             ell=ell,
#                             L=L,
#                             dx=dx,
#                             # --- pente de référence ---
#                             plot_slope=True,
#                             c1=c1,
#                             # --- kurtosis ---
#                             plot_kurtosis=True,
#                             c2=c2)

# plot_increment_pdfs(V_x[tcurrent, :, :, 0],
#                         small_scale=dx,
#                         large_scale=L_LS,
#                         nv=2**6,
#                         pdf_scales_indices=(0, 5, 10, 15),
#                         xlim_pdf=(-5, 5),
#                         nbins=120,
#                         vertical_spacing=5,
#                         min_count=10,
#                         normalize=True)

s_t, S2_t, K_T = plot_structure_functions_3D(
    signal=V_x,
    small_scale=dt,
    large_scale=T,
    nv=16,
    Nreal= 20,
    c1=c1,
    c2=0,
    pdf_scales_indices=(0, 1, 2, 4, 8, 10, 15, 30)
)

# scales, S2, K = plot_structure_functions_3D(
#     signal=img_norm_t_evolve,
#     small_scale=dt,
#     large_scale=T,
#     nv=16,
#     Nreal=100,
#     L_norm=200,
#     c1=c1,
#     c2=c2,
#     Nsamples_per_real=5000,
#     pdf_scales_indices=(0, 5, 10, 15, 30, 45, 100)
# )
