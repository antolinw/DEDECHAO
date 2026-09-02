import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew
from sim_config import Parameters
from plot import plot_surface, plot_isotropic_spatial_psd, \
                 save_field_gif_matplotlib, compute_temporal_spectrum, plot_spectrum, \
                 plot_spectrum_mean_std, zeta_n, plot_structure_functions, plot_increment_pdfs, \
                 plot_structure_functions_temporal, save_spiky_field_gif_matplotlib

       
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

match cfg.output_type:
    case 2:
        stream_z = cfg.run_simulation()
    case 1:
        V_x, V_y = cfg.run_simulation()
    case 0:
        V_x, V_y, V_z, stream_z, vort_z, epsi = cfg.run_simulation()


print("Simulation completed. Now plotting results...")

###################################### Plots ######################################
##### The outputs are 4D arrays with dimensions (time, x, y, ensemble member) #####
###################################################################################

if dim == 3: # take a slice a z = 0
   stream_z = stream_z[:,:,:,0,:]
   V_x = V_x[:,:,:,0,:]
   V_y = V_y[:,:,:,0,:]
   V_z = V_z[:,:,:,0,:]
   vort_z = vort_z[:,:,:,0,:]

### 2D + 3D instant plots :
#plot_surface(stream_z[0,:,:,0], title="stream_z at t=0")
#plot_surface(V_x[0,:,:,0], title="Vx at t=0")
plot_surface(V_y[0,:,:,0], title="Vy at t=0 and z=0")
plot_surface(V_z[0,:,:,0], title="Vz at t=0 and z=0")
plot_surface(vort_z[0,:,:,0], title="vort_z at t=0 and z=0")
#plot_surface(epsi[0,:,:,0], title="epsi at t=0")

#plot_surface(stream_z[-1,:,:,0], title="stream_z at t=tf")
#plot_surface(V_x[-1,:,:,0], title="Vx at t=tf")
#plot_surface(V_y[-1,:,:,0], title="Vy at t=tf")
#plot_surface(vort_z[-1,:,:,0], title="vort_z at t=tf")
#plot_surface(epsi[-1,:,:,0], title="epsi at t=tf")

## KE spatial spectrum :
krOU, psdOU = plot_isotropic_spatial_psd(
   V_x[0,:,:,0],
   label="Vx",
   slope=-rho
)

## intermittency spatial spectrum :
krOU, psdOU = plot_isotropic_spatial_psd(
   epsi[0,:,:,0],
   label="epsi_t0",
   slope=-1
)

# krOU, psdOU = plot_isotropic_spatial_psd(
#    V_x[900,:,:,0],
#    label="Bi-OU_t900",
#    slope=-rho
# )


# Animation of vorticity over time (I took the first member of the ensemble here) :
#save_field_gif_matplotlib(vort_z[:,:,:,0], filename="Vort_r-6_c2-1_zeps-1.gif", fps=15)
#save_field_gif_matplotlib(epsi[:,:,:,0], filename="Epsi_c2-0p05_S2corrected.gif", fps=15)
save_spiky_field_gif_matplotlib(
    epsi[:,:,:,0],
    filename="Epsi_c2-0p05_S2corrected.gif",
    lower_percentile=1.0,
    upper_percentile=99.5
)

###Temporal Spectrum take at a fixed point (here aprx the center of the domain)
signal = V_x[:, :, :, 0]

### Compute temporal spectrum
w, ke_hat = compute_temporal_spectrum(signal, cfg.Delta_t)

### Plot temporal spectrum averaged over space
# ke_mean = ke_hat.mean(axis=(1,2))
# ke_std  = ke_hat.std(axis=(1,2))
#plot_spectrum(w, ke_mean, slope=-gamma)

### Plot temporal spectrum with mean and std
plot_spectrum_mean_std(
    w,
    ke_hat,
    slope=-gamma,
    std_factor=1.0,
    filename= "temporal_spectrum_mean_std_Vx.png"
)

###Temporal Spectrum take at a fixed point (here aprx the center of the domain)
signal = epsi[:, :, :, 0]

### Compute temporal spectrum
w, ke_hat = compute_temporal_spectrum(signal, cfg.Delta_t)

### Plot temporal spectrum averaged over space
# ke_mean = ke_hat.mean(axis=(1,2))
# ke_std  = ke_hat.std(axis=(1,2))
#plot_spectrum(w, ke_mean, slope=-gamma)

### Plot temporal spectrum with mean and std
plot_spectrum_mean_std(
    w,
    ke_hat,
    slope=-1,
    std_factor=1.0,
    filename= "temporal_spectrum_mean_std_epsi.png"
)

### GIF of time evolution
img_norm_t_evolve = np.sqrt(V_x**2 + V_y**2 + V_z**2) 
save_field_gif_matplotlib(img_norm_t_evolve, filename="normV.gif", fps=6)

### Structure functions
tcurrent = 0
img_norm = np.sqrt(V_x[tcurrent, :, :, 0]**2 + V_y[tcurrent, :, :, 0]**2 + V_z[tcurrent, :, :, 0]**2) 
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

#Tested with N=2**8 et Nt=3000
x, S2, K = plot_structure_functions_temporal(
    V_x[:,:,:,0],
    small_scale=1,      # Minimum time lag
    large_scale=np.size(V_x[:,0,0,0])//4,  # Large scale in numbers of step, can be larger than L integral
    dt=dt,              # pas de temps en secondes
    tau_gauss=T,        # échelle intégrale ou de gaussianisation
    c2=cfg.c2,
    pdf_scales_indices=(0, 5, 10, 35, 60)
)
