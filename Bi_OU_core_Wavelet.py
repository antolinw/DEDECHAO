import numpy as np
import torch 
from pytorch_wavelets import DWTForward, DWTInverse
from Bi_OU_core import velocity_from_Streamfunction

def run_simulation_W(cfg): 
    # ---- init shapes ----
    V_x = cfg.zeroField()
    V_y = cfg.zeroField()
    V_z = cfg.zeroField()
    stream = cfg.zeroField()
    vort = cfg.zeroField()
    epsi = cfg.zeroField()

    # -------------------------
    # Initialization
    # -------------------------
    B_A, B_D = wavelet_noise(cfg)  
    lepsi_A, lepsi_D = wavelet_colored_noise(cfg)

    V_x[0,:], V_y[0,:], V_z[0,:], stream[0:], vort[0:], epsi[0,:]\
            = CMG_velocity_W(B_A, B_D, lepsi_A, lepsi_D, cfg)


    # -------------------------
    # Time loop
    # -------------------------
    for t in range(cfg.N_steps - 1):

        # ---- OU process for B ----
        _, noise_B_D = wavelet_noise(cfg)
        B_D = wavelet_ou_update(B_D, cfg.exp_B, cfg.noise_B, noise_B_D)

        # ---- OU process for log-epsilon ----
        _, noise_e_D = wavelet_noise(cfg)
        lepsi_D = wavelet_ou_update(lepsi_D, cfg.exp_e, cfg.noise_e, noise_e_D)

        # ---- Reconstruction & velocity ----
        V_x[t+1,:], V_y[t+1,:], V_z[t+1,:], stream[t+1,:] , vort[t+1,:], epsi[t+1,:] \
                    = CMG_velocity_W(
            B_A, B_D, lepsi_A, lepsi_D, cfg
        )


    return V_x, V_y, V_z, stream, vort, epsi

def wavelet_noise(cfg):  
    # White noise in physical space (ensembliste)
    noise = torch.randn(cfg.n_ens, 1, cfg.N, cfg.N)
    A, D = cfg.dwt(noise)
    # Zero mean (eg B_hat[0, 0] = 0)
    A.zero_()

    return A, D

def spectrum_model_scale(j, alpha, beta):
    return 2.0 ** (alpha * j + beta)

def wavelet_colored_noise(cfg):  
    """
    Generate colored Gaussian noise directly in wavelet space
    with a prescribed scale-dependent spectrum.
    """

    A, D = wavelet_noise(cfg)

    for j in range(cfg.J):
        # Fine -> coarse indexing
        Dj = D[cfg.J - 1 - j]

        # Number of wavelet coefficients at scale j
        npx = 2 ** j
        Nj = 3 * (npx ** 2)

        # Characteristic wavenumber
        kx = (2.0 * np.pi) / ((2 ** (cfg.J - j)) * cfg.dx)
        ky = (2.0 * np.pi) / ((2 ** (cfg.J - j)) * cfg.dx)
        kappa_j = min(kx, ky)

        # Target energy at scale j
        Ej = spectrum_model_scale(cfg.J - j, -cfg.tau_0B, cfg.beta)

        # Energy per coefficient
        coef_s = kappa_j * (3.0 / 4.0) / Nj
        scale_factor = np.sqrt(coef_s * Ej)

        # Apply scaling (broadcasted over ensemble + subbands)
        D[cfg.J - 1 - j] = scale_factor * Dj

    return A, D

def wavelet_ou_update(D, exp_j, noise_j, noise_D): 
    """
    Generic OU update for wavelet detail coefficients.

    Parameters
    ----------
    D : list of torch.Tensor
        Wavelet detail coefficients [D_1, ..., D_J]
    exp_j : array-like
        exp(-dt / tau_j) for each scale j
    noise_j : array-like
        sqrt(1 - exp_j^2) for each scale j
    noise_D : list of torch.Tensor
        White wavelet noise coefficients (same structure as D)

    Returns
    -------
    D_new : list of torch.Tensor
    """

    J = len(D)
    D_new = []

    for j in range(J):
        D_new.append(
            exp_j[j] * D[j] + noise_j[j] * noise_D[j]
        )

    return D_new

###CMG velocity decomposed in 4 parts:
def reconstruct_fields(B_A, B_D, lepsi_A, lepsi_D, cfg):
    B = cfg.idwt((B_A, B_D))[:, 0]        # (ens, N, N)
    lepsi = cfg.idwt((lepsi_A, lepsi_D))[:, 0]

    return B, lepsi

def intermittency(lepsi,cfg):
    lepsi /= lepsi.std() 
    lepsi *= np.sqrt(cfg.c2_eff)
    lepsi -= lepsi.var() 
    return torch.exp(lepsi)

def spectral_velocity(B, epsi, cfg):
    B = B.cpu().numpy()
    epsi = epsi.cpu().numpy()


    B = np.moveaxis(B, 0, -1)
    epsi = np.moveaxis(epsi, 0, -1)

    # WARNING : OLD CMG method !!!
    Bepsi = B * epsi

    Bepsi_hat = np.fft.fft2(Bepsi, axes=(0,1))
    cfg.cleanAliasing(Bepsi_hat)
    
    return velocity_from_Streamfunction(Bepsi_hat,epsi,cfg)


def CMG_velocity_W(B_A, B_D, lepsi_A, lepsi_D, cfg):
    B, lepsi = reconstruct_fields(B_A, B_D, lepsi_A, lepsi_D, cfg)
    epsi = intermittency(lepsi,cfg)
    return spectral_velocity(B, epsi, cfg)
