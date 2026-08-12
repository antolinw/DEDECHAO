import numpy as np
import torch 
from pytorch_wavelets import DWTForward, DWTInverse
from finite_diff import ortho_grad_2d,curl_2d,curl_3d,lap_2d

def run_simulation_W(cfg): 
    # ---- init shapes ----
    V_x = cfg.zeroField()
    V_y = cfg.zeroField()
    V_z = cfg.zeroField()
    stream_z = cfg.zeroField()
    vort_z = cfg.zeroField()
    epsi = cfg.zeroField()

    # -------------------------
    # Initialization
    # -------------------------
    lepsi_A, lepsi_D = wavelet_colored_noise(cfg,cfg.E_epsi_kj)
    epsi_t = epsilon(lepsi_A, lepsi_D,cfg)
    if cfg.new_spTimeGMC:
        B_A, B_D = wavelet_CMG_noise(epsi_t,cfg)
    else:
        B_A, B_D = wavelet_noise(cfg)
    V_x[0,:], V_y[0,:], V_z[0,:], stream_z[0:], vort_z[0:], epsi[0,:]\
            = CMG_velocity_W(B_A, B_D,epsi_t,cfg, intermittency = (not cfg.new_spTimeGMC) )

    # -------------------------
    # Time loop
    # -------------------------
    for t in range(cfg.N_steps - 1):

        # ---- OU process for B ----
        if cfg.new_spTimeGMC:
            _, noise_B_D = wavelet_CMG_noise(epsi_t,cfg)
        else:
            _, noise_B_D = wavelet_noise(cfg)
        B_D = wavelet_ou_update(B_D, cfg.exp_B, cfg.noise_B, noise_B_D)

        # ---- OU process for log-epsilon ----
        _, noise_e_D = wavelet_colored_noise(cfg,cfg.E_epsi_kj)
        lepsi_D = wavelet_ou_update(lepsi_D, cfg.exp_e, cfg.noise_e, noise_e_D)
        epsi_t = epsilon(lepsi_A, lepsi_D,cfg)

        # ---- Reconstruction & velocity ----
        V_x[t+1,:], V_y[t+1,:], V_z[t+1,:], stream_z[t+1,:] , vort_z[t+1,:], epsi[t+1,:] \
                    = CMG_velocity_W(
            B_A, B_D, epsi_t,cfg, intermittency = (not cfg.new_spTimeGMC) )


    return V_x, V_y, V_z, stream_z, vort_z, epsi

def wavelet_noise(cfg):  
    # White noise in physical space (ensembliste)
    noise = torch.randn(cfg.n_ens, 1, cfg.N, cfg.N) # format (cfg.n_ens, 1, cfg.N, cfg.N) for DWT
    A, D = cfg.dwt(noise)
    # Zero mean (eg B_hat[0, 0] = 0)
    A.zero_()

    return A, D

def wavelet_CMG_noise(epsi,cfg):
    return wavelet_CMG_from_noise(cfg.whiteNoise(),epsi,cfg) # default format (cfg.N, cfg.N, cfg.n_ens)

def wavelet_CMG_from_noise(B_real,epsi,cfg, intermittency = True):
    Bepsi = cfg.CMG_from_noise(B_real,epsi, intermittency) # default format (cfg.N, cfg.N, cfg.n_ens)

    Bepsi = np.moveaxis(Bepsi[np.newaxis,:], -1, 0) # format (cfg.n_ens, 1, cfg.N, cfg.N) for DWT
    Bepsi = torch.tensor(Bepsi,dtype=torch.float32) # torch for DWT
    A, D = cfg.dwt(Bepsi)
    # Zero mean (eg B_hat[0, 0] = 0)
    A.zero_()

    return A, D

# def colored_noise(cfg, spectrum):
#     A, D = wavelet_colored_noise(cfg, spectrum)
#     return reconstruct_field(A, D, cfg)

def coloring_noise(cfg, spectrum, noise):
    noise = np.moveaxis(noise[np.newaxis,:], -1, 0) # format (cfg.n_ens, 1, cfg.N, cfg.N) for DWT
    noise = torch.tensor(noise,dtype=torch.float32) # torch for DWT
    A,D = cfg.dwt(noise)
    A.zero_()
    # ---- Streamfunction ----
    A, D = wavelet_colored_noise_from_wavelet_noise(cfg, spectrum, A, D)
    return reconstruct_field(A, D, cfg)

def spectrum_model_scale(j, alpha, beta):
    return 2.0 ** (alpha * j + beta)

def wavelet_colored_noise(cfg, spectrum):  
    """
    Generate colored Gaussian noise directly in wavelet space
    with a prescribed scale-dependent spectrum.
    """

    A, D = wavelet_noise(cfg)

    return  wavelet_colored_noise_from_wavelet_noise(cfg, spectrum, A, D)

def wavelet_colored_noise_from_wavelet_noise(cfg, spectrum, A, D):  
    """
    Generate colored Gaussian noise directly in wavelet space
    with a prescribed scale-dependent spectrum
    and presribed noise
    """
    for j in range(cfg.J):
        # Fine -> coarse indexing
        Dj = D[cfg.J - 1 - j]

        # Number of wavelet coefficients at scale j
        npx = 2 ** j
        Nj = 3 * (npx ** 2)

        # Characteristic wavenumber
        kappa_j = cfg.k_j[j]

        # Target energy at scale j
        Ej = spectrum[j]

        # Energy per coefficient (Talla's magic coefficient)
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
def reconstruct_field(B_A, B_D, cfg):
    B = cfg.idwt((B_A, B_D))[:, 0]        # (ens, N, N)
    B = B.cpu().numpy()
    B = np.moveaxis(B, 0, -1)
    return B

def epsilon(lepsi_A, lepsi_D,cfg):
    lepsi = reconstruct_field(lepsi_A, lepsi_D, cfg)
    return cfg.intermittency(lepsi) 

def CMG_velocity_W(B_A, B_D, epsi,cfg, intermittency = True):
    B = reconstruct_field(B_A, B_D, cfg)
    Bepsi = cfg.CMG_from_noise(B,epsi, intermittency)
    return velocity_from_noise(Bepsi,epsi,cfg)

def velocity_from_noise(Bepsi,epsi,cfg):
    stream = coloring_noise(cfg, cfg.E_stream_kj, Bepsi)
    return velocity_from_stream(stream,epsi,cfg)

def velocity_from_stream(stream,epsi,cfg):
    # ---- Velocity ----
    if cfg.dim == 2:
        V = ortho_grad_2d(stream,cfg.dx,cfg.dx)
        vort = curl_2d(V,cfg.dx,cfg.dx)
    elif cfg.dim == 3:
        V = curl_3d(stream,cfg.dx,cfg.dx,cfg.dx)
        vort = curl_3d(V,cfg.dx,cfg.dx,cfg.dx)

    if cfg.dim == 2:
        V_x = V[:,:,0,:]
        V_y = V[:,:,1,:]
        V_z = cfg.zeroSpatialField()
        stream_z = stream
        vort_z = vort
    elif cfg.dim == 3:
        epsi = epsi[:,:,:,0,:]
        stream_z = stream[:,:,:,2,:]
        vort_z = vort[:,:,:,2,:]
        V_x = V[:,:,:,0,:]
        V_y = V[:,:,:,1,:]
        V_z = V[:,:,:,2,:]

    # Set amplitude
    ampli = cfg.K_LS * np.sqrt((stream_z[:]**2).mean()) # warning : create issues

    stream_z /= ampli
    V_x /= ampli
    V_y /= ampli
    V_z /= ampli
    vort_z /= ampli

    return V_x, V_y, V_z, stream_z, vort_z, epsi

