import numpy as np
import torch 
from pytorch_wavelets import DWTForward, DWTInverse

def run_simulation_W(cfg, derived): 

    N = derived["N"]
    J = derived["J"]
    wave = derived["wave"]
    theta_1 = derived["theta_1"]

    exp_B   = derived["exp_B"]    # shape (J,)
    noise_B = derived["noise_B"]  # shape (J,)

    exp_e   = derived["exp_e"]
    noise_e = derived["noise_e"]
    
    # ---- init shapes ----
    V_x = np.zeros((cfg.N_steps, N, N, cfg.n_ens))
    V_y = np.zeros_like(V_x)
    stream = np.zeros_like(V_x)
    vort = np.zeros_like(V_x)
    epsi = np.zeros_like(V_x)

    # -------------------------
    # Initialization
    # -------------------------
    B_A, B_D = wavelet_noise(cfg, derived)  
    lepsi_A, lepsi_D = wavelet_colored_noise(cfg, derived)

    Vx, Vy, s, w, e = CMG_velocity_W(B_A, B_D, lepsi_A, lepsi_D, cfg, derived)

    V_x[0] = np.moveaxis(Vx, 0, -1)
    V_y[0] = np.moveaxis(Vy, 0, -1)
    stream[0] = np.moveaxis(s, 0, -1)
    vort[0] = np.moveaxis(w, 0, -1)
    epsi[0] = np.moveaxis(e, 0, -1)

    # -------------------------
    # Time loop
    # -------------------------
    for t in range(cfg.N_steps - 1):

        # ---- OU process for B ----
        _, noise_B_D = wavelet_noise(cfg, derived)
        B_D = wavelet_ou_update(B_D, exp_B, noise_B, noise_B_D)

        # ---- OU process for log-epsilon ----
        _, noise_e_D = wavelet_noise(cfg, derived)
        lepsi_D = wavelet_ou_update(lepsi_D, exp_e, noise_e, noise_e_D)

        # ---- Reconstruction & velocity ----
        Vx, Vy, s, w, e = CMG_velocity_W(
            B_A, B_D, lepsi_A, lepsi_D, cfg, derived
        )

        V_x[t+1] = np.moveaxis(Vx, 0, -1)
        V_y[t+1] = np.moveaxis(Vy, 0, -1)
        stream[t+1] = np.moveaxis(s, 0, -1)
        vort[t+1] = np.moveaxis(w, 0, -1)
        epsi[t+1] = np.moveaxis(e, 0, -1)

    return V_x, V_y, stream, vort, epsi

def wavelet_noise(cfg, derived):
    N = derived["N"]
    J = derived["J"]
    wave = derived["wave"]
    
    # Bruit ensembliste
    noise = torch.randn(cfg.n_ens, 1, N, N)

    dwt = DWTForward(J=J, wave=wave, mode='periodization')
    A, D = dwt(noise)

    # Zero mean (eg B_hat[0, 0] = 0)
    A.zero_()

    return A, D

def spectrum_model_scale(j, alpha, beta):
    return 2.0 ** (alpha * j + beta)

def wavelet_colored_noise(cfg, derived):  
    """
    Generate colored Gaussian noise directly in wavelet space
    with a prescribed scale-dependent spectrum.
    """

    N = derived["N"]
    J = derived["J"]
    wave = derived["wave"]
    dx = derived["Lx"] / (2 ** derived["n_order"])
    dy = dx

    # White noise in physical space (ensembliste)
    noise = torch.randn(cfg.n_ens, 1, N, N)

    dwt = DWTForward(J=J, wave=wave, mode='periodization')
    A, D = dwt(noise)

    # Remove mean by zeroing approximate coefficients (k=0 analogue)
    A.zero_()

    for j in range(J):
        # Fine -> coarse indexing
        Dj = D[J - 1 - j]

        # Number of wavelet coefficients at scale j
        npx = 2 ** j
        Nj = 3 * (npx ** 2)

        # Characteristic wavenumber
        kx = (2.0 * np.pi) / ((2 ** (J - j)) * dx)
        ky = (2.0 * np.pi) / ((2 ** (J - j)) * dy)
        kappa_j = min(kx, ky)

        # Target energy at scale j
        Ej = spectrum_model_scale(J - j, -cfg.tau_0B, cfg.beta)

        # Energy per coefficient
        coef_s = kappa_j * (3.0 / 4.0) / Nj
        scale_factor = np.sqrt(coef_s * Ej)

        # Apply scaling (broadcasted over ensemble + subbands)
        D[J - 1 - j] = scale_factor * Dj

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
def reconstruct_fields(B_A, B_D, lepsi_A, lepsi_D, derived):
    idwt = DWTInverse(wave=derived["wave"], mode=derived["mode"])

    B = idwt((B_A, B_D))[:, 0]        # (ens, N, N)
    lepsi = idwt((lepsi_A, lepsi_D))[:, 0]

    return B, lepsi

def intermittency(lepsi):
    lepsi = lepsi - lepsi.mean(dim=(-2, -1), keepdim=True)
    return torch.exp(lepsi)

def spectral_velocity(B, epsi, derived, cfg):
    B = B.cpu().numpy()
    epsi = epsi.cpu().numpy()

    KX = derived["KX"]
    KY = derived["KY"]
    K  = derived["K"]
    A  = derived["A"]
    K_LS = 2 * np.pi / cfg.L_LS

    Bepsi_hat = np.fft.fft2(B * epsi, axes=(-2, -1))

    stream_hat = A * Bepsi_hat
    v_hat_x = 1j * KY * stream_hat
    v_hat_y = -1j * KX * stream_hat
    vort_hat = -K**2 * stream_hat

    stream = np.fft.ifft2(stream_hat, axes=(-2, -1)).real
    Vx = np.fft.ifft2(v_hat_x, axes=(-2, -1)).real
    Vy = np.fft.ifft2(v_hat_y, axes=(-2, -1)).real
    vort = np.fft.ifft2(vort_hat, axes=(-2, -1)).real

    # normalize per ensemble
    ampli = K_LS * np.sqrt((stream**2).mean(axis=(-2, -1), keepdims=True))

    stream /= ampli
    Vx /= ampli
    Vy /= ampli
    vort /= ampli

    return Vx, Vy, stream, vort, epsi

def CMG_velocity_W(B_A, B_D, lepsi_A, lepsi_D, cfg, derived):
    B, lepsi = reconstruct_fields(B_A, B_D, lepsi_A, lepsi_D, derived)
    epsi = intermittency(lepsi)
    return spectral_velocity(B, epsi, derived, cfg)
