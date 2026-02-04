# config.py
import numpy as np
from dataclasses import dataclass
from pytorch_wavelets import DWTForward, DWTInverse

@dataclass
class SimulationConfig:
    # --------- Numerical ---------
    n_order: int = 5              # N = 2^n_order = number of space steps along x
    N_steps: int = 1000            # number of time steps
    Delta_t: float = 1e-3         # time step
    n_ens: int = 1                # ensemble size
    method: str = 'Wavelet'       # Choice of method :'Fourier' or 'Wavelet'
    wave: str = 'db4'             # choice of the wavelet
    J: int = 2                    # Number of level of decomposition for wavelet method
    beta: float = 1.              # Energy injected in wavelet spectrum
    mode: str = 'periodization'

    # --------- Physical ---------
    Lx: float = 2                 # domain size
    L_LS: float = 0.5 * Lx        # large spatial scale

    dim: float = 2.0              # spatial dimension
    c2: float = 0.1               # intermittency coefficient
    rho_s: float = 2.0            # spatial spectrum slope
    gamma_t: float = rho_s        # temporal spectrum slope
    c1 = (rho_s + dim - 2) / 2    # fractal exponent

    tau_0B: float = 1.0
    tau_0e_factor: float = np.inf # steady intermittency
    # tau_0e_factor: float = 1.0  # unsteady intermittency
    z_eps: float = 1.0


def build_derived_quantities(cfg: SimulationConfig):
    N = 2 ** cfg.n_order
    PN = N//2                                      # aliased wavevector index

    z = (cfg.rho_s - 1) / (cfg.gamma_t - 1)        # scaling slope for time scale
    theta_1 = cfg.rho_s + cfg.dim + 1              # spectrum slope of the bidirectionnal spatial spectrum
    K_Lx = 2 * np.pi / cfg.Lx                      # largest-scale wavenumber
    K_LS = 2 * np.pi / cfg.L_LS                    # large-scale wavenumber

    np.random.seed(0)

    # Spatial grid for doubly-periodic image
    x_per = np.fft.fftfreq(N, d=1/cfg.Lx) 
    y_per = np.fft.fftfreq(N, d=1/cfg.Lx) 
    X_per, Y_per = np.meshgrid(x_per, y_per, indexing="ij")
    normX_per = np.sqrt(X_per**2 + Y_per**2)

    # Wavenumbers
    kx = np.fft.fftfreq(N, d=cfg.Lx / N) * 2 * np.pi
    ky = np.fft.fftfreq(N, d=cfg.Lx / N) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    K = np.sqrt(KX**2 + KY**2)

    # Spectrum
    S1 = 1 / (1 + (K / K_LS)**2 ) ** (theta_1 / 2)
    S1[0, 0] = 0
    S1[PN, :] = 0
    S1[:, PN] = 0
    A = np.sqrt(S1)

    # OU timescales
    tau_B = cfg.tau_0B * (1 + (K / K_LS)**2 ) ** ( - z/2)
    tau_e = (cfg.tau_0e_factor * cfg.tau_0B) * (1 + (K / K_LS)**2 ) ** ( - cfg.z_eps/2)
    min_t = min([tau_B.min(),tau_e.min()])
    print("The smallest physical time is = " + str(min_t))
    print("The time step is : dt = " + str(cfg.Delta_t))
    if min_t <= cfg.Delta_t :
        print("ACHTUNG : the time step is too small to prevent aliasing !!!")

    exp_B = np.exp(-cfg.Delta_t / tau_B)
    noise_B = np.sqrt(1 - exp_B**2)

    exp_e = np.exp(-cfg.Delta_t / tau_e)
    noise_e = np.sqrt(1 - exp_e**2)

    np.seterr(divide = 'ignore') 
    S2 = cfg.c2 * ( K / K_Lx ) ** (-1)
    S2[0, 0] = 0
    S2[PN, :] = 0
    S2[:, PN] = 0
    np.seterr(divide = 'warn') 

    return {
        "N": N,
        "PN": PN,
        "n_ens": cfg.n_ens,
        "KX": KX,
        "KY": KY,
        "K": K,
        "A": A,
        "exp_B": exp_B,
        "noise_B": noise_B,
        "exp_e": exp_e,
        "noise_e": noise_e,
        "S2": S2,
    }

def build_derived_W_quantities(cfg: SimulationConfig): 

    # ---------- Resolution ----------
    n_order = cfg.n_order
    N = 2 ** (n_order + 1)
    PN = N // 2  
    Lx = cfg.Lx
    J = cfg.J
    wave = cfg.wave
    mode = cfg.mode
    Jmax = np.log2(N)
    dwt = DWTForward(J=J, wave=wave, mode=mode)
    idwt = DWTInverse(wave=wave, mode=mode)

    if cfg.J > Jmax:
        raise ValueError("ACHTUNG : Requested J exceeds maximum wavelet scale !!!")

    j = np.arange(1, J + 1)

    # ---------- Physical parameters ----------
    z = (cfg.rho_s - 1) / (cfg.gamma_t - 1)   # scaling slope for time scale
    theta_1 = cfg.rho_s + cfg.dim + 1         # spectrum slope of the bidirectionnal spatial spectrum

    K_Lx = 2 * np.pi / cfg.Lx                 # largest-scale wavelet wavenumber
    K_LS = 2 * np.pi / cfg.L_LS               # large-scale wavelet wavenumber

    # Wavenumbers (Fourier) for spectral velocity computation
    kx = np.fft.fftfreq(N, d=cfg.Lx / N) * 2 * np.pi
    ky = np.fft.fftfreq(N, d=cfg.Lx / N) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    K = np.sqrt(KX**2 + KY**2)

    # Spectrum
    S1 = 1 / (1 + (K / K_LS)**2 ) ** (theta_1 / 2)
    S1[0, 0] = 0
    S1[PN, :] = 0
    S1[:, PN] = 0
    A = np.sqrt(S1)
    
    # Wavelet characteristic wavenumbers
    j = np.arange(cfg.J)
    k_j = K_Lx * (2.0 ** j)
    
    # ---------- OU time scales per scales ----------
    tau_B_j = cfg.tau_0B * (1 + (k_j / K_LS)**2) ** (-z / 2)
    tau_e_j = (cfg.tau_0e_factor * cfg.tau_0B * (1 + (k_j / K_LS)**2) ** (-cfg.z_eps / 2))

    # Stability check
    min_tau = min(tau_B_j.min(), tau_e_j.min())
    print(f"Smallest physical time: {min_tau}")
    print(f"Time step: dt = {cfg.Delta_t}")

    if min_tau <= cfg.Delta_t:
        print("ACHTUNG: time step may be too large!")
        
    # ---------- OU coefficients (1D arrays) ----------
    exp_B = np.exp(-cfg.Delta_t / tau_B_j)
    noise_B = np.sqrt(1 - exp_B**2)

    exp_e = np.exp(-cfg.Delta_t / tau_e_j)
    noise_e = np.sqrt(1 - exp_e**2)

    return {
        "N": N,
        "Lx": Lx,
        "n_order": n_order,
        "J": J,
        "wave": wave,
        "mode": mode,
        "dwt": dwt,
        "idwt": idwt,
        "j": j,
        "k_j": k_j,
        "KX": KX,
        "KY": KY,
        "K": K,
        "A": A,
        "exp_B": exp_B,
        "noise_B": noise_B,
        "exp_e": exp_e,
        "noise_e": noise_e,
        "theta_1": theta_1,
    }
