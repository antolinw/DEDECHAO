# config.py
import numpy as np
from dataclasses import dataclass

RegulNorm = lambda x, epsilon: np.sqrt(x**2 + epsilon**2)

@dataclass
class SimulationConfig:
    # --------- Numerical ---------
    n_order: int = 9            # N = 2^n_order = number of space steps along x
    N_steps: int = 10           # number of time steps
    Delta_t: float = 0.1        # time step
    n_ens: int = 3              # ensemble size

    # --------- Physical ---------
    L_factor: float = 0.5       # L = L_factor * N = large spatial scale ?
    Lx = 2                      # domain size
    dim: float = 2.0
    c2: float = 0.1
    rho_s: float = 6.0
    gamma_t: float = 2.0

    tau_0B: float = 0.1
    tau_0e_factor: float = 1e10
    z_eps: float = 1.0


def build_derived_quantities(cfg: SimulationConfig):
    N = 2 ** cfg.n_order
    PN = N//2                                      # aliased wavevector index
    L = cfg.L_factor * N

    z = - (cfg.rho_s - 1) / (cfg.gamma_t + 1)      # scaling slope for time scale
    theta_1 = cfg.rho_s + 2 * cfg.c2 + cfg.dim + 1 # spectrum slope of the bidirectionnal spatial spectrum
    K_LS = 2 * np.pi / L                           # large-scale wavenumber
    epsilon = L / N                                # spatial regularisation ?

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
    tau_B = cfg.tau_0B * (1 + K / K_LS) ** (-z)
    tau_e = (cfg.tau_0e_factor * cfg.tau_0B) * (1 + K / K_LS) ** (-cfg.z_eps)

    exp_B = np.exp(-cfg.Delta_t / tau_B)
    noise_B = np.sqrt(1 - exp_B**2)

    exp_e = np.exp(-cfg.Delta_t / tau_e)
    noise_e = np.sqrt(1 - exp_e**2)

    mycorr = (cfg.c2**2) * np.log(L / RegulNorm(normX_per, epsilon))
    S2 = np.fft.fft2(mycorr)

    return {
        "N": N,
        "PN": PN,
        "n_ens": cfg.n_ens,
        "L": L,
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
