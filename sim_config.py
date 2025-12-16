# config.py
import numpy as np
from dataclasses import dataclass

RegulNorm = lambda x, epsilon: np.sqrt(x**2 + epsilon**2)

@dataclass
class SimulationConfig:
    # --------- Numerical ---------
    n_order: int = 9
    N_steps: int = 10
    Delta_t: float = 0.1

    # --------- Physical ---------
    L_factor: float = 0.5
    dim: float = 2.0
    c2: float = 0.1
    rho_s: float = 6.0
    gamma_t: float = 2.0

    tau_0B: float = 0.1
    tau_0e_factor: float = 1e10
    z_eps: float = 1.0


def build_derived_quantities(cfg: SimulationConfig):
    N = 2 ** cfg.n_order
    L = cfg.L_factor * N

    z = - (cfg.rho_s - 1) / (cfg.gamma_t + 1)
    theta_1 = cfg.rho_s + 2 * cfg.c2 + cfg.dim + 1
    K_LS = 2 * np.pi / L
    epsilon = L / N

    # Wavenumbers
    kx = np.fft.fftfreq(N, d=L / N) * 2 * np.pi
    ky = np.fft.fftfreq(N, d=L / N) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    K = np.sqrt(KX**2 + KY**2)

    # Spectrum
    S = 1 / (1 + (K / K_LS)) ** (theta_1 / 2)
    A = np.sqrt(S)

    # OU timescales
    tau_B = cfg.tau_0B * (1 + K / K_LS) ** (-z)
    tau_e = (cfg.tau_0e_factor * cfg.tau_0B) * (1 + K / K_LS) ** (-cfg.z_eps)

    exp_B = np.exp(-cfg.Delta_t / tau_B)
    noise_B = np.sqrt(1 - exp_B**2)

    exp_e = np.exp(-cfg.Delta_t / tau_e)
    noise_e = np.sqrt(1 - exp_e**2)

    mycorr = (cfg.c2**2) * np.log(L / RegulNorm(np.abs(K), epsilon))

    return {
        "N": N,
        "L": L,
        "KX": KX,
        "KY": KY,
        "K": K,
        "A": A,
        "exp_B": exp_B,
        "noise_B": noise_B,
        "exp_e": exp_e,
        "noise_e": noise_e,
        "mycorr": mycorr,
    }
