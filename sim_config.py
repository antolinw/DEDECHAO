# config.py
import numpy as np
from dataclasses import dataclass

@dataclass
class SimulationConfig:
    # --------- Numerical ---------
    n_order: int = 9            # N = 2^n_order = number of space steps along x
    N_steps: int = 100           # number of time steps
    Delta_t: float = 1e-3        # time step
    n_ens: int = 1              # ensemble size

    # --------- Physical ---------
    Lx: float = 2                 # domain size
    L_LS: float = 0.5 * Lx           # large spatial scale

    dim: float = 2.0            # spatial dimension
    c2: float = 0.1             # intermittency coefficient
    rho_s: float = 2.0          # spatial spectrum slope
    gamma_t: float = rho_s        # temporal spectrum slope
    c1 = (rho_s + dim - 2) / 2    # fractal exponent

    tau_0B: float = 1.0
    tau_0e_factor: float = np.inf # steady intermittency
    # tau_0e_factor: float = 1.0  # unsteady intermittency
    z_eps: float = 1.0

class Parameters(SimulationConfig): 
    def __init__(self):
        super().__init__()

        N = 2 ** self.n_order
        self.PN = N//2                                      # aliased wavevector index

        z = (self.rho_s - 1) / (self.gamma_t - 1)      # scaling slope for time scale
        theta_1 = self.rho_s + self.dim + 1            # spectrum slope of the bidirectionnal spatial spectrum
        K_Lx = 2 * np.pi / self.Lx                      # largest-scale wavenumber
        K_LS = 2 * np.pi / self.L_LS                           # large-scale wavenumber

        np.random.seed(0)

        # Wavenumbers
        kx = np.fft.fftfreq(N, d=self.Lx / N) * 2 * np.pi
        ky = np.fft.fftfreq(N, d=self.Lx / N) * 2 * np.pi
        KX, KY = np.meshgrid(kx, ky, indexing="ij")
        K = np.sqrt(KX**2 + KY**2)

        # Spectrum
        S1 = 1 / (1 + (K / K_LS)**2 ) ** (theta_1 / 2)
        self.cleanAliasing(S1)
        A = np.sqrt(S1)

        # OU timescales
        tau_B = self.tau_0B * (1 + (K / K_LS)**2 ) ** ( - z/2)
        tau_e = (self.tau_0e_factor * self.tau_0B) * (1 + (K / K_LS)**2 ) ** ( - self.z_eps/2)
        min_t = min([tau_B.min(),tau_e.min()])
        print("The smallest physical time is = " + str(min_t))
        print("The time step is : dt = " + str(self.Delta_t))
        if min_t <= self.Delta_t :
            print("ACHTUNG : the time step is too small to prevent aliasing !!!")

        exp_B = np.exp(-self.Delta_t / tau_B)
        noise_B = np.sqrt(1 - exp_B**2)

        exp_e = np.exp(-self.Delta_t / tau_e)
        noise_e = np.sqrt(1 - exp_e**2)

        np.seterr(divide = 'ignore') 
        S2 = self.c2 * ( K / K_Lx ) ** (-1)
        self.cleanAliasing(S2)
        np.seterr(divide = 'warn') 

        self.N = N
        self.KX = KX
        self.KY = KY
        self.K = K
        self.A = A
        self.exp_B = exp_B
        self.noise_B = noise_B
        self.exp_e = exp_e
        self.noise_e = noise_e
        self.S2 = S2
    
    def cleanAliasing(self,S1):
        S1[0, 0] = 0
        S1[self.PN, :] = 0
        S1[:, self.PN] = 0
