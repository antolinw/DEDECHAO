
import numpy as np 
import torch 
import pywt
import matplotlib.animation as animation
import os
from scipy.fft import fftn as fft
from scipy.fft import ifftn as ifft
from scipy.fft import fft2 as fft2
from pytorch_wavelets import DWTForward, DWTInverse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

RegulNorm = lambda x, epsilon : np.sqrt(x**2+epsilon**2)

# ---------------- SETTINGS ---------------- #
n_order = 5
N = 2 ** (4 + n_order)
N_steps = 10
Delta_t = 1

#### MAIN INPUTS
L = 0.5 * N     # Echelle integrale spatiale
dim = 2.         # Dimension
c2 = 0.1         # Intermittence 
rho_s = 6    # Stats spatiales => pente spectrale
gamma_t = 2  # Stats temporelles => pente spectrale

#### CALCULATED 
z = - (rho_s - 1) / (gamma_t + 1)
theta_1 = rho_s + 2 * c2 + dim + 1
c1 = c2 + (rho_s + dim - 2 ) / 2 
K_LS = 2 * np.pi / ( L )
epsilon = L / N

tau_0B = 0.1
#tau_0e = 0.1*tau_0B
tau_0e = 10e9*tau_0B
z_eps=1

# ---------------- WAVENUMBERS ---------------- #
kx = np.fft.fftfreq(N, d=L / N) * 2 * np.pi
ky = np.fft.fftfreq(N, d=L / N) * 2 * np.pi
KX, KY = np.meshgrid(kx, ky, indexing='ij')
K = np.sqrt(KX ** 2 + KY ** 2)
# K[0,0] = K[0,1] / 2

# Power spectrum
S = 1 / (1 + (K /K_LS)) ** ( theta_1 / 2)
A = np.sqrt(S)

# OU time scales
with np.errstate(divide='ignore', invalid='ignore'):
    tau_B = tau_0B * (1 + K/K_LS) ** (-z)
    tau_e = tau_0e * (1 + K/K_LS) ** (-z_eps)
    # tau_B = np.where(K != 0, K ** (-0.8), np.inf)
    # tau_e = np.where(K != 0, K ** (-1), np.inf)

exp_B  = np.exp(-Delta_t / tau_B)
noise_B = np.sqrt(1 - exp_B ** 2)

exp_e  = np.exp(-Delta_t / tau_e)
noise_e = np.sqrt(1 - exp_e ** 2)

# amplitude for lepsi noise
mycorr = (c2 ** 2) * np.log(L / RegulNorm(abs(K), epsilon))


# ---------------- INIT ---------------- #
B_hat = np.zeros((N,N), dtype=complex)
lepsi_hat = np.zeros((N,N), dtype=complex)

# Allocate output arrays
V_x = np.zeros((N_steps, N, N), dtype=float)
V_y = np.zeros((N_steps, N, N), dtype=float)

rng = np.random.default_rng()


# ---------------- MAIN LOOP ---------------- #
for t in range(N_steps):
    # ----- Update B (hat) (Ornstein Uhlenbeck 1) -----
    xi  = rng.normal(size=(N,N))
    eta = rng.normal(size=(N,N))
    B_hat = B_hat * exp_B + noise_B * (xi + 1j * eta) / np.sqrt(2)

    # ----- Update l epsilon(hat) (Ornstein Uhlenbeck 2) -----
    xi_e  = rng.normal(size=(N,N)) * np.sqrt(mycorr)
    eta_e = rng.normal(size=(N,N)) * np.sqrt(mycorr)
    lepsi_hat = lepsi_hat * exp_e + noise_e * (xi_e + 1j * eta_e) / np.sqrt(2)
    lepsi_hat[0,0] = 0

    # ----- Go to real space -----
    B_real = np.fft.ifft2(B_hat).real * N**2
    B_real -= B_real.mean()
    B_real /= B_real.std()

    lepsi_real = np.fft.ifft2(lepsi_hat).real #* N**2
    lepsi_real /= lepsi_real.std()
    lepsi_real -= lepsi_real.var() / 2
    epsi = np.exp(lepsi_real)

    # ----- Compute B epsilon directly -----
    Bepsi = B_real * epsi
    Bepsi_hat = np.fft.fft2(Bepsi)
    Bepsi_hat[0,0] = 0

    # ----- Velocity field -----
    v_hat_x = 1j * KY * A * Bepsi_hat
    v_hat_y = -1j * KX * A * Bepsi_hat

    V_x[t] = np.fft.ifft2(v_hat_x).real 
    V_y[t] = np.fft.ifft2(v_hat_y).real 