import numpy as np

def run_simulation(cfg, derived, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    N = derived["N"]
    KX = derived["KX"]
    KY = derived["KY"]
    A = derived["A"]

    exp_B = derived["exp_B"]
    noise_B = derived["noise_B"]
    exp_e = derived["exp_e"]
    noise_e = derived["noise_e"]
    mycorr = derived["mycorr"]

    B_hat = np.zeros((N, N), dtype=complex)
    lepsi_hat = np.zeros((N, N), dtype=complex)

    V_x = np.zeros((cfg.N_steps, N, N))
    V_y = np.zeros((cfg.N_steps, N, N))

    for t in range(cfg.N_steps):
        # ---- OU process for B ----
        xi = rng.normal(size=(N, N))
        eta = rng.normal(size=(N, N))
        B_hat = B_hat * exp_B + noise_B * (xi + 1j * eta) / np.sqrt(2)

        # ---- OU process for epsilon ----
        xi_e = rng.normal(size=(N, N)) * np.sqrt(mycorr)
        eta_e = rng.normal(size=(N, N)) * np.sqrt(mycorr)
        lepsi_hat = lepsi_hat * exp_e + noise_e * (xi_e + 1j * eta_e) / np.sqrt(2)
        lepsi_hat[0, 0] = 0

        # ---- Real space ----
        B_real = np.fft.ifft2(B_hat).real * N**2
        B_real = (B_real - B_real.mean()) / B_real.std()

        lepsi_real = np.fft.ifft2(lepsi_hat).real
        lepsi_real = lepsi_real / lepsi_real.std()
        lepsi_real -= lepsi_real.var() / 2
        epsi = np.exp(lepsi_real)

        Bepsi_hat = np.fft.fft2(B_real * epsi)
        Bepsi_hat[0, 0] = 0

        # ---- Velocity ----
        v_hat_x = 1j * KY * A * Bepsi_hat
        v_hat_y = -1j * KX * A * Bepsi_hat

        V_x[t] = np.fft.ifft2(v_hat_x).real
        V_y[t] = np.fft.ifft2(v_hat_y).real

    return V_x, V_y