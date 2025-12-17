import numpy as np

def run_simulation(cfg, derived, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    N = derived["N"]

    exp_B = derived["exp_B"]
    noise_B = derived["noise_B"]
    exp_e = derived["exp_e"]
    noise_e = derived["noise_e"]
    S2 = derived["S2"]

    B_hat = np.zeros((N, N), dtype=complex)
    lepsi_hat = np.zeros((N, N), dtype=complex)

    V_x = np.zeros((cfg.N_steps, N, N))
    V_y = np.zeros((cfg.N_steps, N, N))
    stream = np.zeros((cfg.N_steps, N, N))
    vort = np.zeros((cfg.N_steps, N, N))

    for t in range(cfg.N_steps):
        # ---- OU process for B ----
        B_hat = exp_B * B_hat + noise_B * noise(cfg, derived)
        # ---- OU process for epsilon ----
        lepsi_hat = exp_e  * lepsi_hat + noise_e * np.sqrt(S2) * noise(cfg, derived)

        V_x[t+1,:,:,:], V_y[t+1,:,:,:], stream[t+1,:,:,:] , vort[t+1,:,:,:] \
            = CMG_velocity(B_hat,lepsi_hat,cfg, derived)

    return V_x, V_y, stream, vort



def noise(cfg, derived):
    rng = np.random.default_rng()
    N = derived["N"]
    n_ens = derived["n_ens"]

    xi = rng.normal(size=(N, N, n_ens))
    eta = rng.normal(size=(N, N, n_ens))
    B_hat = (xi + 1j * eta) / np.sqrt(2)

    # B_hat = np.fft.fft2(rng.normal(size=(N, N, n_ens))) / N**2

    return B_hat


def CMG_velocity(B_hat,lepsi_hat,cfg, derived, rng=None):

    N = derived["N"]
    n_ens = derived["n_ens"]
    KX = derived["KX"][:,:,np.newaxis]
    KY = derived["KY"][:,:,np.newaxis]
    K = derived["K"][:,:,np.newaxis]
    A = derived["A"][:,:,np.newaxis]

    exp_B = derived["exp_B"][:,:,np.newaxis]
    noise_B = derived["noise_B"][:,:,np.newaxis]
    exp_e = derived["exp_e"][:,:,np.newaxis]
    noise_e = derived["noise_e"][:,:,np.newaxis]
    S2 = derived["S2"][:,:,np.newaxis]

    # ---- Real space ----
    B_real = np.fft.ifft2(B_hat).real
    B_real *= N**2
    # B_real = (B_real - B_real.mean()) / B_real.std() # needed ???

    lepsi_real = np.fft.ifft2(lepsi_hat).real
    # lepsi_real = lepsi_real / lepsi_real.std() # needed ???
    lepsi_real -= lepsi_real.var() / 2
    epsi = np.exp(lepsi_real)

    Bepsi_hat = np.fft.fft2(B_real * epsi)

    # ---- Velocity ----
    stream_hat = A * Bepsi_hat
    v_hat_x = 1j * KY * A * Bepsi_hat
    v_hat_y = -1j * KX * A * Bepsi_hat
    vort_hat = - K**2 * A * Bepsi_hat

    stream = np.fft.ifft2(stream_hat, axes=(0,1)).real
    V_x = np.fft.ifft2(v_hat_x, axes=(0,1)).real
    V_y = np.fft.ifft2(v_hat_y, axes=(0,1)).real
    vort = np.fft.ifft2(vort_hat, axes=(0,1)).real

    return V_x, V_y, stream, vort

