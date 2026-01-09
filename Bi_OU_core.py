import numpy as np

def run_simulation(cfg, derived):

    N = derived["N"]
    n_ens = derived["n_ens"]

    exp_B = derived["exp_B"][:,:,np.newaxis]
    noise_B = derived["noise_B"][:,:,np.newaxis]
    exp_e = derived["exp_e"][:,:,np.newaxis]
    noise_e = derived["noise_e"][:,:,np.newaxis]
    S2 = derived["S2"][:,:,np.newaxis]

    # ---- init ----
    V_x = np.zeros((cfg.N_steps, N, N, n_ens))
    V_y = np.zeros((cfg.N_steps, N, N, n_ens))
    stream = np.zeros((cfg.N_steps, N, N, n_ens))
    vort = np.zeros((cfg.N_steps, N, N, n_ens))
    epsi = np.zeros((cfg.N_steps, N, N, n_ens))

    B_hat = noise(cfg, derived)
    lepsi_hat =  np.sqrt(S2) * noise(cfg, derived)
    V_x[0,:,:,:], V_y[0,:,:,:], stream[0,:,:,:], vort[0,:,:,:], epsi[0,:,:,:]\
        = CMG_velocity(B_hat,lepsi_hat,cfg, derived)

    for t in range(cfg.N_steps-1):
        # ---- OU process for B ----
        B_hat = exp_B * B_hat + noise_B * noise(cfg, derived)
        # ---- OU process for epsilon ----
        lepsi_hat = exp_e  * lepsi_hat + noise_e * np.sqrt(S2) * noise(cfg, derived)

        V_x[t+1,:,:,:], V_y[t+1,:,:,:], stream[t+1,:,:,:] , vort[t+1,:,:,:], epsi[t+1,:,:,:] \
            = CMG_velocity(B_hat,lepsi_hat,cfg, derived)

    return V_x, V_y, stream, vort, epsi



def noise(cfg, derived):
    N = derived["N"]
    PN = derived["PN"]
    n_ens = derived["n_ens"]

    xi = np.random.randn(N, N, n_ens)
    B_hat = np.fft.fft2(xi, axes=(0,1)) 

    B_hat[0, 0] = 0
    B_hat[PN, :] = 0
    B_hat[:, PN] = 0

    return B_hat


def CMG_velocity(B_hat,lepsi_hat,cfg, derived):

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
    B_real = np.fft.ifft2(B_hat, axes=(0,1)).real
    # B_real = (B_real - B_real.mean()) / B_real.std() # needed ???

    lepsi_real = np.fft.ifft2(lepsi_hat, axes=(0,1)).real
    lepsi_real -= lepsi_real.var() 
    epsi = np.exp(lepsi_real)

    Bepsi_hat = np.fft.fft2(B_real * epsi, axes=(0,1))

    # ---- Velocity ----
    stream_hat = A * Bepsi_hat
    v_hat_x = 1j * KY * stream_hat
    v_hat_y = -1j * KX * stream_hat
    vort_hat = - K**2 * stream_hat

    stream = np.fft.ifft2(stream_hat, axes=(0,1)).real
    V_x = np.fft.ifft2(v_hat_x, axes=(0,1)).real
    V_y = np.fft.ifft2(v_hat_y, axes=(0,1)).real
    vort = np.fft.ifft2(vort_hat, axes=(0,1)).real

    return V_x, V_y, stream, vort, epsi

