import numpy as np

def run_simulation(cfg):

    N = cfg.N
    n_ens = cfg.n_ens

    exp_B = cfg.exp_B[:,:,np.newaxis]
    noise_B = cfg.noise_B[:,:,np.newaxis]
    exp_e = cfg.exp_e[:,:,np.newaxis]
    noise_e = cfg.noise_e[:,:,np.newaxis]
    S2 = cfg.S2[:,:,np.newaxis]

    # ---- init ----
    V_x = cfg.zeroField()
    V_y = cfg.zeroField()
    stream = cfg.zeroField()
    vort = cfg.zeroField()
    epsi = cfg.zeroField()

    B_hat = noise(cfg)
    lepsi_hat =  np.sqrt(S2) * noise(cfg)
    V_x[0,:,:,:], V_y[0,:,:,:], stream[0,:,:,:], vort[0,:,:,:], epsi[0,:,:,:]\
        = CMG_velocity(B_hat,lepsi_hat,cfg)

    for t in range(cfg.N_steps-1):
        # ---- OU process for B ----
        B_hat = exp_B * B_hat + noise_B * noise(cfg)
        # ---- OU process for epsilon ----
        lepsi_hat = exp_e  * lepsi_hat + noise_e * np.sqrt(S2) * noise(cfg)

        V_x[t+1,:,:,:], V_y[t+1,:,:,:], stream[t+1,:,:,:] , vort[t+1,:,:,:], epsi[t+1,:,:,:] \
            = CMG_velocity(B_hat,lepsi_hat,cfg)

    return V_x, V_y, stream, vort, epsi



def noise(cfg):
    N = cfg.N
    PN = cfg.PN
    n_ens = cfg.n_ens

    xi = np.random.randn(N, N, n_ens)
    B_hat = np.fft.fft2(xi, axes=(0,1)) 

    cfg.cleanAliasing(B_hat)

    return B_hat


def CMG_velocity(B_hat,lepsi_hat,cfg):

    N = cfg.N
    n_ens = cfg.n_ens
    KX = cfg.KX[:,:,np.newaxis]
    KY = cfg.KY[:,:,np.newaxis]
    K = cfg.K[:,:,np.newaxis]
    A = cfg.A[:,:,np.newaxis]
    K_LS = 2 * np.pi / cfg.L_LS

    exp_B = cfg.exp_B[:,:,np.newaxis]
    noise_B = cfg.noise_B[:,:,np.newaxis]
    exp_e = cfg.exp_e[:,:,np.newaxis]
    noise_e = cfg.noise_e[:,:,np.newaxis]
    S2 = cfg.S2[:,:,np.newaxis]

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

    # Set amplitude
    ampli = K_LS * np.sqrt((stream[:]**2).mean())
    stream /= ampli
    V_x /= ampli
    V_y /= ampli
    vort /= ampli

    return V_x, V_y, stream, vort, epsi

