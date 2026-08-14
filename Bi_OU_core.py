import numpy as np

def run_simulation(cfg):

    # ---- init ----
    V_x = cfg.zeroField()
    V_y = cfg.zeroField()
    V_z = cfg.zeroField()
    stream_z = cfg.zeroField()
    vort_z = cfg.zeroField()
    epsi = cfg.zeroField()

    lepsi_hat =  np.sqrt(cfg.S2) * fftnoise(cfg, scalar = True)
    epsi_t = epsilon(lepsi_hat,cfg)
    if cfg.new_spTimeGMC:
        B_hat = FFT_CMG_noise(epsi_t,cfg)
    else:
        B_hat = fftnoise(cfg)
    V_x[0,:], V_y[0,:], V_z[0,:], stream_z[0:], vort_z[0:], epsi[0,:]\
        = CMG_velocity(B_hat,epsi_t,cfg, intermittency = (not cfg.new_spTimeGMC) )

    for t in range(cfg.N_steps-1):
        # ---- OU process for B ----
        if cfg.new_spTimeGMC:
            B_hat = cfg.exp_B * B_hat + cfg.noise_B * FFT_CMG_noise(epsi_t,cfg)
        else:
            B_hat = cfg.exp_B * B_hat + cfg.noise_B * fftnoise(cfg)
        # ---- OU process for epsilon ----
        lepsi_hat = cfg.exp_e  * lepsi_hat + cfg.noise_e * np.sqrt(cfg.S2) * fftnoise(cfg, scalar = True)
        epsi_t = epsilon(lepsi_hat,cfg)

        V_x[t+1,:], V_y[t+1,:], V_z[t+1,:], stream_z[t+1,:] , vort_z[t+1,:], epsi[t+1,:] \
            = CMG_velocity(B_hat,epsi_t,cfg, intermittency = (not cfg.new_spTimeGMC) )

    return V_x, V_y, V_z, stream_z, vort_z, epsi



def fftnoise(cfg, scalar = False):
    xi = cfg.whiteNoise(scalar)
    B_hat = cfg.FFT(xi)
    cfg.cleanAliasing(B_hat)

    return B_hat

def FFT_CMG_noise(epsi,cfg):
    return FFT_CMG_from_noise(cfg.whiteNoise(),epsi,cfg)

def CMG_velocity(B_hat,epsi,cfg, intermittency = True):
    # ---- Real space ----
    B_real = cfg.IFFT(B_hat)
    Bepsi_hat = FFT_CMG_from_noise(B_real,epsi,cfg, intermittency)

    return velocity_from_FFTnoise(Bepsi_hat,epsi,cfg)

def epsilon(lepsi_hat,cfg):
    lepsi_real = cfg.IFFT(lepsi_hat)
    if cfg.dim == 3:
        lepsi_real = lepsi_real[:,:,:,:,np.newaxis]
    return cfg.intermittency(lepsi_real) 

def FFT_CMG_from_noise(B_real,epsi,cfg, intermittency = True):
    Bepsi = cfg.CMG_from_noise(B_real,epsi, intermittency)
    Bepsi_hat = cfg.FFT(Bepsi)
    cfg.cleanAliasing(Bepsi_hat)

    return Bepsi_hat

def FFT_streamfunction_from_noise(Bepsi_hat,cfg):
    return cfg.A * Bepsi_hat

def velocity_from_FFTnoise(Bepsi_hat,epsi,cfg):
    # ---- Streamfunction ----
    stream_hat = FFT_streamfunction_from_noise(Bepsi_hat,cfg)
    return velocity_from_FFTstream(stream_hat,epsi,cfg)

def velocity_from_FFTstream(stream_hat,epsi,cfg):
    # ---- Velocity ----
    if cfg.dim == 2:
        v_hat_x = - 1j * cfg.KY * stream_hat
        v_hat_y = + 1j * cfg.KX * stream_hat
        vort_hat = - cfg.K**2 * stream_hat
    elif cfg.dim == 3:
        v_hat = np.cross( 1j * cfg.Kv, stream_hat, axisa=3, axisb=3, axisc=3)
        vort_hat = np.cross( 1j * cfg.Kv, v_hat, axisa=3, axisb=3, axisc=3)

    stream = cfg.IFFT(stream_hat)
    vort = cfg.IFFT(vort_hat)
    if cfg.dim == 2:
        V_x = cfg.IFFT(v_hat_x)
        V_y = cfg.IFFT(v_hat_y)
        V_z = cfg.zeroSpatialField()
        stream_z = stream
        vort_z = vort
    elif cfg.dim == 3:
        V = cfg.IFFT(v_hat)

        epsi = epsi[:,:,:,0,:]
        stream_z = stream[:,:,:,2,:]
        vort_z = vort[:,:,:,2,:]
        V_x = V[:,:,:,0,:]
        V_y = V[:,:,:,1,:]
        V_z = V[:,:,:,2,:]

    # Set amplitude
    cfg.normalizing(V_x, V_y, V_z, stream_z, vort_z)

    return V_x, V_y, V_z, stream_z, vort_z, epsi

