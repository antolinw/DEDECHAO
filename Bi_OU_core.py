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
    B_hat = fftnoise(cfg)
    V_x[0,:], V_y[0,:], V_z[0,:], stream_z[0:], vort_z[0:], epsi[0,:]\
        = CMG_velocity(B_hat,lepsi_hat,cfg)

    for t in range(cfg.N_steps-1):
        # ---- OU process for B ----
        B_hat = cfg.exp_B * B_hat + cfg.noise_B * fftnoise(cfg)
        # ---- OU process for epsilon ----
        lepsi_hat = cfg.exp_e  * lepsi_hat + cfg.noise_e * np.sqrt(cfg.S2) * fftnoise(cfg, scalar = True)

        V_x[t+1,:], V_y[t+1,:], V_z[t+1,:], stream_z[t+1,:] , vort_z[t+1,:], epsi[t+1,:] \
            = CMG_velocity(B_hat,lepsi_hat,cfg)

    return V_x, V_y, V_z, stream_z, vort_z, epsi



def fftnoise(cfg, scalar = False):
    xi = cfg.whiteNoise(scalar)
    B_hat = np.fft.fft2(xi, axes=(0,1)) 
    if cfg.dim == 3:
        B_hat = np.fft.fft(B_hat, axis=2) 

    cfg.cleanAliasing(B_hat)

    return B_hat


def CMG_velocity(B_hat,lepsi_hat,cfg):

    # ---- Real space ----
    B_real = np.fft.ifft2(B_hat, axes=(0,1)).real
    if cfg.dim == 3:
        B_real = np.fft.ifft(B_real, axis=2).real
    # B_real = (B_real - B_real.mean()) / B_real.std() # needed ???

    lepsi_real = np.fft.ifft2(lepsi_hat, axes=(0,1)).real
    if cfg.dim == 3:
        lepsi_real = np.fft.ifft(lepsi_real, axis=2).real
        lepsi_real = lepsi_real[:,:,:,:,np.newaxis]
    # Set the correct variance (related to c2)
    lepsi_real /= lepsi_real.std() 
    lepsi_real *= np.sqrt( cfg.c2_eff )
    lepsi_real -= lepsi_real.var() 
    epsi = np.exp(lepsi_real)

    Bepsi_hat = np.fft.fft2(B_real * epsi, axes=(0,1))
    if cfg.dim == 3:
        Bepsi_hat = np.fft.fft(Bepsi_hat, axis=2) 
    cfg.cleanAliasing(Bepsi_hat)

    # ---- Velocity ----
    stream_hat = cfg.A * Bepsi_hat
    if cfg.dim == 2:
        v_hat_x = 1j * cfg.KY * stream_hat
        v_hat_y = -1j * cfg.KX * stream_hat
    elif cfg.dim == 3:
        v_hat = np.cross( 1j * cfg.Kv, stream_hat, axisa=3, axisb=3, axisc=3)
    vort_hat = - cfg.K**2 * stream_hat

    stream = np.fft.ifft2(stream_hat, axes=(0,1)).real
    vort = np.fft.ifft2(vort_hat, axes=(0,1)).real
    if cfg.dim == 2:
        V_x = np.fft.ifft2(v_hat_x, axes=(0,1)).real
        V_y = np.fft.ifft2(v_hat_y, axes=(0,1)).real
        V_z = cfg.zeroSpatialField()
        stream_z = stream
        vort_z = vort
    if cfg.dim == 3:
        stream = np.fft.ifft(stream, axis=2).real
        vort = np.fft.ifft(vort, axis=2).real
        V = np.fft.ifft2(v_hat, axes=(0,1)).real
        V = np.fft.ifft(V, axis=2).real

        epsi = epsi[:,:,:,0,:]
        stream_z = stream[:,:,:,2,:]
        vort_z = vort[:,:,:,2,:]
        V_x = V[:,:,:,0,:]
        V_y = V[:,:,:,1,:]
        V_z = V[:,:,:,2,:]

    # Set amplitude
    ampli = cfg.K_LS * np.sqrt((stream_z[:]**2).mean())
    stream_z /= ampli
    V_x /= ampli
    V_y /= ampli
    V_z /= ampli
    vort_z /= ampli

    return V_x, V_y, V_z, stream_z, vort_z, epsi

