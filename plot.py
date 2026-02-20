import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import kurtosis, skew
import cv2

def zeta_n(c1, c2, n):
    return c1*n - c2*n*(n-2)/2

def plot_surface(field, title="", cmap="viridis"):
    N = field.shape[0]
    x = np.linspace(-1, 1, N)
    y = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(X, Y, field, cmap=cmap)
    fig.colorbar(surf)
    ax.set_title(title)

    plt.show()

    plot_img(field, title=title, cmap=cmap)

def plot_img(field, title="", cmap="viridis"):
    plt.imshow(field)
    plt.title(title)
    plt.colorbar()
    plt.show()

def isotropic_psd_2d(u, dx=1.0, dy=1.0, window=False):
    """
    Isotropic 2D power spectral density of a real 2D field u.
    Code adapted from Professor. Talla (@mctallamakou)

    Parameters
    ----------
    u : 2D ndarray
    dx, dy : grid spacing
    window : bookr, ke_spec = isotropic_psd_2d(u_np, dx=1.0, dy=1.0, window=True)l
        If True, apply 2D Hamming window

    Returns
    -------
    kr : radial wavenumbers
    psd : isotropic PSD
    """
    nx, ny = u.shape
    assert nx % 2 == 0
    assert ny % 2 == 0

    # --- Optional Hamming window ---
    if window:
        wx = np.hamming(nx)[:, None]
        wy = np.hamming(ny)[None, :]
        u = u * (wx * wy)

    # Fourier transform
    uhat = np.fft.fft2(u)
    ke_hat = np.abs(uhat)**2
    ke_hat /= (nx * ny)**2

    # Wavenumbers
    kx = np.fft.fftfreq(nx, dx / (2*np.pi))
    ky = np.fft.fftfreq(ny, dy / (2*np.pi))
    kx, ky = np.meshgrid(kx, ky, indexing="ij")
    kh = np.sqrt(kx**2 + ky**2).ravel()

    # Radial bins
    dkx = 2*np.pi / (nx * dx)
    dky = 2*np.pi / (ny * dy)
    dkr = np.sqrt(dkx**2 + dky**2)

    kmax = min(np.pi/dx, np.pi/dy)
    kr = np.arange(0, kmax, dkr)

    # Ring mask
    lower = kr[:, None]
    upper = np.concatenate((kr[1:], [kr[-1] + dkr]))[:, None]
    mask = (kh >= lower) & (kh < upper)

    # Ring integration
    psd = (ke_hat.ravel()[None, :] * mask).sum(axis=1) / dkr

    kr = kr + dkr / 2

    return kr, psd

def plot_isotropic_psd(
    field,
    dx=1.0,
    dy=1.0,
    label="Field",
    slope=None,
    window=False
):
    """
    Compute and plot isotropic PSD of a 2D field.

    Parameters
    ----------
    field : 2D ndarray
    dx, dy : grid spacing
    scale : float
        Multiplicative factor for PSD (visual scaling)
    label : str
        Label for PSD curve
    slope : float or None
        Power-law slope (e.g. -5/3)
    slope_label : str or None
        Label for slope curve
    window : bool
        Apply Hamming window
    """
    kr, psd = isotropic_psd_2d(field, dx=dx, dy=dy, window=window)
    logA = np.mean(np.log(psd) - slope * np.log(kr))
    A = np.exp(logA)
    plt.figure()
    plt.loglog(kr, psd, label=label)

    if slope is not None:
        plt.loglog(
            kr,
            (kr**slope) * A,
            "--",
            label=rf"$k^{{{slope}}}$"
        )

    plt.legend()
    plt.xlabel("Wavenumber k")
    plt.ylabel("E(k)")
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.show()

    return kr, psd

def save_field_gif_matplotlib(field,
                              filename="field.gif",
                              fps=10,
                              vmin=None,
                              vmax=None,
                              cmap="viridis"):
    """
    Save a GIF from a time-evolving 2D field using matplotlib.animation.

    Parameters
    ----------
    field : np.ndarray
        Shape (t, x, y)
    filename : str
        Output GIF filename (current directory)
    fps : int
        Frames per second
    vmin, vmax : float or None
        Color scale limits
    cmap : str
        Colormap name
    """

    nt = field.shape[0]

    if vmin is None:
        vmin = field.min()
    if vmax is None:
        vmax = field.max()

    fig, ax = plt.subplots()
    im = ax.imshow(field[0], vmin=vmin, vmax=vmax, cmap=cmap)
    ax.axis("off")

    def update(frame):
        im.set_array(field[frame])
        return (im,)

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=nt,
        interval=1000 / fps,
        blit=True
    )

    # Save GIF
    ani.save(filename, writer="pillow", fps=fps)
    plt.close(fig)

    print(f"GIF saved as {filename}")

def compute_spectrum(signal, dt, axis=0):
    """
    Compute temporal power spectrum of a signal.

    Parameters
    ----------
    signal : np.ndarray
        Array with time as one axis, e.g. (t, x, y, 0)
    dt : float
        Time step
    axis : int
        Axis corresponding to time

    Returns
    -------
    w : np.ndarray
        Frequencies
    ke_hat : np.ndarray
        PSD (1D if 1 point, else multi-D)
    """

    N = signal.shape[axis]

    uhat = np.fft.rfft(signal, axis=axis)
    ke_hat = np.abs(uhat)**2 / N

    w = np.fft.rfftfreq(N, dt)
    
    return w, ke_hat

def plot_spectrum(w,
                  ke_hat,
                  filename="spectrum.png",
                  slope_ref=True,
                  slope=-5/3,
                  skip_zero=True):
    """
    Plot spectrum in log-log scale.

    Parameters
    ----------
    w : np.ndarray
        Frequencies
    ke_hat : np.ndarray
        Power spectral density
    slope_ref : bool
        Plot reference slope (default: True)
    slope : float
        Slope exponent (default: -5/3)
    slope_amp : float
        Amplitude of reference slope
    skip_zero : bool
        Skip zero frequency
    """

    if skip_zero:
        w_plot = w[2:]
        ke_plot = ke_hat[2:]
    else:
        w_plot = w
        ke_plot = ke_hat

    plt.figure(figsize=(8, 5))
    plt.loglog(w_plot, ke_plot, label="Spectrum")
    #slope_amp = ke_hat[2]
    logA = np.mean(np.log(ke_plot) - slope * np.log(w_plot))
    A = np.exp(logA)
    if slope_ref:
        plt.loglog(
            w_plot,
            A * w_plot**slope,
            '--',
            label=rf"$f^{{{slope}}}$",
            color="darkorange"
        )

    plt.xlabel("Frequency [Hz]")
    plt.ylabel("PSD")
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()

def plot_spectrum_mean_std(
    w,
    ke_hat,
    filename="spectrum_mean_std.png",
    slope_ref=True,
    slope=-5/3,
    skip_zero=True,
    std_factor=1.0,
):
    """
    Plot mean spectrum with standard deviation shading.

    Parameters
    ----------
    w : np.ndarray
        Frequencies (nfreq,)
    ke_hat : np.ndarray
        PSD array with shape (nfreq, ...)
    slope_ref : bool
        Plot reference slope
    slope : float
        Reference slope exponent
    skip_zero : bool
        Skip lowest frequencies
    std_factor : float
        Multiplier for std shading (1 = ±1σ)
    """

    # Remove zero / very-low frequencies if requested
    if skip_zero:
        w_plot = w[2:]
        ke_plot = ke_hat[2:]
    else:
        w_plot = w
        ke_plot = ke_hat

    # Average & std over spatial dimensions
    spatial_axes = tuple(range(1, ke_plot.ndim))

    log_ke = np.log(ke_plot)    
    log_mean = log_ke.mean(axis=spatial_axes)
    log_std = log_ke.std(axis=spatial_axes)

    ke_mean = np.exp(log_mean)                        # Mean spectrum
    ke_low  = np.exp(log_mean - std_factor * log_std) # Lower envelope
    ke_high = np.exp(log_mean + std_factor * log_std) # Upper envelope

    plt.figure(figsize=(8, 5))

    # Mean spectrum
    plt.loglog(w_plot, ke_mean, label="Mean spectrum")

    # Std envelope
    plt.fill_between(
        w_plot,
        ke_low,
        ke_high,
        alpha=0.3,
        label=rf"$\pm {std_factor}\sigma$",
    )

    # Reference slope
    if slope_ref:
        logA = np.mean(np.log(ke_mean) - slope * np.log(w_plot))
        A = np.exp(logA)
        plt.loglog(
            w_plot,
            A * w_plot**slope,
            "--",
            color="darkorange",
            label=rf"$f^{{{slope}}}$",
        )

    plt.xlabel("Frequency [Hz]")
    plt.ylabel("PSD")
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()


### New structure functions (not used in the main code) ###
def Incrs_anisotropic_generator2d(image, scalex, scaley):
    '''    
    This function generate a vector of increments at a given scale_x, scale_y from a 2d image
 
    
    incrs = Incrs_anisotropic_generator2d(image, scalex, scaley)
    
    Input arguments :
        
        image  : image to analyze
        scalex    : size of scale in x dim
        scaley    : size of scale in y dim
        
    Output arguments :
        
        incr : vector of increments to compute statistics
        
    Usage example :
        
        image=np.random.randn((1000,1000))
        incr = Incrs_anisotropic_generator2d(image,scalex, scaley)

    Comments: In python first dimension is vertical alignment (y direction in this program). Second dimension is horizontal alignment (x direction)

    Comments: If image=array([[1, 2, 3],
                              [4, 5, 6],
                              [7, 8, 9]])
    
    scalex=scaley=1 then: image2=array([[9, 7, 8],
                                       [3, 1, 2],
                                       [6, 4, 5]])
    and so to define the increments such as Frisch 1995 (x(r+l) -x(r)) we have to define image -image2.
    

    ##
    C. Granero-Belinchon (IMT- Atlantique), September 2021
 
    '''  
    # check parameters
    if  image.ndim != 2 :
        raise TypeError('Image must be two dimensional.\n')
    
    image2=np.roll(image , scaley, axis=0)
    image2=np.roll(image2, scalex, axis=1)
    
    if scaley>=0 and scalex>=0:
        incr=-(image2[scaley::,scalex::]-image[scaley::,scalex::])
    elif scaley<0 and scalex<0:
        incr=-(image2[:scaley,:scalex]-image[:scaley,:scalex])
    elif scaley>=0 and scalex<0:
        incr=-(image2[scaley::,:scalex]-image[scaley::,:scalex])
    elif scaley<0 and scalex>=0:
        incr=-(image2[:scaley,scalex::]-image[:scaley,scalex::])
    
    return incr

def compute_moments_2D(data, moms=np.array((2,3))):
    """
    Function Moms = compute_moments_2D(data, moms) estimates the statistical moments of order moms of the given ''data''

    Input arguments :
        data   : signal of length Nx x Ny
        moms     : (1d array) order of the moments to estimate (default=np.array((2,3)))
    
    Output :
        Moms    : moments of the sampled image, len(moms) moments are returned
        
    Usage example :
        
        data=np.random.randn((1000,1000))
        incr = compute_moments_2D(data, moms=2)
    
    ##
    C. Granero-Belinchon (IMT- Atlantique), N.B Garnier (ENS Lyon), October 2021  
    """
    
    m = np.array(moms).flatten()
    Moments=np.zeros(len(m)) # Initialization of moments matrix 
    
    for i in range(len(m)):      
            Moments[i]=np.nanmean(data.flatten()**m[i]) 
             
    return Moments

def polar_axis(data, n_theta=-1):
    '''    
    This function returns 2 numpy arrays characterizing the polar coordinates 
    of the output field of the function field_2D_polar:
 
    axis_t, axis_r, size = polar_axis(data, n_theta=-1)

    Input arguments :
        
        data  : image to analyze
        n_theta : nb points desired in the theta direction if =-1: auto-detect (as R_max*pi) 
        
    Output arguments :
        
        axis_t : vector of angles
        axis_r : vector of radius
        size : shape of field in polar coordinates
        
    Usage example :
        
        image=np.random.randn((1000,1000))
        axis_t, axis_r, size = polar_axis(image, n_theta=-1)

   
    ##
    N.B Garnier (ENS Lyon) and C. Granero-Belinchon (IMT- Atlantique), September 2021
 
    '''  

    Nx,Ny = data.shape
    R_max = min(Nx//2,Ny//2)
    N_theta = n_theta if n_theta>0 else (int)(R_max*np.pi)
    size  = (R_max+1, N_theta)      # N_theta pts in [0,2*pi[ and R (tau) in [O, R_max]
    axis_t= np.arange(N_theta)/N_theta*2*np.pi - np.pi/2
    axis_r= np.arange(R_max+1)
    return axis_t, axis_r, size

   
def field_2D_polar(H, n_theta=-1):    
    '''    
    This function returns the cartesian field H in polar coordinates. The center of the field is used as origin.
    First dimension is the angle in [0, 2*pi[, second dimension is the radius in [0, R_max]
 
    H_polar = field_2D_polar(H, n_theta=-1)

    Input arguments :
        
        H  : field to analyze
        n_theta : nb points desired in the theta direction if =-1: auto-detect (as R_max*pi) 
        
    Output arguments :
        
        H_polar : H field in polar coordinates
        
    Usage example :
        
        image=np.random.randn((1000,1000))
        image_polar = field_2D_polar(image, n_theta=-1)

   
    ##
    N.B Garnier (ENS Lyon) and C. Granero-Belinchon (IMT- Atlantique), September 2021
 
    '''  

    Nx,Ny = H.shape
    H_t= np.transpose(H) # cv2 works on images, which are transposed of matrices
    origin= (Nx//2, Ny//2) 
    axis_t, axis_r, size = polar_axis(H, n_theta)
    R_max = size[0]-1
    return cv2.warpPolar(H_t, size, origin, R_max, cv2.WARP_POLAR_LINEAR+cv2.INTER_CUBIC)

def compute_structure_function_polar(H, dx, l, L, nv=16, moms=np.array([2, 3]), n_theta=-1):
    '''
    Calcule les fonctions de structure d'ordre `moms` du champ H en coordonnées polaires.
    Retourne uniquement les moments demandés.

    Args:
        H: Champ à analyser (2D array).
        dx; Pas de grille (taille d'un pixel).
        l: Smallest scale
        L: Integral scale
        nv: Nombre de points pour la densité des échelles
        moms: Tableau des ordres des moments à calculer (ex: [2, 3, 4]).
        n_theta: Nombre de points dans la direction theta. Si -1, auto-détection.

    Returns:
        axis_t, axis_r, S_polar, S_Om: Axes et résultats pour chaque moment demandé.
    '''
    N = H.shape[0]

    # ---- échelles physiques log ----
    r_phys = 2**np.arange(np.log2(l), np.log2(L), 1/nv)

    # conversion en pixels
    r_pix = np.unique((r_phys / dx).astype(int))

    # garder seulement les valeurs valides
    r_pix = r_pix[(r_pix > 0) & (r_pix < N//2)]

    # échelles cartésiennes
    scalesx = np.concatenate((-r_pix[::-1], [0], r_pix))
    scalesy = scalesx.copy()

    # ---- Initialisation ----
    S = {m: np.zeros((len(scalesy), len(scalesx))) for m in moms}

    # ---- Boucle principale ----
    for isx, scalex in enumerate(scalesx):
        for isy, scaley in enumerate(scalesy):

            if scalex != 0 or scaley != 0:
                incrs = Incrs_anisotropic_generator2d(H, scalex, scaley)
                for m in moms:
                    S[m][isy, isx] = compute_moments_2D(incrs, moms=m)

    # ---- Centre ----
    center = len(scalesx)//2    
    for m in moms:
        S[m][center, center] = np.nan
        S[m][center, center] = np.nanmean(
            S[m][center-1:center+2, center-1:center+2]
        )

    # ---- Passage en polaire ----
    axis_t, axis_r, _ = polar_axis(S[moms[0]], n_theta=n_theta)

    S_polar = {m: field_2D_polar(S[m], n_theta=n_theta) for m in moms}
    S_Om = {m: np.sum(S_polar[m], axis=0) for m in moms}

    return axis_t, axis_r, S_polar, S_Om


def plot_structure_function_polar(
    H,
    dx,
    l,
    L,
    nv=16,
    moms_to_plot=[2, 3],  # Par défaut, on trace S2 et S3
    slopes={},            # Dictionnaire des pentes (ex: {2: 0.5, 3: 0.8})
    title="Structure functions in polar coordinates",
    ref_r=None           # Point de référence pour l'alignement vertical
):
    # Calcule uniquement les moments demandés
    axis_t, axis_r, S_polar, S_Om = compute_structure_function_polar(H, dx, l, L, nv, moms=moms_to_plot)

    plt.figure(figsize=(8, 5))

    for m in moms_to_plot:
        S_Om_temp = axis_r * S_Om[m]
        plt.loglog(axis_r, axis_r * S_Om_temp, label=rf"omnidirectionnal $S_{m}$")

        if m in slopes:
            slope = slopes[m]
            # Calcule la valeur de S_Om_temp au point de référence
            ref_S_Om = np.interp(ref_r, axis_r, S_Om_temp)
            # Calcule la valeur de la référence au point de référence
            ref_ref = ref_r ** slope
            # Facteur de normalisation
            norm_factor = ref_S_Om / ref_ref
            # Trace la courbe de référence normalisée
            plt.loglog(
                axis_r,
                norm_factor * axis_r**slope,
                "--",
                label=rf"$r^{{{slope}}}$ (ref of $S_{m}$)"
            )

    plt.xlabel("Radius r")
    plt.ylabel("Structure function")
    plt.title(title)
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.show()


    # flatness
    plt.figure(figsize=(8, 5))
    flatness = (axis_r * S_Om[4]) / ( (axis_r * S_Om[2])**2 )
    slope = slopes[4] - 2*slopes[2]

    plt.loglog(axis_r, flatness, label=rf"flatness")
    # Calcule la valeur de S_Om_temp au point de référence
    ref_flatness = np.interp(ref_r, axis_r, flatness)
    # Calcule la valeur de la référence au point de référence
    ref_ref = ref_r ** slope
    # Facteur de normalisation
    norm_factor = ref_flatness / ref_ref
    # Trace la courbe de référence normalisée
    plt.loglog(
        axis_r,
        norm_factor * axis_r**slope,
        "--",
        label=rf"$r^{{{slope}}}$ (ref for flatness)"
    )

    plt.xlabel("Radius r")
    plt.ylabel("Flatness")
    plt.title(title)
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.show()