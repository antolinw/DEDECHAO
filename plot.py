import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew

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
    plt.xlabel("Fréquence")
    plt.ylabel("Amplitude")
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.show()

    return kr, psd

def Incrs_anisotropic_generator2d(image, scalex, scaley):
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

def correlations_notime(data,scale_xlim, scale_ylim, dcx, dcy, Nanalyse, Nreal, scaleth):
    #data=np.load(path+file)
    #u=data['uf']
    u=data
    uabs=np.linalg.norm(u,axis=0)

    # Standard deviation estimations
    sigmax=np.std(u[:,:])
    sigmaabs=np.std(uabs)

    # Normalization
    u[:,:]=(u[:,:])/sigmaabs
    uabs=uabs/sigmaabs
    l1,l2=u.shape
    u=u+np.random.normal(0,10**-4,(l1,l2))
    ux=np.squeeze(u[:,:])
    uabs=uabs+np.random.normal(0,10**-4,(uabs.shape))
    N=len(ux)
    # Definition of scales
    scalesx=np.arange(-scale_xlim,scale_xlim+1,dcx)
    scalesy=np.arange(-scale_ylim,scale_ylim+1,dcy)

    # Initialization of matrix
    S2_x=np.zeros((Nreal, len(scalesx),len(scalesy)))
    Flat_x=np.zeros((Nreal, len(scalesx),len(scalesy)))
    S3_x=np.zeros((Nreal, len(scalesx),len(scalesy)))    
    # Estimation of entropy (and number of samples for entropy estimation) for each combination of scales (scalex,scaley)
    for isx in range(len(scalesx)):
        for isy in range(len(scalesy)): 
            scalex=scalesx[isx]
            scaley=scalesy[isy]
       
            if scalex!=0 or scaley !=0:
                # Generation of increments
                incrsx = Incrs_anisotropic_generator2d(ux, scalex, scaley)
                
                # Theiler + random permutation + reshape
                incrsx=incrsx[0:-1:scaleth,0:-1:scaleth].flatten()
                incrsx = np.random.permutation(incrsx)[0:int(len(incrsx)/Nanalyse)*Nanalyse].reshape(-1,Nanalyse)
           
                for ir in range(Nreal):
                    S2_x[ir,isy,isx] = np.mean(incrsx[ir,:]**2)
                    Flat_x[ir,isy,isx] = kurtosis(incrsx[ir,:], fisher=False, bias=False)
                    S3_x[ir,isy,isx] = np.mean(np.abs(incrsx[ir,:])**3)
               
    return S2_x, Flat_x, S3_x

def compute_correlations_notime(
    field,
    scale_lim=2**7,
    scale_step=5,
    scale_th=5,
    nb_realisation=1,
    N_ana=2**8
):
    """
    Compute structure-function-based correlations without time.

    Parameters
    ----------
    field : 2D ndarray
        Input field
    scale_lim : int
        Maximum scale
    scale_step : int
        Scale step
    scale_th : int
        Threshold scale
    nb_realisation : int
        Number of realizations
    N_ana : int
        Analysis grid size

    Returns
    -------
    S2, F, S3 : arrays
        Correlation outputs
    """
    S2, F, S3 = correlations_notime(
        field,
        scale_lim, scale_lim,
        scale_step, scale_step,
        N_ana,
        nb_realisation,
        scale_th
    )

    return S2, F, S3

def radial_average_2d(img):
    """
    Radial average of a 2D image around its center. 
    May be optimised as the 2D spectrum to avoid the loop.
    Not generalized for ensemble at the moment.

    Parameters
    ----------
    img : 2D ndarray

    Returns
    -------
    r : ndarray
        Radius (pixels)
    radial_mean : ndarray
        Mean value at each radius
    """
    h, w = img.shape
    cy, cx = (h - 1) / 2, (w - 1) / 2

    y, x = np.indices(img.shape)
    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
    r = dist.astype(int)

    rmax = r.max()
    radial_mean = np.zeros(rmax + 1)

    for radius in range(rmax + 1):
        mask = (r == radius)
        radial_mean[radius] = img[mask].mean()

    r_dist = np.arange(len(radial_mean))
    return r_dist, radial_mean

def plot_radial_profile_s(
    img,
    slope=None,
    slope_prefactor=1.0,
    slope_label=None,
    xlim=(1, 40),
    title="Radial Intensity"
):
    r, profile = radial_average_2d(img)

    plt.figure()
    plt.loglog(r[1:], profile[1:], label="Radial mean")

    if slope is not None:
        plt.loglog(
            r,
            slope_prefactor * r**slope,
            "r--",
            label=slope_label if slope_label else rf"$r^{{{slope}}}$"
        )

    plt.xlabel("Radius (pixels)")
    plt.ylabel("Mean intensity")
    plt.title(title)
    plt.xlim(*xlim)
    plt.grid(True, which="both")
    plt.legend()
    plt.show()

    return r, profile   

def plot_radial_profile(
    img,
    slope=None,
    slope_label=None,
    xlim=(1, 40),
    title="Radial Intensity"
):
    r, profile = radial_average_2d(img)
    slope_prefactor = profile[1]
    plt.figure()
    plt.loglog(r[1:], profile[1:], label="Radial mean")

    if slope is not None:
        plt.loglog(
            r,
            slope_prefactor * r**slope,
            "r--",
            label=slope_label if slope_label else rf"$r^{{{slope}}}$"
        )

    plt.xlabel("Radius (pixels)")
    plt.ylabel("Mean intensity")
    plt.title(title)
    plt.xlim(*xlim)
    plt.grid(True, which="both")
    plt.legend()
    plt.show()

    return r, profile 