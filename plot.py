import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import kurtosis, skew, norm
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

def save_spiky_field_gif_matplotlib(
        field,
        filename="epsi_field.gif",
        fps=10,
        vmin=None,
        vmax=None,
        cmap="inferno",
        lower_percentile=1,
        upper_percentile=99.7,
        add_colorbar=True):

    """
    Save a GIF from a time-evolving intermittent positive field.

    Parameters
    ----------
    field : np.ndarray
        Array of shape (t, x, y)

    filename : str
        Output GIF filename

    fps : int
        Frames per second

    vmin, vmax : float or None
        Color scale limits.
        If None, computed from percentiles.

    cmap : str
        Matplotlib colormap

    lower_percentile : float
        Percentile used for vmin if vmin is None

    upper_percentile : float
        Percentile used for vmax if vmax is None

    add_colorbar : bool
        Add colorbar to figure
    """

    nt = field.shape[0]

    # Compute robust global limits
    if vmin is None:
        vmin = np.percentile(field, lower_percentile)

    if vmax is None:
        vmax = np.percentile(field, upper_percentile)

    print(f"Using vmin={vmin:.4e}, vmax={vmax:.4e}")

    # Create figure
    fig, ax = plt.subplots()

    im = ax.imshow(
        field[0],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        animated=True
    )

    ax.axis("off")

    # Optional colorbar
    if add_colorbar:
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Intermittency [m^2/s^3]",labelpad=15)

    # Animation update
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

    # Save animation
    ani.save(
        filename,
        writer="pillow",
        fps=fps
    )

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

def _compute_structure_functions(signal, small_scale, large_scale, nv, Nreal):
    """
    Internal computation function.
    """

    N = signal.shape[0]

    # --- Log scales ---
    r_phys = 2**np.arange(np.log2(small_scale),
                          np.log2(large_scale),
                          1/nv)

    r_pix = np.unique((r_phys / small_scale).astype(int))
    r_pix = r_pix[(r_pix > 0) & (r_pix < N//2)]
    scales = r_pix

    S2_rows  = np.zeros((Nreal, len(scales)))
    S2_cols  = np.zeros((Nreal, len(scales)))
    Kurt_rows = np.zeros((Nreal, len(scales)))
    Kurt_cols = np.zeros((Nreal, len(scales)))

    for isc, scale in enumerate(scales):

        # Rows
        incrs = (signal[:-scale, :] - signal[scale:, :]).ravel()
        np.random.shuffle(incrs)
        usable = Nreal * (len(incrs) // Nreal)
        incrs = incrs[:usable].reshape(Nreal, -1)

        S2_rows[:, isc]  = np.var(incrs, axis=1)
        Kurt_rows[:, isc] = kurtosis(incrs, axis=1,
                                     fisher=False, bias=False)

        # Columns
        incrs = (signal[:, :-scale] - signal[:, scale:]).ravel()
        np.random.shuffle(incrs)
        usable = Nreal * (len(incrs) // Nreal)
        incrs = incrs[:usable].reshape(Nreal, -1)

        S2_cols[:, isc]  = np.var(incrs, axis=1)
        Kurt_cols[:, isc] = kurtosis(incrs, axis=1,
                                     fisher=False, bias=False)

    return scales, S2_rows, S2_cols, Kurt_rows, Kurt_cols

def plot_structure_functions(signal,
                             small_scale,
                             large_scale,
                             nv=16,
                             Nreal=100,
                             # --- S2 options ---
                             normalize=False,
                             scale_factor_R=1.0,
                             scale_factor_C=1.0,
                             # --- theory ---
                             plot_theory=False,
                             dim=2,
                             nu=None,
                             ell=None,
                             L=None,
                             dx=None,
                             # --- slopes ---
                             plot_slope=False,
                             c1=None,
                             # --- kurtosis ---
                             plot_kurtosis=False,
                             c2=None):
    
    scales, S2_r, S2_c, K_r, K_c = _compute_structure_functions(
        signal, small_scale, large_scale, nv, Nreal
    )

    x = scales * small_scale

    R_m = np.mean(S2_r, axis=0)
    C_m = np.mean(S2_c, axis=0)

    # =========================
    # S2 PLOT
    # =========================
    if normalize:
        R_m = R_m / np.max(R_m)
        C_m = C_m / np.max(C_m)

    plt.figure(figsize=(10, 6))
    plt.loglog(x, scale_factor_R * R_m, label='R_s')
    plt.loglog(x, scale_factor_C * C_m, label='C_s')

    # --- slope S2 ---
    if plot_slope:
        if c1 is None:
            raise ValueError("c1 required for S2 slope")

        K_ref = np.logspace(np.log10(x.min()), np.log10(x.max()), 100)
        S_ref = K_ref ** (2 * c1)
        S_ref *= R_m[-1] / S_ref[-1]

        plt.loglog(K_ref, S_ref, 'g--', label=f"Slope = {2*c1:.2f}")

    # --- theory ---
    if plot_theory:
        from sklearn.gaussian_process.kernels import Matern

        kernel = Matern(length_scale=ell, nu=nu)

        r_th = np.linspace(dx, L/2, 200)
        X = r_th.reshape(-1, 1)

        C_r = kernel(X, [[0]]).ravel()
        C0 = kernel([[0]], [[0]])[0, 0]

        S2_th = (C0 - C_r)
        S2_th /= C0

        plt.loglog(r_th, S2_th, 'r--', label="Theory (Matérn)")

    plt.xlabel("r")
    plt.ylabel("S2(r)")
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.show()

    # =========================
    # KURTOSIS PLOT
    # =========================
    if plot_kurtosis:
        K_r_m = np.mean(K_r, axis=0) / 3
        K_c_m = np.mean(K_c, axis=0) / 3

        plt.figure(figsize=(10, 6))

        plt.loglog(x, K_r_m, label=r"$F_r / 3$")
        plt.loglog(x, K_c_m, label=r"$F_c / 3$")

        # --- slope kurtosis ---
        if c2 is not None:
            K_ref = np.logspace(np.log10(x.min()), np.log10(x.max()), 100)
            F_ref = K_ref ** (-4 * c2)

            # normalize visually
            F_ref *= K_r_m[-1] / F_ref[-1]

            plt.loglog(K_ref, F_ref, 'k--', label=f"Slope = {-4*c2:.2f}")

        plt.xlabel("r")
        plt.ylabel("Flatness / 3")
        #plt.ylim(1,10)
        plt.legend()
        plt.grid(True, which="both", ls="--")
        plt.tight_layout()
        plt.show()

    return scales, x, S2_r, S2_c, K_r, K_c

def plot_increment_pdfs(signal,
                        small_scale,
                        large_scale,
                        nv=16,
                        pdf_scales_indices=(0, 5, 10, 15),
                        xlim_pdf=(-5, 5),
                        nbins=120,
                        vertical_spacing=5,
                        min_count=10,
                        normalize=True):
    
    N = signal.shape[0]

    # --- Log scales ---
    r_phys = 2**np.arange(np.log2(small_scale),
                          np.log2(large_scale),
                          1/nv)

    r_pix = np.unique((r_phys / small_scale).astype(int))
    r_pix = r_pix[(r_pix > 0) & (r_pix < N//2)]
    scales = r_pix #* small_scale


    # =========================
    # PDFs des incréments
    # =========================
    plt.figure()

    vertical_spacing = 5  # decalage entre courbes
    ncurves = len(pdf_scales_indices)

    for i, idx in enumerate(pdf_scales_indices):
        if idx >= len(scales):
            continue

        scale = scales[idx] 

        incr = (signal[:-scale, :] - signal[scale:, :]).ravel()
        incr = (incr - np.mean(incr)) / np.std(incr)

        nbins = 120
        hist, bins = np.histogram(incr, bins=nbins, density=False)

        centers = 0.5 * (bins[1:] + bins[:-1])
        bin_width = bins[1] - bins[0]

        pdf = hist / (np.sum(hist) * bin_width)

        # enlever bins mal résolus
        mask = hist > 10
        centers = centers[mask]
        pdf = pdf[mask]

        # Décalage multiplicatif
        shift_factor = vertical_spacing ** (ncurves - i - 1)
        pdf_shifted = pdf * shift_factor

        plt.scatter(centers, pdf_shifted,
                    s=15,
                    alpha=0.7,
                    label=f"l={scale}")

    plt.yscale("log")
    plt.xlim(-10, 10)
    plt.xlabel(r"$\delta u / \sigma$")
    plt.ylabel("PDF (offset)")
    plt.legend()
    plt.tight_layout()

    plt.show()

    def _compute_structure_functions_3D_ultra(signal,
                                          small_scale,
                                          large_scale,
                                          nv,
                                          Nreal,
                                          Nsamples_per_real=5000,
                                          pdf_scales_indices=None):

    signal = signal.astype(np.float32, copy=False)

    Nt, Nx, Ny, Nm = signal.shape

    # --- scales log ---
    r_phys = 2**np.arange(np.log2(small_scale),
                          np.log2(large_scale),
                          1/nv)

    r_pix = np.unique((r_phys / small_scale).astype(int))
    r_pix = r_pix[(r_pix > 0) & (r_pix < Nt // 2)]
    scales = r_pix

    S2 = np.zeros((Nreal, len(scales)), dtype=np.float32)
    Kurt = np.zeros((Nreal, len(scales)), dtype=np.float32)

    pdf_data = {}

    usable = Nreal * Nsamples_per_real

    for isc, scale in enumerate(scales):

        # ===== sampling rapide =====
        t = np.random.randint(0, Nt - scale, size=usable)
        x = np.random.randint(0, Nx, size=usable)
        y = np.random.randint(0, Ny, size=usable)

        incrs = signal[t, x, y, 0] - signal[t + scale, x, y, 0]
        incrs = incrs.reshape(Nreal, Nsamples_per_real)

        # ===== stats =====
        S2[:, isc] = np.var(incrs, axis=1)

        Kurt[:, isc] = kurtosis(incrs, axis=1,
                               fisher=False, bias=False)

        # ===== stockage PDF =====
        if pdf_scales_indices is not None and isc in pdf_scales_indices:
            pdf_data[isc] = incrs.copy()

    return scales, S2, Kurt, pdf_data

def plot_structure_functions_3D(signal,
                                small_scale,
                                large_scale,
                                nv=16,
                                Nreal=100,
                                Nsamples_per_real=5000,
                                L_norm=None,
                                c1=None,
                                c2=None,
                                pdf_scales_indices=(0, 5, 10, 15),
                                xlim_pdf=(-5, 5)):

    scales, S2, K, pdf_data = _compute_structure_functions_3D_ultra(
        signal,
        small_scale,
        large_scale,
        nv,
        Nreal,
        Nsamples_per_real,
        pdf_scales_indices
    )

    if L_norm is None:
        L_norm = signal.shape[0]

    x = scales * small_scale

    # =========================
    # S2
    # =========================
    plt.figure()
    plt.errorbar(x,
                 np.mean(S2, axis=0),
                 np.std(S2, axis=0),
                 color='b', label=r"S$^{t}_2$")

    if c1 is not None:
        plt.plot(x, 2 * c1 * x + 10.5, 'k--')

    plt.xlabel(r"l/L")
    plt.ylabel(r"$S_2$")
    plt.legend()
    plt.tight_layout()

    # =========================
    # Flatness
    # =========================
    plt.figure()
    plt.errorbar(x,
                 np.mean((K / 3), axis=0),
                 np.std((K / 3), axis=0),
                 color='b', label=r"$\frac{F^t}{3}$")

    if c2 is not None:
        plt.plot(x, -4 * c2 * x - 0.1, 'k--')

    plt.xlabel(r"l/L")
    plt.ylabel(r"F/3")
    #plt.ylim(0,1)
    plt.legend()
    plt.tight_layout()

    # =========================
    # PDFs
    # =========================
    if pdf_data:

        plt.figure()

        vertical_spacing = 5
        ncurves = len(pdf_data)

        for i, (isc, incrs) in enumerate(sorted(pdf_data.items())):

            incr = incrs.ravel()

            # normalisation
            incr = (incr - np.mean(incr)) / np.std(incr)

            hist, bins = np.histogram(incr, bins=120, density=False)

            centers = 0.5 * (bins[1:] + bins[:-1])
            bin_width = bins[1] - bins[0]

            pdf = hist / (np.sum(hist) * bin_width)

            mask = hist > 10
            centers = centers[mask]
            pdf = pdf[mask]

            shift = vertical_spacing ** (ncurves - i - 1)

            plt.scatter(centers,
                        pdf * shift,
                        s=15,
                        alpha=0.7,
                        label=f"$\\tau={scales[isc]}$")

        plt.yscale("log")
        plt.xlim(xlim_pdf)

        plt.xlabel(r"$\delta u / \sigma$")
        plt.ylabel("PDF (offset)")
        plt.legend()
        plt.tight_layout()

    plt.show()

    return x, S2, K