import numpy as np
from scipy.signal.windows import tukey
from SR_plots import isotropic_psd_2d 
from dataclasses import replace
from genCHAO.Bi_OU_core import run_simulation
from genCHAO.sim_config import Parameters   

def filtre_image_k(image, dx, dy, k_coupure, passe_haut=True):
    """
    Filtre passe-haut / passe-bas dans l'espace des wavenumbers physiques,
    cohérent avec un binning radial en 2*pi/dx.

    Paramètres :
    -----------
    image : ndarray (2D)
        L'image à filtrer.
    dx, dy : float
        Résolution spatiale (mêmes unités que pour dkx, dky, kr).
    k_coupure : float
        Fréquence de coupure en wavenumber physique.
    passe_haut : bool
        True = passe-haut, False = passe-bas.

    Retourne :
    ----------
    image_filtree : ndarray
        Image filtrée (réelle).
    """
    ny, nx = image.shape

    # wavenumbers (convention 2*pi)
    kx = np.fft.fftfreq(nx, dx / (2*np.pi))
    ky = np.fft.fftfreq(ny, dy / (2*np.pi))
    kx, ky = np.meshgrid(kx, ky, indexing="ij")
    kh = np.sqrt(kx**2 + ky**2)

    if passe_haut:
        masque = kh > k_coupure
    else:
        masque = kh <= k_coupure

    # Pas de fftshift ici : kh est dans le même ordre "natif" que fft2
    fft_image = np.fft.fft2(image)
    fft_filtree = fft_image * masque

    image_filtree = np.fft.ifft2(fft_filtree).real

    return image_filtree

def radial_average(field_2d, k, k_bins):
    """Somme radiale d'un champ 2D (ex: |FFT|^2) sur des couronnes de k,
    normalisée par dk pour obtenir une densité spectrale E(k)."""
    dk = k_bins[1] - k_bins[0]
    E_1d = np.zeros(len(k_bins))
    for idx, k_i in enumerate(k_bins):
        mask = (k >= k_i) & (k < k_i + dk)
        if mask.any():
            E_1d[idx] = field_2d[mask].sum() / dk
    return E_1d

# Version dans QGMLU-DEDECHAO

def moduler_bruit(image, bruit, k_c, alpha, dk=1.0):
    """
    image : array 2D réel, l'écoulement original
    bruit : array 2D réel, ton bruit déjà filtré (pas de basse fréquence)
    k_c   : nombre d'onde de coupure (là où on veux raccorder les pentes)
    alpha : pente théorique cible (E_theo(k) ~ k^-alpha)
    """

    N = image.shape[0]
    assert bruit.shape == image.shape

    # --- grille de nombre d'onde ---
    kx = np.fft.fftfreq(N) * N
    ky = np.fft.fftfreq(N) * N
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    k = np.sqrt(KX**2 + KY**2)

    #k_max = kx.max()
    k_max = k.max()
    k_bins = np.arange(0, k_max, dk)

    # --- FFT de l'image et du bruit ---
    I_hat = np.fft.fft2(image)
    B_hat = np.fft.fft2(bruit)

    E_orig_2d  = np.abs(I_hat)**2
    E_bruit_2d = np.abs(B_hat)**2

    E_orig_1d  = radial_average(E_orig_2d,  k, k_bins)
    E_bruit_1d = radial_average(E_bruit_2d, k, k_bins)

    # --- spectre théorique cible, raccordé en k_c ---

    idx_c = int((k_c - k_bins[0]) // dk)
    idx_c = np.clip(idx_c, 0, len(k_bins) - 1)
    
    #idx_c = np.searchsorted(k_bins, k_c)
    A = E_orig_1d[idx_c] * k_c**alpha if E_orig_1d[idx_c] > 0 else E_bruit_1d[idx_c] * k_c**alpha
    E_theo_1d = A * np.where(k_bins > 0, k_bins, np.inf)**(-alpha)

    # --- résidu : ce qui manque par rapport à l'image ---
    R_1d = np.maximum(E_theo_1d - E_orig_1d, 0)
    R_1d[k_bins < k_c] = 0   # rien à ajouter en dessous de k_c

    # --- facteur de modulation M(k) = sqrt(R / E_bruit), en évitant les divisions par 0 ---
    M_1d = np.zeros_like(R_1d)
    valid = E_bruit_1d > 1e-9
    ratio = np.zeros_like(R_1d)
    np.divide(R_1d, E_bruit_1d, out=ratio, where=valid)
    #M_1d[valid] = np.sqrt(R_1d[valid] / E_bruit_1d[valid])
    #M_1d = np.sqrt(R_1d / E_bruit_1d)
    M_1d = np.sqrt(ratio)
    
    # si le bruit n'a aucune énergie à un k donné mais qu'on en voudrait -> impossible de créer
    # de l'énergie avec un facteur multiplicatif, on laisse M=0 

    # --- passage du filtre radial 1D vers la grille 2D ---
    k_centers = k_bins + dk/2
    M_2d = np.interp(k, k_bins, M_1d)

    # --- application du filtre au bruit (multiplicatif, dans Fourier) ---
    B_hat_modulé = B_hat * M_2d

    bruit_modulé = np.real(np.fft.ifft2(B_hat_modulé))

    image_finale = image + bruit_modulé

    return image_finale, bruit_modulé

def lin_slope_fit(kr, psd, kr_cut):
    k_min_fit = kr[2]          # évite l'injection)
    k_max_fit = kr_cut         # jusqu'à la coupure
    
    mask = (kr >= k_min_fit) & (kr <= k_max_fit)
    
    log_k   = np.log10(kr[mask])
    log_psd = np.log10(psd[mask])
    
    # Fit linéaire -> pente spectrale
    coeffs = np.polyfit(log_k, log_psd, deg=1)
    pente  = coeffs[0]         # exposant spectral  (ex: -5/3 Kolmogorov)
    offset = coeffs[1]

    print(f"Effective spectral slope : {pente:.3f}  (Kolmogorov théorique : {-5/3:.3f})")
    
    return pente, offset

def fourier_upsample_tukey(img_lr, target_size, alpha=0.2):
    """
    Upsample par zero-padding Fourier avec fenêtrage Tukey.
    alpha=0.2 : 20% des bords sont atténués, 80% du centre intact.
    """
    N_in  = img_lr.shape[0]
    N_out = target_size

    # Fenêtre Tukey 2D
    t     = tukey(N_in, alpha=alpha)
    window = np.outer(t, t)

    # FFT avec fenêtrage
    #F_lr = np.fft.fftshift(np.fft.fft2(img_lr * window))
    F_lr = np.fft.fftshift(np.fft.fft2(img_lr))
    
    # Zero-padding
    F_hr = np.zeros((N_out, N_out), dtype=complex)
    pad  = (N_out - N_in) // 2
    F_hr[pad:pad+N_in, pad:pad+N_in] = F_lr

    # IFFT
    img_hr = np.fft.ifft2(np.fft.ifftshift(F_hr)).real
    img_hr *= (N_out / N_in) ** 2

    return img_hr

def up_and_down_sample(field, factor):
    # facteur d'upscale (8 et 16 sont ok)
    HR = field
    HR = (HR - HR.min()) / (HR.max() - HR.min())  # normalisation [0, 1]

    N_HR = np.shape(HR)[0]  # taille HR
    N_LR = N_HR // factor   # taille LR   
    
    dx_HR = 2 / N_HR
    dy_HR = 2 / N_HR

    k_cut_physique = (N_LR / 2) * (2*np.pi / 2)
    
    HR_U_blurred = filtre_image_k(HR, dx_HR, dx_HR, k_coupure=k_cut_physique, passe_haut=False)
    
    # Downsampling par moyenne de blocs
    U_LR = HR_U_blurred.reshape(N_LR, factor, N_LR, factor).mean(axis=(1, 3))
    
    # Spectra
    kr_u_HR,  psd_u_HR  = isotropic_psd_2d(HR, dx=2/np.shape(HR)[0], dy=2/np.shape(HR)[0])
    kr_u_LR,  psd_u_LR  = isotropic_psd_2d(U_LR, dx=2/np.shape(U_LR)[0], dy=2/np.shape(U_LR)[0])

    #slope lin-fit
    pente, offset = lin_slope_fit(kr_u_HR, psd_u_HR, k_cut_physique)
    
    #upsample
    U_LR_up = fourier_upsample_tukey(U_LR, N_HR, alpha=0.2)
    
    return U_LR, U_LR_up, pente, offset

def super_resolution(field, factor, k_injection, variance_red = 200):

    U_LR, U_LR_up, slope, offset = up_and_down_sample(field, factor)

    # Generate noise of the wanted slope
    cfg0 = Parameters()
    cfg = replace(cfg0, rho_s=-slope, N_steps=1)
    V_x, V_y = run_simulation(cfg)
    fieldx = V_x[0,:,:,0] - V_x[0,:,:,0].mean()
    fieldx /= (fieldx.std() * variance_red)

    # Modulating
    IM_U, _ = moduler_bruit(U_LR_up, fieldx, k_c= k_injection, alpha= -slope, dk=1.0)  # factor = 16

    return IM_U, U_LR

def load_uv_field(filename, remove_mean=True, normalize=False):
    """
    Lit un fichier texte (x, y, U, V) et reconstruit les champs sur une grille 2D.

    Paramètres
    ----------
    filename    : str   chemin du fichier .txt
    remove_mean : bool  soustrait la moyenne à U et V
    normalize   : bool  divise par l'écart-type (après soustraction de la moyenne)

    Retourne
    -------
    X, Y : grilles 2D (shape (ny, nx))
    U, V : champs 2D (shape (ny, nx))
    """
    # 1. Lecture
    data = np.loadtxt(filename)
    x, y, U, V = data[:, 0], data[:, 1], data[:, 2], data[:, 3]

    # 2. Dimensions de la grille
    x_unique = np.unique(x)
    y_unique = np.unique(y)
    nx, ny = len(x_unique), len(y_unique)

    # 3. Reconstruction (tri par y puis x pour être robuste à l'ordre du fichier)
    order = np.lexsort((x, y))  # tri principal sur y, secondaire sur x
    U_matrix = U[order].reshape((ny, nx))
    V_matrix = V[order].reshape((ny, nx))

    # 4. Grilles X, Y
    X, Y = np.meshgrid(x_unique, y_unique)

    # 5. Post-traitement optionnel
    if remove_mean:
        U_matrix = U_matrix - U_matrix.mean()
        V_matrix = V_matrix - V_matrix.mean()
    if normalize:
        U_matrix = U_matrix / U_matrix.std()
        V_matrix = V_matrix / V_matrix.std()

    return X, Y, U_matrix, V_matrix