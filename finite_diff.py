import numpy as np

def _ddx(f: np.ndarray, dx: float) -> np.ndarray:
    """d/dx along axis 0, central in interior, one-sided at boundaries."""
    df = np.empty_like(f)
    df[1:-1, ...] = (f[2:, ...] - f[:-2, ...]) / (2.0 * dx)
    df[0, ...]    = (f[1, ...] - f[0, ...]) / dx
    df[-1, ...]   = (f[-1, ...] - f[-2, ...]) / dx
    return df


def _ddy(f: np.ndarray, dy: float) -> np.ndarray:
    """d/dy along axis 1, central in interior, one-sided at boundaries."""
    df = np.empty_like(f)
    df[:, 1:-1, ...] = (f[:, 2:, ...] - f[:, :-2, ...]) / (2.0 * dy)
    df[:, 0, ...]    = (f[:, 1, ...] - f[:, 0, ...]) / dy
    df[:, -1, ...]   = (f[:, -1, ...] - f[:, -2, ...]) / dy
    return df


def _ddz(f: np.ndarray, dz: float) -> np.ndarray:
    """d/dz along axis 2, central in interior, one-sided at boundaries."""
    df = np.empty_like(f)
    df[:, :, 1:-1, ...] = (f[:, :, 2:, ...] - f[:, :, :-2, ...]) / (2.0 * dz)
    df[:, :, 0, ...]    = (f[:, :, 1, ...] - f[:, :, 0, ...]) / dz
    df[:, :, -1, ...]   = (f[:, :, -1, ...] - f[:, :, -2, ...]) / dz
    return df

def ortho_grad_2d(field: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """
    Compute the 2D spatial gradient of an ensemble of fields using finite differences.

    Parameters
    ----------
    field : ndarray, shape (Nx, Ny, n_ens)
    dx, dy : float

    Returns
    -------
    grad : ndarray, shape (Nx, Ny, 2, n_ens)
    """
    Nx, Ny, n_ens = field.shape
    grad = np.empty((Nx, Ny, 2, n_ens), dtype=field.dtype)
    grad[..., 0, :] = - _ddy(field, dx)
    grad[..., 1, :] = _ddx(field, dy)
    return grad

def grad_2d(field: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """
    Compute the 2D spatial gradient of an ensemble of fields using finite differences.

    Parameters
    ----------
    field : ndarray, shape (Nx, Ny, n_ens)
    dx, dy : float

    Returns
    -------
    grad : ndarray, shape (Nx, Ny, 2, n_ens)
    """
    Nx, Ny, n_ens = field.shape
    grad = np.empty((Nx, Ny, 2, n_ens), dtype=field.dtype)
    grad[..., 0, :] = _ddx(field, dx)
    grad[..., 1, :] = _ddy(field, dy)
    return grad

def lap_2d(field: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """
    Compute the 2D spatial Laplacian of an ensemble of fields using finite differences.

    Parameters
    ----------
    field : ndarray, shape (Nx, Ny, n_ens)
    dx, dy : float

    Returns
    -------
    lap : ndarray, shape (Nx, Ny, n_ens)
    """
    Nx, Ny, n_ens = field.shape
    lap = np.empty((Nx, Ny, n_ens), dtype=field.dtype)
    lap = _ddx(_ddx(field, dx), dx) + _ddy(_ddy(field, dy), dy)
    return lap


def curl_2d(vec_field: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """
    Scalar 2D curl: curl = dv/dx - du/dy.

    Parameters
    ----------
    vec_field : ndarray, shape (Nx, Ny, 2, n_ens)

    Returns
    -------
    curl : ndarray, shape (Nx, Ny, n_ens)
    """
    u = vec_field[:,:, 0, :]
    v = vec_field[:,:, 1, :]
    return _ddx(v, dx) - _ddy(u, dy)


def curl_3d(vec_field: np.ndarray, dx: float, dy: float, dz: float) -> np.ndarray:
    """
    Compute the 3D curl of an ensemble of vector fields using finite differences.

    curl F = ( dFz/dy - dFy/dz,
               dFx/dz - dFz/dx,
               dFy/dx - dFx/dy )

    Parameters
    ----------
    vec_field : ndarray, shape (Nx, Ny, Nz, 3, n_ens)
        vec_field[..., 0, :] = Fx
        vec_field[..., 1, :] = Fy
        vec_field[..., 2, :] = Fz
    dx, dy, dz : float
        Grid spacing in x, y, z.

    Returns
    -------
    curl : ndarray, shape (Nx, Ny, Nz, 3, n_ens)
        curl[..., 0, :] = dFz/dy - dFy/dz
        curl[..., 1, :] = dFx/dz - dFz/dx
        curl[..., 2, :] = dFy/dx - dFx/dy
    """
    Nx, Ny, Nz, three, n_ens = vec_field.shape
    assert three == 3, "vec_field must have shape (Nx, Ny, Nz, 3, n_ens)"

    Fx = vec_field[..., 0, :]  # (Nx, Ny, Nz, n_ens)
    Fy = vec_field[..., 1, :]
    Fz = vec_field[..., 2, :]

    dFz_dy = _ddy(Fz, dy)
    dFy_dz = _ddz(Fy, dz)

    dFx_dz = _ddz(Fx, dz)
    dFz_dx = _ddx(Fz, dx)

    dFy_dx = _ddx(Fy, dx)
    dFx_dy = _ddy(Fx, dy)

    curl = np.empty_like(vec_field)
    curl[..., 0, :] = dFz_dy - dFy_dz
    curl[..., 1, :] = dFx_dz - dFz_dx
    curl[..., 2, :] = dFy_dx - dFx_dy
    return curl

