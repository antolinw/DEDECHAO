import pywt
import torch

# Separable orthogonal DWT with periodization, for any number of spatial dimensions
# (pytorch_wavelets only provides 1D and 2D transforms).
# Same interface as pytorch_wavelets.DWTForward / DWTInverse :
#   input  x : (n_ens, C, N_1, ..., N_d)
#   output A : (n_ens, C, n_1, ..., n_d)              coarsest approximation
#          D : [D_1 (finest), ..., D_J (coarsest)]    with D_j : (n_ens, C, 2^d - 1, n_1, ..., n_d)
# The subbands of D_j are ordered by the binary code (b_1,...,b_d), 0 = low-pass, 1 = high-pass
# along each axis, without the all-low-pass subband (e.g. in 3D: aad, ada, add, daa, dad, dda, ddd).
# With periodization, the transform is orthonormal for any N_i = 2^n (even when N_i < filter length).


def _filters(wave, mode):
    if mode != 'periodization':
        raise ValueError("Only mode='periodization' is implemented for the N-D wavelet transform.")
    w = pywt.Wavelet(wave)
    if not w.orthogonal:
        raise ValueError("Wavelet " + wave + " is not orthogonal.")
    return [float(c) for c in w.rec_lo], [float(c) for c in w.rec_hi]


def _analysis_1d(x, axis, h, g):
    """ lo[n] = sum_k h[k] x[(2n+k) mod L] ; hi[n] = sum_k g[k] x[(2n+k) mod L] """
    L = x.shape[axis]
    n = torch.arange(L // 2, device=x.device)
    lo = 0.
    hi = 0.
    for k, (hk, gk) in enumerate(zip(h, g)):
        xk = x.index_select(axis, (2 * n + k) % L)
        lo = lo + hk * xk
        hi = hi + gk * xk
    return lo, hi


def _synthesis_1d(lo, hi, axis, h, g):
    """ Adjoint (= inverse, by orthonormality) of _analysis_1d """
    L = 2 * lo.shape[axis]
    shape = list(lo.shape)
    shape[axis] = L
    x = lo.new_zeros(shape)
    n = torch.arange(L // 2, device=lo.device)
    for k, (hk, gk) in enumerate(zip(h, g)):
        x.index_add_(axis, (2 * n + k) % L, hk * lo + gk * hi)
    return x


class DWTForwardND:
    def __init__(self, J=1, wave='db4', mode='periodization'):
        self.J = J
        self.h, self.g = _filters(wave, mode)

    def __call__(self, x):
        axes = range(2, x.ndim)
        D = []
        A = x
        for _ in range(self.J):
            if min(A.shape[2:]) < 2:
                raise ValueError("Too many decomposition levels for the field size.")
            bands = [A]
            for axis in axes:
                bands = [b for band in bands for b in _analysis_1d(band, axis, self.h, self.g)]
            A = bands[0]
            D.append(torch.stack(bands[1:], dim=2))
        return A, D


class DWTInverseND:
    def __init__(self, wave='db4', mode='periodization'):
        self.h, self.g = _filters(wave, mode)

    def __call__(self, coeffs):
        A, D = coeffs
        axes = range(2, A.ndim)
        for Dj in reversed(D):
            bands = [A] + list(Dj.unbind(dim=2))
            for axis in reversed(axes):
                bands = [_synthesis_1d(bands[i], bands[i + 1], axis, self.h, self.g)
                         for i in range(0, len(bands), 2)]
            A = bands[0]
        return A
