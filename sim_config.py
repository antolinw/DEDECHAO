# config.py
import numpy as np
from dataclasses import dataclass
from pytorch_wavelets import DWTForward, DWTInverse

@dataclass
class SimulationConfig:
    # --------- Numerical ---------
    n_order: int = 9              # N = 2^n_order = number of space steps along x
    N_steps: int = 128            # number of time steps
    n_ens: int = 1                # ensemble size
    method: str = 'Wavelet'       # Choice of method :'Fourier' or 'Wavelet'
    #method: str = 'Fourier'       # Choice of method :'Fourier' or 'Wavelet'
    wave: str = 'db4'             # choice of the wavelet
    J: int = 9                    # Number of level of decomposition for wavelet method
    beta: float = 1.              # Energy injected in wavelet spectrum
    mode: str = 'periodization'

    # --------- Physical ---------
    Lx: float = 2                  # domain size
    L_LS: float = 0.5 * Lx         # large spatial scale

    dim: float = 2.0               # spatial dimension
    c2: float = 0.025              # intermittency coefficient (must be < c1 /2, ]0,0.5[ in 2D)
    rho_s: float = 1.66            # spatial spectrum slope (]0,3[ in 2D; ]0,2[ in 3D)
    gamma_t: float = rho_s         # temporal spectrum slope
    c1 = (rho_s - 1) / 2.          # fractal exponent

    tau_0B: float = 1.0
    tau_0e_factor: float = np.inf # steady intermittency
    # tau_0e_factor: float = 1.0  # unsteady intermittency
    z_eps: float = 1.0

class Parameters(SimulationConfig): 
    def __init__(self):
        super().__init__()
    
        #Checking core dynamical parameters (2D)
        if (self.c2 >= self.c1 / 2) or (self.c1 >= 1.0):
            raise ValueError("Parameters must repect : c2 < c1/2 and c1 < 1.0 for 2D simulations. " \
                             + "Here : c1 = " + str(self.c1) + " and c2 = " + str(self.c2))
        else:
            print(f"Using consistent parameters: " \
                  + "c1 = {self.c1}, c2 = {self.c2}, rho_s = {self.rho_s}, gamma_t = {self.gamma_t}")

        N = 2 ** self.n_order
        self.PN = N//2                                 # aliased wavevector index
        self.dx = self.Lx / N

        z = (self.rho_s - 1) / (self.gamma_t - 1)      # scaling slope for time scale
        theta_1 = self.rho_s + self.dim + 1            # spectrum slope of the bidirectionnal spatial spectrum
        K_Lx = 2 * np.pi / self.Lx                     # largest-scale wavenumber
        K_LS = 2 * np.pi / self.L_LS                   # large-scale wavenumber
        self.K_LS = K_LS

        np.random.seed(0)

        # Wavenumbers
        kx = np.fft.fftfreq(N, d=self.dx) * 2 * np.pi
        if self.dim == 2 :
            KX, KY = np.meshgrid(kx, kx, indexing="ij")
            KX = KX[:,:,np.newaxis]
            KY = KY[:,:,np.newaxis]
            K = np.sqrt(KX**2 + KY**2)
            Kv = np.concat((KX, KY), axis= 2)
            self.KX = KX
            self.KY = KY
        elif self.dim == 3 :
            KX, KY, KZ = np.meshgrid(kx, kx, kx, indexing="ij")
            KX = KX[:,:,:,np.newaxis]
            KY = KY[:,:,:,np.newaxis]
            KZ = KZ[:,:,:,np.newaxis]
            K = np.sqrt(KX**2 + KY**2 + KZ**2)
            Kv = np.concat((KX, KY, KZ), axis= 3)
            self.KX = KX
            self.KY = KY
            self.KZ = KZ
            self.Kv = Kv[:,:,:,:,np.newaxis]
        else:
            raise ValueError("Incorrect dim")

        # Spectrum
        S1 = 1 / (1 + (K / K_LS)**2 ) ** (theta_1 / 2)
        self.cleanAliasing(S1)
        A = np.sqrt(S1)
        self.N = N

        self.wavelet = self.method == 'Wavelet'
        if (self.method == 'Wavelet'): # Wavelet method
            print(f"Using wavelet method with wavelet {self.wave} and J={self.J}")
            self.Jmax = np.log2(N)
            self.dwt = DWTForward(J=self.J, wave=self.wave, mode=self.mode)
            self.idwt = DWTInverse(wave=self.wave, mode=self.mode)
            if self.J > self.Jmax:
                raise ValueError("ACHTUNG : Requested J exceeds maximum wavelet scale !!!")
            
            # j = np.arange(1, self.J + 1)
            # Wavelet characteristic wavenumbers
            self.j = np.arange(self.J)
            self.k_j = K_Lx * (2.0 ** self.j)

            # ---------- OU time scales per scales ----------
            tau_B = self.tau_0B * (1 + (self.k_j / K_LS)**2) ** (-z / 2)
            tau_e = (self.tau_0e_factor * self.tau_0B * (1 + (self.k_j / K_LS)**2) ** (-self.z_eps / 2))
            # min_tau = min(tau_B.min(), tau_e.min())
            # print(f"Smallest physical time: {min_tau}")
            # print(f"Time step: dt = {self.Delta_t}")

            self.Delta_t = min(tau_B.min(), tau_e.min()) * 0.5

            # if min_tau <= self.Delta_t:
            #     print("ACHTUNG: time step may be too large!")
            exp_B = np.exp(-self.Delta_t / tau_B)
            noise_B = np.sqrt(1 - exp_B**2)

            exp_e = np.exp(-self.Delta_t / tau_e)
            noise_e = np.sqrt(1 - exp_e**2)
            
            self.KX = KX
            self.KY = KY
            self.K = K
            self.A = A

            self.c2_eff = self.c2 * np.log( self.L_LS / (self.dx) )

            self.exp_B = exp_B
            self.noise_B = noise_B
            self.exp_e = exp_e
            self.noise_e = noise_e

        elif (self.method == 'Fourier'): # Fourier method
            print("Using Fourier method")

            # OU timescales
            tau_B = self.tau_0B * (1 + (K / K_LS)**2 ) ** ( - z/2)
            tau_e = (self.tau_0e_factor * self.tau_0B) * (1 + (K / K_LS)**2 ) ** ( - self.z_eps/2)

            self.Delta_t = min(tau_B.min(), tau_e.min()) * 0.5

            exp_B = np.exp(-self.Delta_t / tau_B)
            noise_B = np.sqrt(1 - exp_B**2)

            exp_e = np.exp(-self.Delta_t / tau_e)
            noise_e = np.sqrt(1 - exp_e**2)

            np.seterr(divide = 'ignore') 
            self.c2_eff = self.c2 * np.log( self.L_LS / (self.dx) )
            S2 = ( K / K_Lx ) ** (-1)
            self.cleanAliasing(S2)
            np.seterr(divide = 'warn') 
        
        else:
            raise ValueError("Invalid method. Choose 'Fourier' or 'Wavelet'.")

        print("The time step is : dt = " + str(self.Delta_t))

        if self.dim == 3 :
            K = K[:,:,:,:,np.newaxis]
            A = A[:,:,:,:,np.newaxis]
            exp_B = exp_B[:,:,:,:,np.newaxis]
            noise_B = noise_B[:,:,:,:,np.newaxis]

        self.N = N
        self.K = K
        self.A = A
        self.exp_B = exp_B
        self.noise_B = noise_B
        self.exp_e = exp_e
        self.noise_e = noise_e
        self.S2 = S2
    
    def cleanAliasing(self,S1):
        if self.dim == 2 :
            S1[0, 0] = 0
            S1[self.PN, :] = 0
            S1[:, self.PN] = 0
        elif self.dim == 3 :
            S1[0, 0, 0] = 0
            S1[self.PN, :, :] = 0
            S1[:, self.PN, :] = 0
            S1[:, :, self.PN] = 0
        else:
            raise ValueError("Incorrect dim")

    def zeroField(self):
        if self.dim == 2 :
            zeroF = np.zeros((self.N_steps, self.N, self.N, self.n_ens))
        elif self.dim == 3 :
            zeroF = np.zeros((self.N_steps, self.N, self.N, self.N, self.n_ens))
        else:
            raise ValueError("Incorrect dim")
        return zeroF

    def zeroSpatialField(self):
        if self.dim == 2 :
            zeroF = np.zeros((self.N, self.N, self.n_ens))
        elif self.dim == 3 :
            zeroF = np.zeros((self.N, self.N, self.N, self.n_ens))
        else:
            raise ValueError("Incorrect dim")
        return zeroF
    
    def whiteNoise(self, scalar = False):
        if self.dim == 2 :
            xi = np.random.randn(self.N, self.N, self.n_ens)
        elif self.dim == 3 :
            xi = np.random.randn(self.N, self.N, self.N, 3*(not scalar) + scalar, self.n_ens)
            if scalar:
                xi = xi[:,:,:,0,:]
        else:
            raise ValueError("Incorrect dim")
        return xi
