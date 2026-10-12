# config.py
import numpy as np
from dataclasses import dataclass
from pytorch_wavelets import DWTForward, DWTInverse
from wavelet_nd import DWTForwardND, DWTInverseND
from Bi_OU_core import run_simulation
from Bi_OU_core_Wavelet import run_simulation_W

@dataclass
class SimulationConfig:
    # --------- Numerical ---------
    n_order: int = 9              # N = 2^n_order = number of space steps along x
    N_steps: int = 512            # number of time steps
    n_ens: int = 1                # ensemble size
    # method: str = 'Wavelet'       # Choice of method :'Fourier' or 'Wavelet'
    method: str = 'Fourier'       # Choice of method :'Fourier' or 'Wavelet'
    wave: str = 'db4'             # choice of the wavelet
    J: int = n_order                    # Number of level of decomposition for wavelet method
    beta: float = 1.              # Energy injected in wavelet spectrum
    mode: str = 'periodization'
    new_spTimeGMC = True          # Wandrille's method for spatiotemporal GMC

    output_type = 0               # 0 (default) : V_x, V_y, V_z, stream_z, vort_z, epsi
                                  # 1 : V_x, V_y
                                  # 2 : stream_z
    normaliz_fields = 0        # Set to False if log-spectrum offset is fitted

    # --------- Physical ---------
    Lx: float = 1                  # domain size
    L_LS: float = Lx         # large spatial scale

    dim: float = 2.0               # spatial dimension
    c2: float = 0.025              # intermittency coefficient (must be < c1 /2, ]0,0.5[ in 2D)
    rho_s: float = 1.66            # spatial KE spectrum slope (]0,3[ in 2D; ]0,2[ in 3D)
    gamma_t: float = rho_s         # temporal KE spectrum slope
    c1 = (rho_s - 1) / 2.          # fractal exponent

    D3 = 1
    T_LS: float = L_LS / D3
    tau_0B: float = T_LS
    # tau_0e_factor: float = np.inf # steady intermittency
    tau_0e_factor: float = 1.0  # unsteady intermittency
    z_eps: float = 1.0

    # --------- Matérn parameters ---------
    nu_m: float = c1
    l_m: float = (np.sqrt(2 * nu_m) * L_LS) / (2 * np.pi)

class Parameters(SimulationConfig): 
    def __init__(self):
        super().__init__()
        if (self.c1 >= 1.0):
            print("ACHTUNG : c1 >=1.0 i.e the velocity field is diffenrentiable.")
            print("This is not realistic for turbulence.")
        #Checking core dynamical parameters (2D)
        if (self.c2 >= np.sqrt(2*self.dim)) or (self.c2 >= self.c1 / 2) or (self.c1 <= 0.0):
            raise ValueError("Parameters must repect : c2 < c1/2 and c1 > 0.0 for 2D simulations. " \
                             + "Here : c1 = " + str(self.c1) + " and c2 = " + str(self.c2))
        else:
            print(f"Using consistent parameters: " \
                  + f"dim = {self.dim}, c1 = {self.c1}, c2 = {self.c2}, rho_s = {self.rho_s}, gamma_t = {self.gamma_t}")

        N = 2 ** self.n_order
        self.N = N
        self.PN = N//2                                 # aliased wavevector index
        self.dx = self.Lx / N

        self.wavelet = self.method == 'Wavelet'
        z = (self.rho_s - 1) / (self.gamma_t - 1)      # scaling slope for time scale of the streamfunction
        rho_streamfct = self.rho_s + 2            # spectrum slope of the omnidirectional spatial spectrum of the streamfunction
        theta_1 = 2 * self.nu_m + self.dim + 2         # spectrum slope of the bidirectionnal spatial spectrum of the streamfunction
        # i.e. theta_1 = rho_s + 2 + self.dim-1        
        K_Lx = 2 * np.pi / self.Lx                     # largest-scale wavenumber
        K_LS = 2 * np.pi / self.L_LS                   # large-scale wavenumber
        self.K_LS = K_LS

        np.random.seed(0)

        if (self.method == 'Wavelet'): # Wavelet method
            print(f"Using wavelet method with wavelet {self.wave} and J={self.J}")
            self.Jmax = np.log2(N)
            if self.dim == 2:
                self.dwt = DWTForward(J=self.J, wave=self.wave, mode=self.mode)
                self.idwt = DWTInverse(wave=self.wave, mode=self.mode)
            elif self.dim == 3: # pytorch_wavelets has no 3D transform
                self.dwt = DWTForwardND(J=self.J, wave=self.wave, mode=self.mode)
                self.idwt = DWTInverseND(wave=self.wave, mode=self.mode)
            else:
                raise ValueError("Incorrect dim")
            if self.J > self.Jmax:
                raise ValueError("ACHTUNG : Requested J exceeds maximum wavelet scale !!!")
            
            # j = np.arange(1, self.J + 1)
            # Wavelet characteristic wavenumbers
            self.j = np.arange(self.J)
            self.k_j = K_Lx * (2.0 ** (self.J - 1 - self.j))  # characteristic wavenumber of each wavelet scale 

            # ---------- OU time scales per scales ----------
            tau_B = self.tau_0B * (1 + (self.k_j / K_LS)**2) ** (-z / 2)

            self.E_epsi_kj = ( self.k_j / K_Lx ) ** (-1) 
            self.E_stream_kj = (1 + (self.k_j / K_LS)**2) ** (-rho_streamfct/2) 

        elif (self.method == 'Fourier'): # Fourier method
            print("Using Fourier method")
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
                Kv = Kv[:,:,:,:,np.newaxis]
                self.KX = KX
                self.KY = KY
                self.KZ = KZ
                self.Kv = Kv
            else:
                raise ValueError("Incorrect dim")
            self.K = K

            # Spectrum
            S1 = 1 / (1 + (K / K_LS)**2 ) ** (theta_1 / 2)
            self.cleanAliasing(S1)
            self.A = np.sqrt(S1)

            # OU timescales
            tau_B = self.tau_0B * (1 + (K / K_LS)**2 ) ** ( - z/2)

            np.seterr(divide = 'ignore') 
            S2 = ( K / K_Lx ) ** (-self.dim) # cf eq (34) of Ruffenach et al. (2026)
            np.seterr(divide = 'warn') 
            self.cleanAliasing(S2)
            self.S2 = S2

            self.lepsi_hat =  self.zeroField()
            self.epsi_t = self.zeroField()
            self.B_hat = self.zeroField()
        
        else:
            raise ValueError("Invalid method. Choose 'Fourier' or 'Wavelet'.")

        tau_e = self.tau_0e_factor * tau_B
        self.Delta_t = min(tau_B.min(), tau_e.min()) * 0.5
        tau_B_filtered = tau_B[~np.isinf(tau_B)]
        self.Big_t = np.max(tau_B_filtered) * 0.5
        print('my two small times are: ', tau_B.min() * 0.5, tau_e.min() * 0.5)
        print("The time step is : dt = " + str(self.Delta_t))

        exp_B = np.exp(-self.Delta_t / tau_B)
        noise_B = np.sqrt(1 - exp_B**2)
        exp_e = np.exp(-self.Delta_t / tau_e)
        noise_e = np.sqrt(1 - exp_e**2)

        if self.dim == 3 and not self.wavelet:
            self.K = self.K[:,:,:,:,np.newaxis]
            self.A = self.A[:,:,:,:,np.newaxis]
            exp_B = exp_B[:,:,:,:,np.newaxis]
            noise_B = noise_B[:,:,:,:,np.newaxis]

        self.c2_eff = self.c2 * np.log( self.L_LS / (self.dx) )
        self.exp_B = exp_B
        self.noise_B = noise_B
        self.exp_e = exp_e
        self.noise_e = noise_e


    def intermittency(self,lepsi_real, intermittency = True):
        if intermittency:
            # Set the correct variance (related to c2)
            lepsi_real /= lepsi_real.std() 
            lepsi_real *= np.sqrt( self.c2_eff )
            lepsi_real -= lepsi_real.var() 
        else:
            lepsi_real = self.zeroSpatialField()
        return np.exp(lepsi_real)

    def CMG_from_noise(self,B_real,epsi, intermittency = True):
        if intermittency:
            Bepsi = B_real * epsi
        else:
            Bepsi = B_real
        return Bepsi

    def FFT(self,B_hat):
        if self.dim == 2:
            return np.fft.fft2(B_hat, axes=(0,1))
        elif self.dim == 3 :
            return np.fft.fftn(B_hat, axes=(0,1,2))
        else:
            raise ValueError("Incorrect dim")

    def IFFT(self,B_hat):
        if self.dim == 2:
            return np.fft.ifft2(B_hat, axes=(0,1)).real
        elif self.dim == 3 :
            return np.fft.ifftn(B_hat, axes=(0,1,2)).real
        else:
            raise ValueError("Incorrect dim")

    
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

    def normalizing(self,V_x, V_y, V_z, stream_z, vort_z):
        if self.normaliz_fields:
            ampli = self.K_LS * np.sqrt((stream_z[:]**2).mean()) # warning : create issues

            stream_z /= ampli
            V_x /= ampli
            V_y /= ampli
            V_z /= ampli
            vort_z /= ampli

    def outputs(self,V_x, V_y, V_z, stream_z, vort_z, epsi):
        match self.output_type:
            case 2:
                return stream_z
            case 1:
                return V_x, V_y
            case 0:
                return V_x, V_y, V_z, stream_z, vort_z, epsi

    def run_simulation(self):
        if not self.wavelet:
            return run_simulation(self)
        else:
            return run_simulation_W(self)
