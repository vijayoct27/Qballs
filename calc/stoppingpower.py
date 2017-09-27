"""
Compute stopping powers for particles incident on a T=0 Fermi sea. 
"""


import numpy as np 
import scipy.integrate as integ 
import scipy.optimize as opt 
import scipy.linalg as la 
import skmonaco as skm 

import kinematics_2plus1 as kin 


MeV_to_InvCM = 8065.54*(10**6)


class FermiSea_StoppingPower(object):
    """
    Fixes the properties of a T=0 Fermi sea and gives functions to
    compute the stopping powers in the sea for incident particles.
    """
    def __init__(self, m=1.0, n0=None, z=-1.0, masstolength=1.0,
                 massunit=None, lengthunit=None, label=None):
        """ 
        Args:
        m - float > 0, default=1
            The mass of the target sea particles. m=1 will set the
            energy unit for all subsequent calculations. 
        n0 - float > 0 
            Number density of sea particles. Assumed to be given in
            units of length^{-3}, set by masstolength. 
        z - float
            Charge of sea particles (in units of elementary charge)
        masstolength - float > 0, default=1 
            Unit conversion factor: (quantity in units of 1/length) = 
            masstolength*(quantity in units mass).  Default is to work
            in natural units.  In practice this is used only to convert
            an input density to natural units, and then convert the
            final result to units (energy/distance)
        massunit - string
            Name of this mass unit used, for display purposes
        lengthunit - string
            Name of this length unit used, for display purposes
        label - string
            Name of this calculation
        """
        self.m = float(m)
        self.z = float(z)
        self.masstolength = float(masstolength)
        self.n0_m = n0/(self.masstolength**3)  # number density in mass^3
        self.label = str(label)
        self.units = {"mass":str(massunit),
                      "length":str(lengthunit)}
        self.pfermi = (3*(np.pi**2)*self.n0_m)**(1.0/3.0)
        self.Efermi = np.sqrt(self.m**2 + self.pfermi**2)

    def get_stopping_power_integrand(self, pM, M, Asq_func):
        """
        Construct the integrand function for stopping powers. See 
        notes for derivation.
        """
        def to_integrate(x):
            pm_mag, pm_costh, scatter_costh = x  
            # boosts
            pm = kin.polarmomentum_to_4vec(pm_mag, np.arccos(pm_costh), self.m)
            vel_com = kin.get_com_velocity(pM, pm)      
            lab_to_com, com_to_lab = kin.boost_and_inverse_matricies(vel_com)
            rotate = kin.rot_matrix(np.arccos(scatter_costh))
            pm_com = lab_to_com.dot(pm)
            pM_com = lab_to_com.dot(pM)
            # energy transfer
            pm_final_lab = com_to_lab.dot(rotate.dot(pm_com))
            omega = pm_final_lab[0] - pm[0] # positive if energy lost by ion
            # Pauli blocking
            blocked = pm_final_lab[0] <= self.Efermi
            if blocked:
                return 0.0
            # kinematics, boost, measure factors
            Ecom = pm_com[0] + pM_com[0]
            pM_mag_com = la.norm(pM_com[1:3])
            factor = (pm_mag**2)*pM_mag_com/(pm[0]*Ecom)
            # square amplitude
            Asq = Asq_func(
                *kin.mandelstam_varibales_COM(pM_com, M, pm_com, self.m,
                                              scatter_costh, com=True))
            return factor*omega*Asq
        return to_integrate

    def get_weighted_integrand(self, pM, M, Asq_func):
        """
        Construct the integrand function for stopping powers. See 
        notes for derivation.
        """
        def to_integrate(x):
            pm_mag, pm_costh, scatter_costh = x  
            # boosts
            pm = kin.polarmomentum_to_4vec(pm_mag, np.arccos(pm_costh), self.m)
            vel_com = kin.get_com_velocity(pM, pm)      
            lab_to_com, com_to_lab = kin.boost_and_inverse_matricies(vel_com)
            rotate = kin.rot_matrix(np.arccos(scatter_costh))
            pm_com = lab_to_com.dot(pm)
            pM_com = lab_to_com.dot(pM)
            # energy transfer
            pm_final_lab = com_to_lab.dot(rotate.dot(pm_com))
            omega = pm_final_lab[0] - pm[0] # positive if energy lost by ion
            # Pauli blocking
            blocked = pm_final_lab[0] <= self.Efermi
            if blocked:
                return 0.0
            # kinematics, boost, measure factors
            Ecom = pm_com[0] + pM_com[0]
            pM_mag_com = la.norm(pM_com[1:3])
            factor = pM_mag_com/(pm[0]*Ecom)  # removed p_mag**2
            # square amplitude
            Asq = Asq_func(
                *kin.mandelstam_varibales_COM(pM_com, M, pm_com, self.m,
                                              scatter_costh, com=True))
            return factor*omega*Asq
        norm = (self.pfermi**3)/3.0     
        def weight_func(size):
            sample_pts = np.ones((size, 3))*np.nan
            sample_pts[:, 1:3] = 2*np.random.rand(size, 2) - 1    
                # cosine angles uniform in -1 to 1
            sample_pts[:, 0] = self.pfermi*np.random.power(3, size)
                # p distribution: 3*p^2/pfermi^3
            return sample_pts
        return to_integrate, weight_func

    def get_stopping_power_func(self, M, Asq_func):
        """ 
        Return a function that computes the contribution to the stopping
        power for incident particle due to a specific interaction as 
        a function of incident kinetic energy. 
        Args:
        ke - 1D vector, floats > 0
            Kinetic energy of incident particle (units of target mass)
        M - float > 0
            Mass of incident particle (units of target mass)
        Asq_func - func(s, t, u)
            Invariant amplitude-squared for the interaction 
        samples
        """
        def stopping_power(ke, samples=10**3, important=False):
            """
            The stopping power as a function of incident kinetic energy.
            """
            ke = np.asarray(ke, dtype=np.float)
            if not ke.shape:
                ke.resize((1,))  # convert scalar to 1-element array
            theta_xdir = 0.0  # incident particle moving in x direction
            sp = np.ones(ke.shape)*np.nan
            sp_err = np.ones(ke.shape)*np.nan
            for index, ke_i in enumerate(ke):                
                pM = kin.polarkinetic_to_4vec(ke_i, theta_xdir, M)
                pM_mag = la.norm(pM[1:3])
                prefactor = 3*self.n0_m/(64*np.pi*(self.pfermi**3)*pM_mag)
                    # for prefactor derivation, see notes
                prefactor *= self.masstolength 
                    # convert from mass^2 to mass/length
                if important:
                    base_integrand, dist = self.get_weighted_integrand(pM, M, Asq_func) 
                    result, error = skm.mcimport(base_integrand, npoints=samples,
                                                 distribution=dist, nprocs=4,
                                                 weight=4*(self.pfermi**3)/3.0)
                else:
                    integrand = self.get_stopping_power_integrand(pM, M, Asq_func) 
                    result, error = skm.mcmiser(integrand, npoints=samples, 
                                               xl = [0.0, -1.0, -1.0], 
                                               xu = [self.pfermi, 1.0, 1.0],
                                               nprocs=4)
                sp[index] = prefactor*result
                sp_err[index] = prefactor*error
            return np.array([sp, sp_err])
        return stopping_power

    def get_ion_coulomb_stopping_power(self, M, Z):
        Asq_func = lambda s, t, u: Asq_coulomb(s, t, u, self.m, self.z, M, Z)
        return self.get_stopping_power_func(M, Asq_func)

    def high_density_slow_ion(self, M, Z, alpha=1.0/137.0):
        factor = 4*self.n0_m*(self.z**2)*(Z**2)*(alpha**2)/self.Efermi
        factor *= self.masstolength 
            # convert from mass^2 to mass/length
        angular_integral = 2*np.pi  # need a real integral here, see notes
        fuck_it = 0.008 # empirically determined bullshit 
        def le_limit(ke):
            return factor*angular_integral*fuck_it*np.sqrt(2*ke/M)
        return le_limit

    def high_density_fast_ion(self, M, Z, alpha=1.0/137.0):
        factor = 2*np.pi*self.n0_m*(self.z**2)*(Z**2)*(alpha**2)/self.Efermi
        factor *= self.masstolength 
            # convert from mass^2 to mass/length
        coulomb_log = 10.0  # need a real integral here, see notes
        def he_limit(ke):
            return factor*coulomb_log*np.ones(ke.shape)
        return he_limit

    def high_density_limiting(self, M, Z, alpha=1.0/137.0):
        """
        For high densities, where the electron Fermi energy is greater
        than it's mass, this returns a quadrature sum of the low 
        ion energy (k^{1/2}) and the high ion energy {p->0, k^0} limits.
        """
        le = self.high_density_slow_ion(M, Z, alpha=1.0/137.0)
        he = self.high_density_fast_ion(M, Z, alpha=1.0/137.0)
        return lambda ke: (le(ke)**(-2) + he(ke)**(-2))**(-0.5)


def Asq_coulomb(s, t, u, m, z, M, Z, alpha=1.0/137.0):
    """
    Unpolarized amplitude-squared for 2 --> 2 Coulomb scattering of
    distinguishable spin-1/2 particles of mass and charge number m, z 
    and M, Z.  Given as a function of the Mandelstam variables.    
    """
    scale = 32*(np.pi**2)*(alpha**2)*(z**2)*(Z**2)
    msq = m**2 + M**2
    numerator = u**2 + s**2 + 4*t*msq - 2*(msq**2)
    return scale*numerator/(t**2)
