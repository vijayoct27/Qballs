"""
Compute stopping powers for particles incident on a T=0 Fermi sea. 
"""


import numpy as np 
import scipy.integrate as integ 
import scipy.optimize as opt 
import scipy.linalg as la 
import skmonaco as skm 

import kinematics_2plus1 as kin 


MeV_to_InvCM = 5.0*(10**10)


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
        self.KEfermi = self.Efermi - self.m

    def get_weighted_integrand(self, pM, M, Asq_func):
        """
        Construct the integrand function for stopping powers. See 
        notes for derivation.

        This uses an importance-sampling monte carlo integration over
        the target momentum, weighted by p^2. 
        """
        def to_integrate(x):
            """ Stopping power integrand, modulo target momentum weighting """
            pm_mag, pm_costh, scatter_costh = x  
            # boosts
            pm = kin.polarmomentum_to_4vec(self.m, pm_mag, np.arccos(pm_costh))
                # in the lab frame the scattering particles are in the xz plane
            vel_com = kin.get_com_velocity(pM, pm)      
            lab_to_com = kin.boost_matrix(vel_com)
            com_to_lab = kin.boost_matrix(-vel_com)
            pm_com = lab_to_com.dot(pm)
            pM_com = lab_to_com.dot(pM)
            # energy transfer
            pm_final_com = kin.vec4_rotation(pm_com, [0.0, 1.0, 0.0],
                                             np.arccos(scatter_costh))
                # rotate about y-axis, which is perpendicular to the com plane
            pm_final_lab = com_to_lab.dot(pm_final_com)
            omega = pm_final_lab[0] - pm[0] # positive if energy lost by ion
            # Pauli blocking
            blocked = pm_final_lab[0] <= self.Efermi
                # compare total energies here, not just kinetic piece
            if blocked:
                return 0.0
            # kinematics, boost, measure factors
            Ecom = pm_com[0] + pM_com[0]
            pM_mag_com = la.norm(pM_com[1:])
            factor = pM_mag_com/(pm[0]*Ecom)  # removed p_mag**2
            # square amplitude
            Asq = Asq_func(
                *kin.mandelstam_varibales_COM(pM_com, M, pm_com, self.m,
                                              scatter_costh, com=True))
            return factor*omega*Asq
        def weight_func(size):
            """ 
            Returns target momentum samples weighted by p^2 between 0
            and p_fermi, and uniform samples for the cos(angles).
            """
            sample_pts = np.ones((size, 3))*np.nan
            sample_pts[:, 1:3] = 2*np.random.rand(size, 2) - 1    
                # cosine angles uniform in -1 to 1
            sample_pts[:, 0] = self.pfermi*np.random.power(3, size)
                # p distribution: 3*p^2/pfermi^3 between 0 and pfermi
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
        def stopping_power(ke, samples=10**3):
            """
            The stopping power as a function of incident kinetic energy.
            """
            ke = np.asarray(ke, dtype=np.float)
            if not ke.shape:
                ke.resize((1,))  # convert scalar to 1-element array
            sp = np.ones(ke.shape)*np.nan
            sp_err = np.ones(ke.shape)*np.nan
            for index, ke_i in enumerate(ke):                
                pM = kin.polarkinetic_to_4vec(M, ke_i) # in z-direction
                pM_mag = la.norm(pM[1:])
                prefactor = 3*self.n0_m/(64*np.pi*(self.pfermi**3)*pM_mag)
                    # for prefactor derivation, see notes
                prefactor *= self.masstolength 
                    # convert from mass^2 to mass/length
                integrand, dist = self.get_weighted_integrand(pM, M, Asq_func) 
                dist_norm = 4*(self.pfermi**3)/3.0  
                    # norm of sampling distribution
                result, error = skm.mcimport(integrand, npoints=samples,
                                             distribution=dist, nprocs=4,
                                             weight=dist_norm)
                sp[index] = prefactor*result
                sp_err[index] = prefactor*error
            return np.array([sp, sp_err])
        return stopping_power

    def get_azimuthal_weighted_integrand(self, pM, M, Asq_func):
        """
        Includes integration over the com azimuthal scattering angle
        """
        def to_integrate(x):
            """ Stopping power integrand, modulo target momentum weighting """
            pm_mag, pm_costh, scatter_costh, scatter_phi = x  
            # boosts
            pm = kin.polarmomentum_to_4vec(self.m, pm_mag, np.arccos(pm_costh))
                # in the lab frame the scattering particles are in the xz plane
            vel_com = kin.get_com_velocity(pM, pm)      
            lab_to_com = kin.boost_matrix(vel_com)
            com_to_lab = kin.boost_matrix(-vel_com)
            pm_com = lab_to_com.dot(pm)
            pM_com = lab_to_com.dot(pM)
            # energy transfer
            intermediate = kin.vec4_rotation(pm_com, [0.0, 1.0, 0.0],
                                             np.arccos(scatter_costh))
            pm_final_com = kin.vec4_rotation(intermediate, pm_com[1:],
                                             scatter_phi)
                # rotate about y-axis, which is perpendicular to the com plane
            pm_final_lab = com_to_lab.dot(pm_final_com)
            omega = pm_final_lab[0] - pm[0] # positive if energy lost by ion
            # Pauli blocking
            blocked = pm_final_lab[0] <= self.Efermi
                # compare total energies here, not just kinetic piece
            if blocked:
                return 0.0
            # kinematics, boost, measure factors
            Ecom = pm_com[0] + pM_com[0]
            pM_mag_com = la.norm(pM_com[1:])
            factor = pM_mag_com/(pm[0]*Ecom)  # removed p_mag**2
            # square amplitude
            Asq = Asq_func(
                *kin.mandelstam_varibales_COM(pM_com, M, pm_com, self.m,
                                              scatter_costh, com=True))
            return factor*omega*Asq
        def weight_func(size):
            """ 
            Returns target momentum samples weighted by p^2 between 0
            and p_fermi, and uniform samples for the cos(angles).
            """
            sample_pts = np.ones((size, 4))*np.nan
            sample_pts[:, 0] = self.pfermi*np.random.power(3, size)
                # p distribution: 3*p^2/pfermi^3 between 0 and pfermi
            sample_pts[:, 1:3] = 2*np.random.rand(size, 2) - 1    
                # cosine angles uniform in -1 to 1
            sample_pts[:, 3] = 2*np.pi*np.random.rand(size)    
                # azimuthal angle uniform in 0 to 2pi
            return sample_pts
        return to_integrate, weight_func

    def get_azimuthal_stopping_power_func(self, M, Asq_func):
        """ 
        Includes integration over the com azimuthal scattering angle
        """
        def stopping_power(ke, samples=10**3):
            """
            The stopping power as a function of incident kinetic energy.
            """
            ke = np.asarray(ke, dtype=np.float)
            if not ke.shape:
                ke.resize((1,))  # convert scalar to 1-element array
            sp = np.ones(ke.shape)*np.nan
            sp_err = np.ones(ke.shape)*np.nan
            for index, ke_i in enumerate(ke):                
                pM = kin.polarkinetic_to_4vec(M, ke_i) # in z-direction
                pM_mag = la.norm(pM[1:])
                prefactor = 3*self.n0_m/(128*(np.pi**2)*(self.pfermi**3)*pM_mag)
                    # for prefactor derivation, see notes
                prefactor *= self.masstolength 
                    # convert from mass^2 to mass/length
                integrand, dist = (
                    self.get_azimuthal_weighted_integrand(pM, M, Asq_func)) 
                dist_norm = 8*np.pi*(self.pfermi**3)/3.0  
                    # norm of sampling distribution
                result, error = skm.mcimport(integrand, npoints=samples,
                                             distribution=dist, nprocs=4,
                                             weight=dist_norm)
                sp[index] = prefactor*result
                sp_err[index] = prefactor*error
            return np.array([sp, sp_err])
        return stopping_power

    def get_coulomb_stopping_power(self, M, Z, azi=True):
        Asq_func = lambda s, t, u: Asq_coulomb(s, t, u, self.m, self.z, M, Z)
        if azi:
            return self.get_azimuthal_stopping_power_func(M, Asq_func)
        elif ~azi:
            return self.get_stopping_power_func(M, Asq_func)

    def get_thomasfermicoulomb_stopping_power(self, M, Z, alpha=1.0/137.0):
        m_tf = np.sqrt(6*np.pi*alpha*self.n0_m/self.KEfermi)
        Asq_func = lambda s, t, u: Asq_massive_coulomb(s, t, u, self.m, self.z,
                                                       M, Z, m_tf)
        return self.get_stopping_power_func(M, Asq_func)

    def get_compton_stopping_power(self, azi=True):
        Asq_func = lambda s, t, u: Asq_compton(s, t, u, self.m, self.z)
        if azi:
            return self.get_azimuthal_stopping_power_func(0.0, Asq_func)
        elif ~azi:
            return self.get_stopping_power_func(0.0, Asq_func)

    def approx_sp_heavyslow(self, M, Z, alpha=1.0/137.0):
        """ 
        Approximate analytic result for stopping power of a very heavy
        and slow particle incident on a field of relativistic targets.

        This using the approximation that the targets scatter off the
        incident as though the incident is stationary in the lab frame. 
        """
        factor = 1.5*self.n0_m*(self.z**2)*(Z**2)*(alpha**2)/self.pfermi
            # Efermi here is total energy, not just kinetic
        factor *= self.masstolength 
            # convert from mass^2 to mass/length
        angular_integral = 75.0 # Integral over incoming polar angle and 
                                # outgoing solid angle. See notes. 
        adjustment = 3.0/8 # empirically determined bullshit
        def sp_approx(ke):
            return factor*angular_integral*np.sqrt(2*ke/M)*adjustment
        return sp_approx

    def kinetic_cutoff(self, M):
        """ 
        Returns minimum incident particle kinetic energy at which
        energy can be transfered to the targets.  This is a consequence
        of the target motion, and is set by the condition that the 
        incident momentum falls below the fermi momentum. 
        """ 
        return kin.momentum_to_kinetic(self.pfermi, M)

    def approx_sp_heavyfast(self, M, Z, alpha=1.0/137.0):
        """ 
        Approximate analytic result for stopping power of a very heavy
        and fast particle incident on a field of relativistic targets.

        This uses the approximation that the incident scatters targets
        as though the targets were stationary, but have an effective 
        relativistic mass due to their motion. Energy transfers are 
        still limited by a Pauli blocking. 
        """
        scale = 2*np.pi*self.n0_m*(self.z**2)*(Z**2)*(alpha**2)/self.Efermi
            # Efermi here is total energy, not just kinetic
        scale *= self.masstolength 
            # convert from mass^2 to mass/length
        blocked_integrand = lambda w: (
            (1.0/w)*(1.0 - (1.0 + w*(w - 2*self.Efermi)/self.pfermi**2)**1.5))
        unblocked_integrand = lambda w: 1.0/w
        undersea_blocked, undersea_blocked_err = integ.quad(
            blocked_integrand, 0.0, self.KEfermi)
            # under-the-sea contribution for partially Pauli blocked case
        def sp_approx(ke):
            results = np.ones(ke.shape)*np.nan
            errors = np.ones(ke.shape)*np.nan
            for index, ke_i in enumerate(ke):
                q = kin.kinetic_to_momentum(ke_i, M)
                w_kin = kin.maximal_energy_transfer(q, M, self.m)
                if w_kin > self.KEfermi: # partially Pauli blocked
                    unblocked, unblocked_err = integ.quad(
                        unblocked_integrand, self.KEfermi, w_kin)
                    total = unblocked + undersea_blocked
                    total_err = np.sqrt(unblocked_err**2 + 
                                        undersea_blocked_err**2) 
                elif w_kin <= self.KEfermi: # fully Pauli blocked 
                    total, total_err = integ.quad(
                        blocked_integrand, 0.0, w_kin)
                results[index] = scale*total
                errors[index] = scale*total_err
            return results 
        return sp_approx

    def approx_sp_piecewise(self, M, Z, alpha=1.0/137.0):
        """ 
        Approximate analytic result for stopping power, piecewise
        over incoming momentum using separate non-relativistic, 
        relativistic, and cutoff results.
        """
        nonrel_approx = self.approx_sp_heavyslow(M, Z)
        rel_approx = self.approx_sp_heavyfast(M, Z)
        kinetic_threshold = self.kinetic_cutoff(M)
        def sp_approx(ke):
            cutoff = ke < kinetic_threshold
            rel = (ke > M) & (~cutoff) 
            nonrel = (ke <= M) & (~cutoff) 
            results = np.ones(ke.shape)*np.nan 
            pauli_supression = (2*M*ke[cutoff])/(self.pfermi**2)
            results[cutoff] = nonrel_approx(ke[cutoff])*pauli_supression
            results[rel] = rel_approx(ke[rel])
            results[nonrel] = nonrel_approx(ke[nonrel])
            return results
        return sp_approx

    def get_fitting_func(self, M, Z, alpha=1.0/137.0):
        """ 
        Approximate analytic result for stopping power, piecewise
        over incoming momentum using separate non-relativistic, 
        relativistic, and cutoff results.
        """
        nonrel_approx = self.approx_sp_heavyslow(M, Z)
        rel_approx = self.approx_sp_heavyfast(M, Z)
        kinetic_threshold = self.kinetic_cutoff(M)
        def sp_approx(ke):
            cutoff = ke < kinetic_threshold
            rel = (ke > M) & (~cutoff) 
            nonrel = (ke <= M) & (~cutoff) 
            results = np.ones(ke.shape)*np.nan 
            pauli_supression = (2*M*ke[cutoff])/(self.pfermi**2)
            results[cutoff] = nonrel_approx(ke[cutoff])*pauli_supression
            results[nonrel] = nonrel_approx(ke[nonrel])
            results[rel] = rel_approx(ke[rel])
            nonrel_edge = results[nonrel][np.argmax(ke[nonrel])]
            rel_edge = results[rel][np.argmin(ke[rel])]
            results[rel] = np.sqrt(nonrel_edge**2 + # empirical correction
                                  (results[rel] - rel_edge)**2)
            return results
        return sp_approx

    def get_compton_piecewise(self, alpha=1.0/137.0):
        """
        """
        def sp_approx(ke):
            slow = ke < self.KEfermi
            fast = ke >= self.KEfermi
            results = np.ones(ke.shape)*np.nan 
            scale_fast = np.pi*(alpha**2)*self.masstolength 
            results[fast] = scale_fast*self.n0_m*(np.log(2*ke[fast]/self.m))/self.Efermi
            scale_slow = (alpha**2)*self.masstolength 
            results[slow] = scale_slow*self.n0_m*(ke[slow]**2)/(
                            (self.Efermi**2)*self.KEfermi)
            # scale_slow = (alpha**2)*self.masstolength/3.0
            # neff = self.n0_m*(1.0 - (1.0 - ke[slow]/self.KEfermi)**3)
            # results[slow] = scale_slow*ke[slow]*neff/(self.Efermi**2)
            return results
        return sp_approx


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


def Asq_massive_coulomb(s, t, u, m, z, M, Z, m_A, alpha=1.0/137.0):
    """
    Unpolarized amplitude-squared for 2 --> 2 "Coulomb" scattering of
    distinguishable spin-1/2 particles of mass and charge number m, z 
    and M, Z, via exchange of a massive m_A vector mediator.  Given
    as a function of the Mandelstam variables.    
    """
    scale = 32*(np.pi**2)*(alpha**2)*(z**2)*(Z**2)
    msq = m**2 + M**2
    numerator = u**2 + s**2 + 4*t*msq - 2*(msq**2)
    return scale*numerator/((t - m_A**2)**2)


def Asq_compton(s, t, u, m, z, alpha=1.0/137.0):
    """
    Unpolarized amplitude-squared for EM 2 --> 2 Compton scattering
    off spin-1/2 particles of mass m and charge number z. Given
    as a function of the Mandelstam variables.    
    """
    scale = 32*(np.pi**2)*(alpha**2)*(z**2)
    msq = m**2
    s_m2 = s - msq  # = 2 p \cdot k, for photon 4-vec k and target p
    u_m2 = u - msq  # = -2 p \cdot k', for final photon k'
    term = msq*(1.0/s_m2 + 1.0/u_m2)
    return scale*(4*term*(term + 1) - u_m2/s_m2 - s_m2/u_m2)