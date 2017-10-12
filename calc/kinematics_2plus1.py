""" Utilities for relativistic 2D kinematics """


import numpy as np 
import scipy.linalg as la 


def polarmomentum_to_4vec(m, mag, theta=0.0, phi=0.0):
    """ 
    Convert the 3D spacial momentum with magnitude mag and spherical
    angles theta, phi of a particle with mass m to the corresponding
    on-shell 4-momentum. Theta and phi default to zero (z-direction).
    """
    cart = mag*np.array([np.sin(theta)*np.cos(phi),
                         np.sin(theta)*np.sin(phi),
                         np.cos(theta)])
    E = np.sqrt(mag**2 + m**2)
    return np.concatenate(([E], cart))    


def polarkinetic_to_4vec(m, ke, theta=0.0, phi=0.0):
    """ 
    Convert the kinetic energy ke in a direction given by spherical
    angles theta, phi of a particle with mass m to the corresponding
    on-shell 4-momentum. Theta and phi default to zero (z-direction).
    """
    mag = np.sqrt(ke*(ke + 2*m))
    return polarmomentum_to_4vec(m, mag, theta, phi)


def momentum_to_kinetic(p, m):
    """ 
    Given the momentum p of a particle of mass m, returns its kinetic energy
    """
    # This is going to susceptible to numerical errors if p is much
    # smaller than m due to a subtraction of nearly equal numbers. 
    # Perhaps add non-relativistic case to catch that?
    return np.sqrt(p**2 + m**2) - m


def kinetic_to_momentum(ke, m):
    """ 
    Given the kinetic energy ke of a particle of mass m, returns its momentum
    """
    return np.sqrt(ke**2 + 2*ke*m)


def boost_matrix(v):
    """ 3+1 Lorentz boost to a frame moving with 3D velocity v """
    vsq = v.dot(v)
    g = 1.0/np.sqrt(1.0 - vsq)  # Lorentz gamma 
    # matrix elements
    m01, m02, m03 = -g*v
    mdia = (g - 1)/vsq
    m12 =  mdia*v[0]*v[1]
    m13 =  mdia*v[2]*v[0]
    m23 =  mdia*v[2]*v[1]
    # forward boost
    boost = np.array(
        [[g,     m01,                 m02,                 m03],
         [m01,   1 + mdia*(v[0]**2),  m12,                 m13],
         [m02,   m12,                 1 + mdia*(v[1]**2),  m23],
         [m03,   m13,                 m23,                 1 + mdia*(v[2]**2)]])
    return boost


def general_rotation(v, n, phi):
    """ 
    Rotate the 3D vector v by phi around the direction of 3D
    vector n.  Implemented using the Rodrigues Formula. 
    """
    v = np.asarray(v, dtype=float)
    n = np.asarray(n, dtype=float)
    n = n/la.norm(n)
    return (v*np.cos(phi) + 
            np.cross(n, v)*np.sin(phi) + 
            n*np.dot(n, v)*(1.0 - np.cos(phi)))


def vec4_rotation(v4, n3, phi):
    """
    Rotate the spacial components of the 4-vector v4 by phi
    around the direction given by the 3-vector n4.
    """
    new = np.ones(v4.shape)*np.nan
    new[0] = v4[0]
    new[1:] = general_rotation(v4[1:], n3, phi)
    return new


def get_com_velocity(q, p):
    """
    Return the velocity of the CoM frame for d+1 Lorentz momenta q, p
    """
    total = q + p 
    return total[1:]/total[0]


def is_com(q, p):
    """ Check if 2+1 momenta q,p have stationary center of momentum """
    v_com = get_com_velocity(q, p)
    speed_com = la.norm(v_com)  
        # check speed instead of total momentum so the
        # comparison is done with O(1) numbers 
    return np.isclose(speed_com, 0.0)


def mandelstam_varibales_COM(pM, M, pm, m, costh, com=False):
    """
    For two particles with masses M, m and incident 2+1 momenta pM, pm
    scattering elastically through costh (cosine of the CoM scattering
    angle), returns the mandelstam variables s, t, u, where
        s = (pm + pM)^2                # total CoM energy squared  
        t = (pm - pm_aftercollison)^2  # 4-momentum transfer squared
        u = (pM - pm_aftercollison)^2  
    and energy/momentum conservation implies s + t + u = 2(m^2 + M^2).   
    """
    if com or is_com(pM, pm):
        s = (pM[0] + pm[0])**2
        t = 2*(la.norm(pm[1:])**2)*(costh - 1.0)
        u = 2*(m**2 + M**2) - s - t 
        return s, t, u   
    else:
        raise Exception("vectors are not in the CoM frame")


def maximal_energy_transfer(q_in, m_in, m_tar):
    """
    For an incident particle with momentum q_in and mass m_in striking
    a stationary target with mass m_tar, this gives the maximal
    kinematically allowed energy transfer to the stationary particle. 
    """
    E = np.sqrt(q_in**2 + m_in**2)
    classical = 2*m_tar*(q_in**2)/(m_in**2)
    denominator = 1.0 + 2*m_tar*E/(m_in**2) + (m_tar**2)/(m_in**2)
    return classical/denominator