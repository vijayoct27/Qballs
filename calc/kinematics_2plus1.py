""" Utilities for relativistic 2D kinematics """


import numpy as np 
import scipy.linalg as la 


def polarmomentum_to_4vec(pmag, ptheta, m):
    """ 
    Convert the 2D spacial momentum with magnitude pmag and angle
    with the x-axis ptheta of a particle with mass m to corresponding
    the on-shell 2+1 momentum.
    """
    pcart = pmag*np.array([np.cos(ptheta), np.sin(ptheta)])
    E = np.sqrt(pmag**2 + m**2)
    return np.array([E, pcart[0], pcart[1]])    


def polarkinetic_to_4vec(ke, ptheta, m):
    """ 
    Convert the kinetic energy ke in a direction that makes an angle
    ptheta with the x-axis for a particle with mass m to the 
    corresponding on-shell 2+1 momentum.
    """
    pmag = np.sqrt(ke*(ke + 2*m))
    return polarmomentum_to_4vec(pmag, ptheta, m)


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


def boost_and_inverse_matricies(v):
    """ 2+1 Lorentz boost to a frame moving with 2D velocity v """
    vx, vy = v
    vx_sq, vy_sq, vxvy = vx**2, vy**2, vx*vy
    vsq = v.dot(v)
    g = 1.0/np.sqrt(1.0 - vsq)  # Lorentz gamma 
    # matrix elements
    m01 =  -g*vx
    m02 =  -g*vy
    m12 =  (g - 1)*vxvy/vsq
    mdia = (g - 1)/vsq
    # forward boost
    boost = np.array(
        [[g,     m01,             m02],
         [m01,   1 + mdia*vx_sq,  m12],
         [m02,   m12,             1 + mdia*vy_sq]])
    # backward boost (v -> -v)
    inverse = np.array(
        [[g,     -m01,             -m02],
         [-m01,   1 + mdia*vx_sq,  m12],
         [-m02,   m12,             1 + mdia*vy_sq]])
    return boost, inverse


def rot_matrix(phi):
    """ 2+1 xy spacial rotation about z, counterclockwise by phi """
    return np.array([[1,  0,             0],
                     [0,  np.cos(phi),  -np.sin(phi)],
                     [0,  np.sin(phi),   np.cos(phi)]])


def get_com_velocity(q, p):
    """
    Return the 2D velocity of the CoM frame for 2+1 4-momenta q, p
    """
    total = q + p 
    return total[1:3]/total[0]


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
        t = 2*(la.norm(pm[1:3])**2)*(costh - 1.0)
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