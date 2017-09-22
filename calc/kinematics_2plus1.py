""" Utilities for relativistic 2D kinematics """


import numpy as np 
import scipy.linalg as la 


def lorentz_dot(v1, v2):
    """ 2+1 Lorentz dot product, signature (1, -1, -1) """
    return v1[0]*v2[0] - v1[1:3].dot(v2[1:3])


def lorentz_gamma(v):
    """ Lorentz gamma factor for the 2D velocity vector v """
    speed = np.sqrt(v.dot(v))
    return 1.0/np.sqrt(1.0 - speed**2)


def cartmomentum_to_4vec(pcart, m):
    """ 
    convert 2D Cartesian spacial momentum pcart of a particle with
    mass m to the on-shell 2+1 4-momentum.
    """
    pmag = la.norm(pcart)
    E = np.sqrt(pmag**2 + m**2)
    return np.array([E, pcart[0], pcart[1]])    


def polarmomentum_to_4vec(pmag, ptheta, m):
    """ 
    Convert the 2D spacial momentum with magnitude pmag and angle
    with the x-axis ptheta of a particle with mass m to the on-shell
    2+1 4-momentum.
    """
    pcart = pmag*np.array([np.cos(ptheta), np.sin(ptheta)])
    return cartmomentum_to_4vec(pcart, m)


def kinetic_to_momentum(ke, m):
    """
    Return the momentum for a particle of mass m with kinetic energy ke
    """
    return np.sqrt(2*m*ke + ke**2)


def polarkinetic_to_4vec(ke, theta, m):
    """
    Convert the kinetic energy and angle of the 2D spacial momentum
    with the x-axis for a particle with mass m to the on-shell
    2+1 4-momentum.
    """    
    pmag = kinetic_to_momentum(ke, m)
    return polarmomentum_to_4vec(pmag, theta, m)


def boost_matrix(v):
    """ 2+1 Lorentz boost to a frame moving with 2D velocity v """
    g = lorentz_gamma(v)
    vx, vy = v
    vsq = v.dot(v)
    return np.array(
        [[g,       -g*vx,                     -g*vy],
         [-g*vx,   1 + (g - 1)*(vx**2)/vsq,   (g - 1)*(vx*vy)/vsq],
         [-g*vy,   (g - 1)*(vx*vy)/vsq,       1 + (g - 1)*(vy**2)/vsq]])


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


def mandelstam_varibales(pM, M, pm, m, costh):
    """
    For two particles with masses M, m and incident 2+1 momenta pM, pm
    scattering elastically through costh (cosine of the CoM scattering
    angle), returns the mandelstam variables s, t, u, where
        s = (pm + pM)^2                # total CoM energy squared  
        t = (pm - pm_aftercollison)^2  # 4-momentum transfer squared
        u = (pM - pm_aftercollison)^2  
    and energy/momentum conservation implies s + t + u = 2(m^2 + M^2).   
    """
    if is_com(pM, pm):
        s = (pM[0] + pm[0])**2
        t = 2*(la.norm(pm[1:3])**2)*(costh - 1.0)
        u = 2*(m**2 + M**2) - s - t 
        return s, t, u
    # Splitting calculation into CoM and non-CoM steps saves runtime
    # if this function is called in the CoM frame, but increases run
    # time for non-CoM calls.  This is typically an improvement as most
    # calculations will require making a CoM boost elsewhere.    
    else:
        boost_to_com = boost_matrix(get_com_velocity(pM, pm))
        pm_com = boost_to_com.dot(pm)
        pM_com = boost_to_com.dot(pM)
        return mandelstam_varibales(pM_com, M, pm_com, m, costh)