import numpy as np
import warnings
from paras import G, ng, h, g_k_2d, g_k_3d


def grav_phi_2d(dens):
    '''
    Calculate gravitational potential from density field using FFT.
    Input:
        dens: density field
    Output:
        phi: gravitational potential
    '''

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # phi_k = -4 * np.pi * np.fft.fftn(dens) * G / g_k * L**2 / (2*np.pi)**2
        phi_k = -1 * np.fft.fftn(dens) * G / g_k_2d * ng**2 / np.pi
    phi_k[0, 0] = 0
    phi = np.fft.ifftn(phi_k).real
    return phi

def grav_phi_3d(dens):
    '''
    Calculate gravitational potential from density field using FFT.
    Input:
        dens: density field
    Output:
        phi: gravitational potential
    '''

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # phi_k = -4 * np.pi * np.fft.fftn(dens) * G / g_k * L**3 / (2*np.pi)**3
        phi_k = -1 * np.pi * np.fft.fftn(dens) * G / g_k_3d * ng**3 / 2 / np.pi**2
    phi_k[0, 0, 0] = 0
    phi = np.fft.ifftn(phi_k).real
    return phi