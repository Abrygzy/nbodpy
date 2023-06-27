import numpy as np
import warnings

def grav_from_den_2d(dens):
    '''
    Calculate gravitational potential from density field using FFT.
    Input:
        rho: density field
    Output:
        phi: gravitational potential with periodic boundary conditions (0 cell padding for all axis)
    '''
    G = 1
    L = len(dens)
    kx, ky = np.meshgrid(np.fft.fftfreq(L, d=1/L), np.fft.fftfreq(L, d=1/L), indexing='ij')
    g_k = kx**2 + ky**2 
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        phi_k = -4 * np.pi * np.fft.fftn(dens) * G / g_k * L**2 / (2*np.pi)**2
    phi_k[0, 0] = 0
    phi = np.fft.ifftn(phi_k).real
    return phi


