import numpy as np
from scipy.fft import fftn, ifftn, fftshift
import warnings
G = 100
def grav_from_den_2d(dens):
    '''
    Calculate gravitational potential from density field using FFT.
    Input:
        rho: density field
    Output:
        phi: gravitational potential with periodic boundary conditions (0 cell padding for all axis)
    '''

    L = len(dens)
    kx, ky = np.meshgrid(np.fft.fftfreq(L, d=1/L), np.fft.fftfreq(L, d=1/L), indexing='ij')
    g_k = kx**2 + ky**2 
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        phi_k = -4 * np.pi * np.fft.fftn(dens) * G / g_k * L**2 / (2*np.pi)**2
    phi_k[0, 0] = 0
    phi = np.fft.ifftn(phi_k).real
    return phi

def poisson_solver_fft2(dens):
    '''
    Description: solve 2D Poisson equation with FFT.
    Input: 
       dens: density field.
    Return: 
       Phi: gravitational potential field.
    '''
    L = len(dens)
    g_k = np.concatenate((np.mgrid[:L//2,:L][0], np.mgrid[:L//2,:L][0] - L/2), axis=0) ** 2 + \
          np.concatenate((np.mgrid[:L,:L//2][1], np.mgrid[:L,:L//2][1] - L/2), axis=1) ** 2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Phi_k = - fftn(dens) / g_k * G * L**2 / np.pi
    Phi_k[0][0] = 0

    Phi = ifftn(Phi_k).real
    
    return Phi
