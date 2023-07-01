import numpy as np
from scipy.fft import fftn, ifftn, fftshift
from astropy import units as u
from astropy import constants as const
import warnings
G = const.G.to(u.Mpc**3/u.Gyr**2/u.M_sun).value
def grav_phi_2d(dens,h=1):
    '''
    Calculate gravitational potential from density field using FFT.
    Input:
        rho: density field
    Output:
        phi: gravitational potential with periodic boundary conditions (0 cell padding for all axis)
    '''

    L = len(dens)
    kx, ky = np.meshgrid(np.fft.fftfreq(L, d=h/L), np.fft.fftfreq(L, d=h/L), indexing='ij')
    g_k = kx**2 + ky**2 
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        phi_k = -4 * np.pi * np.fft.fftn(dens) * G / g_k * L**2 / (2*np.pi)**2
    phi_k[0, 0] = 0
    phi = np.fft.ifftn(phi_k).real
    return phi

def grav_phi_3d(dens,h=1):
    '''
    Calculate gravitational potential from density field using FFT.
    Input:
        dens: density field
    Output:
        phi: gravitational potential with periodic boundary conditions (0 cell padding for all axis)
    '''

    L = len(dens)
    kx, ky, kz = np.meshgrid(np.fft.fftfreq(L, d=h/L), np.fft.fftfreq(L, d=h/L), np.fft.fftfreq(L, d=h/L), indexing='ij')
    g_k = kx**2 + ky**2 + kz**2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        phi_k = -4 * np.pi * np.fft.fftn(dens) * G / g_k * L**3 / (2*np.pi)**3
    phi_k[0, 0, 0] = 0
    phi = np.fft.ifftn(phi_k).real
    return phi