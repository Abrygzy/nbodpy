import numpy as np
from astropy import units as u
from astropy import constants as const

# Gravitational constant in units of Mpc^3/Gyr^2/M_sun
G = const.G.to(u.Mpc**3/u.Gyr**2/u.M_sun).value.astype(np.float32)

# Box size in Mpc
ng = 128

# Cell size in Mpc
h = 1

# frequency in k-space
k_freq = np.fft.fftfreq(ng, d=h/ng).astype(np.float32)
kx_2d, ky_2d = np.meshgrid(k_freq, k_freq, indexing='ij')
g_k_2d = kx_2d**2 + ky_2d**2

kx_3d, ky_3d, kz_3d = np.meshgrid(k_freq, k_freq, k_freq, indexing='ij')
g_k_3d = kx_3d**2 + ky_3d**2 + kz_3d**2