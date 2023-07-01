import numpy as np
from numba import jit

def acc_mesh(phi,h=1):
    ng = phi.shape[0]
    phi_pdc = np.pad(phi, pad_width=1,mode='wrap')
    acc_x_pdc, acc_y_pdc = np.gradient(phi_pdc, 1, edge_order=1)
    acc_x, acc_y= -1 * acc_x_pdc[1:-1, 1:-1], -1 * acc_y_pdc[1:-1, 1:-1]
    return acc_x, acc_y

@jit(nopython=True) 
def acc_par(pars_pos, acc_x, acc_y, h=1):
    par_acc_x, par_acc_y = np.zeros(len(pars_pos)), np.zeros(len(pars_pos))
    ng = acc_x.shape[0]
    pos_float = pars_pos/h - 0.5 # particle positions in float
    pos_floor = np.floor(pos_float).astype(np.int16) # particle positions in int
    pos_cel = pos_floor + 1 # ceiling of floating point index
    pos_star = pos_float - pos_floor # distance from floor
    for i in range(len(pars_pos)):
        for idx_shift in np.array([[0, 0], [0, 1], [1, 0], [1, 1]]): # Density change: loop over 4 cells
            rho_idx = pos_cel[i] - idx_shift
            rho_idx[rho_idx == ng] = 0    
            par_acc_x[i] +=  [pos_star[i], 1-pos_star[i]][idx_shift[0]][0] * [pos_star[i], 1-pos_star[i]][idx_shift[1]][1] * acc_x[rho_idx[0], rho_idx[1]] * (h ** 2)
            par_acc_y[i] +=  [pos_star[i], 1-pos_star[i]][idx_shift[0]][0] * [pos_star[i], 1-pos_star[i]][idx_shift[1]][1] * acc_y[rho_idx[0], rho_idx[1]] * (h ** 2)
    return par_acc_x, par_acc_y