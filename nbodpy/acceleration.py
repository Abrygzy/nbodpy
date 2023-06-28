import numpy as np

def cic_acc_2d(pars, ng, gravity, h=1):
    '''
    Acceleration of each particle using CIC scheme.
    Input:
        pars: list of Particle objects
        ng: grid size
        gravity_pbc: gravitational potential with periodic boundary conditions (0 cell padding for all axis)
        h: grid spacing
    Output:
        Put acceleration into Particle.acc
    '''
    gravity_pdc = np.pad(gravity, pad_width=1,mode='wrap')
    acc_x_pdc, acc_y_pdc = np.gradient(gravity_pdc, 1, edge_order=1)
    acc_x, acc_y= -1 * acc_x_pdc[1:-1, 1:-1], -1 * acc_y_pdc[1:-1, 1:-1]
    
    for par in pars:
        pos = par.pos
        pos_float = pos / h - 0.5   # floating point index
        pos_floor = np.floor(pos_float).astype(int) # floor of floating point index
        pos_cel = pos_floor + 1 # ceiling of floating point index
        pos_star = pos_float - pos_floor # distance from floor
        par_acc_x, par_acc_y = 0, 0
        for idx_shift in [[x, y] for x in range(2) for y in range(2)]: # Density change: loop over 8 cells
            rho_idx = pos_cel - idx_shift
            rho_idx = np.where(rho_idx != ng, rho_idx, 0)
            par_acc_x += np.multiply.reduce(np.where(idx_shift, 1-pos_star, pos_star)) * acc_x[rho_idx[0], rho_idx[1]] * (h ** 2)
            par_acc_y += np.multiply.reduce(np.where(idx_shift, 1-pos_star, pos_star)) * acc_y[rho_idx[0], rho_idx[1]] * (h ** 2)
        par.acc = np.array([par_acc_x, par_acc_y])
    return acc_x, acc_y

