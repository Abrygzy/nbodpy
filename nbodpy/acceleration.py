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

def accel(Phi,Ng):
    '''
    Description:
      Calculate acceleration field using potential.
    Input: 
        - Phi
          Gravitational potential field given by the poisson solver.
    Return: acc
      Acceleration field.
    '''
    acc = np.zeros((2, Ng, Ng))
    
    # main part
    acc[0][1: Ng - 1]  = (Phi[:Ng-2] - Phi[2:Ng]) / 2
    acc[1][:,1:Ng - 1] = (Phi[:,:Ng-2] - Phi[:,2:Ng]) / 2
    
    acc[0][0]     = (Phi[-1] - Phi[1]) / 2
    acc[1][:,0]   = (Phi[:,-1] - Phi[:,1]) / 2
    
    acc[0][Ng - 1]   = (Phi[Ng-2] - Phi[-1]) / 2
    acc[1][:,Ng - 1] = (Phi[:,Ng-2] - Phi[:,-1]) / 2
    
    return acc

def Force(pos, acc,Ng):
    '''
    Description:
      Calculate specific force for a particle.
    Input: 
        - pos: 
          position of the particle
        - acc:
          acceleration field
    Return:
      F: 3D force
    '''
    F = np.array([0., 0.])
    q, p = int(np.floor(pos[0] - 1/2)), int(np.floor(pos[1] - 1/2))
    qs, ps = pos[0] - 1/2 - q, pos[1] - 1/2 - p
    
    F += acc[:, q % Ng, p % Ng]             * (1 - ps) * (1 - qs) 
    F += acc[:, q % Ng, (p + 1) % Ng]       * ps * (1 - qs)
    F += acc[:, (q + 1) % Ng, p % Ng]       * (1 - ps) * qs
    F += acc[:, (q + 1) % Ng, (p + 1) % Ng] * ps * qs
    
        
    return F
