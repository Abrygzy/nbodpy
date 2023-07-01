import numpy as np
import pandas as pd

class Particle:
    def __init__(self, pos, vel, mass,id):
        self.pos = pos.astype(np.float32) # position_x，position_y，position_z, 
        self.vel = vel.astype(np.float32) # velocity_x, velocity_y, velocity_z,
        self.acc = np.zeros_like(pos).astype(np.float32) # acceleration_z, acceleration_y, acceleration_x
        self.mass = mass.astype(np.float32)
        self.id = id.astype(int)
    def periodic(self, ng):
        '''
        Periodic boundary conditions.
        '''
        self.pos = np.where(self.pos < 0, self.pos%ng, self.pos)
        self.pos = np.where(self.pos >= ng, self.pos%ng, self.pos)

def par_create_2d(pars_df):
    '''
    Create particles from pandas DataFrame.
    Input:
        pars_df: pandas DataFrame with columns ['pos_x', 'pos_y', 'vel_x', 'vel_y', 'mass']
    Output:
        par_list: list of Particle objects
    '''
    # pars_pos = pars_df.loc[:, ['pos_y','pos_x']].values
    # pars_vel = pars_df.loc[:, ['vel_y', 'vel_x']].values
    pars_pos = pars_df.loc[:, ['pos_x','pos_y']].values
    pars_vel = pars_df.loc[:, ['vel_x','vel_y']].values
    pars_mass = pars_df.loc[:, 'mass'].values
    pars_id = pars_df.loc[:, 'id'].values
    par_list = [Particle(pos, vel, mass, id) for pos, vel, mass, id in zip(pars_pos, pars_vel, pars_mass,pars_id)]
    return par_list

def par_create_3d(pars_df):
    '''
    Create particles from pandas DataFrame.
    Input:
        pars_df: pandas DataFrame with columns ['pos_x', 'pos_y', 'pos_z', 'vel_x', 'vel_y', 'vel_z', 'mass']
    Output:
        par_list: list of Particle objects
    '''
    pars_pos = pars_df.loc[:, ['pos_x', 'pos_y', 'pos_z']].values
    pars_vel = pars_df.loc[:, ['vel_x', 'vel_y', 'vel_z']].values
    pars_mass = pars_df.loc[:, 'mass'].values
    pars_id = pars_df.loc[:, 'id'].values
    par_list = [Particle(pos, vel, mass, id) for pos, vel, mass, id in zip(pars_pos, pars_vel, pars_mass, pars_id)]
    return par_list

def par_to_array(par_list):
    '''
    Convert list of Particle objects to numpy array.
    Input:
        par_list: list of Particle objects
    Output:
        pars: numpy array with columns ['pos_y', 'pos_x', 'vel_x', 'vel_y', 'mass']
    '''
    par_array = np.zeros((len(par_list), 5))
    for i, par in enumerate(par_list):
        par_array[i, :] = np.array([par.pos[0], par.pos[1], par.vel[0], par.vel[1], par.mass])
    return par_array