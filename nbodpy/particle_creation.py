import numpy as np
import pandas as pd

class Particle:
    def __init__(self, pos, vel, mass):
        self.pos = pos
        self.vel = vel
        self.acc = np.zeros_like(pos)
        self.mass = mass

def par_create_2d(pars_df):
    '''
    Create particles from pandas DataFrame.
    Input:
        pars_df: pandas DataFrame with columns ['pos_x', 'pos_y', 'vel_x', 'vel_y', 'mass']
    Output:
        par_list: list of Particle objects
    '''
    pars_pos = pars_df.loc[:, ['pos_x', 'pos_y']].values
    pars_vel = pars_df.loc[:, ['vel_x', 'vel_y']].values
    pars_mass = pars_df.loc[:, 'mass'].values
    par_list = [Particle(pos, vel, mass) for pos, vel, mass in zip(pars_pos, pars_vel, pars_mass)]
    return par_list