
import h5py
import numpy as np
import pandas as pd
import sys,os
import pylab
from scipy import stats,signal,io
import matplotlib.pyplot as plt

import pickle
import seaborn as sns
sns.set_style('white')

if __name__ == "__main__":

    head_data = h5py.File('all_head_data.hdf5','r')
    print head_data.keys()
    dx = head_data['dx'][:]
    dy = head_data['dy'][:]
    dz = head_data['dz'][:]
    oz = head_data['oz'][:]
    oy = head_data['oy'][:]
    ox = head_data['ox'][:]
    ay = head_data['ay'][:]
    ax = head_data['ax'][:]
    az = head_data['az'][:]

    head_data.close()

    xyz = np.sqrt(ax**2 + ay**2 + az**2)

    theta = np.rad2deg(np.arctan(ax/ay))

    head_variables = np.vstack([ox,oy,oz,dx,dy,dz,ax,ay,az,xyz,theta])

    head_names = ['yaw','roll','pitch','d_yaw','d_roll','d_pitch','x acc','y acc','z acc','xyz','theta']

    time_axis = np.arange(0,ox.shape[0]*.1,.1)


    num_plots = len(head_names)
    f, axarr = plt.subplots(num_plots, sharex=True,dpi=600)

    #stop = 1000

    for i in range(num_plots):
        axarr[i].plot(time_axis,head_variables[i,:],lw=.1,color='black')

        axarr[i].set_ylabel(head_names[i],rotation = 0)
        #axarr[i].set_yticks(rotation = 90)

    axarr[-1].set_xlabel('Time (sec)')
    sns.despine(left=True,bottom=True)


    f.savefig('./plots/head_data.pdf')
