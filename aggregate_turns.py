import numpy as np
import pandas as pd
import sys,os
import pylab
from scipy import stats,signal,io
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as patches
from matplotlib import cm
from sklearn import linear_model
import pickle
import seaborn as sns
sns.set_style('white')
import h5py
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split
import sklearn.linear_model
from sklearn import metrics
from skimage import exposure
from sklearn.linear_model import LogisticRegression,SGDClassifier
plt.rcParams['pdf.fonttype'] = 'truetype'

def get_experiment_names(all_files,rat):
    exp_names = []
    for fil in all_files:
        for exp in os.listdir('./' + fil + '/'):
            if exp.startswith('%s_0' % rat):
                exp_names.append(exp[exp.find('m_')+2:exp.find('.txt')]) ### m_ is from the am_ or pm_ which ends the experiment timestamp
                
    return np.asarray(exp_names)


def concatenate_dxs(all_files,dx_type):
    ## dx_type = dx, dy, or dz
    dxs = []
 
    for fil in all_files:
        dx_fil = np.load('./' + fil + '/%s.npz' % dx_type)       
        

        dx_fil['']

        dxs.append(dx_fil)
        
        dx_fil.close()
  
        ## keys for each = ['y_left', 'y_right', 'X_left', 'X_right']

    return dxs

def get_condition_indexes(exp_names,behavior_conditions):
    indexes = {}
    for condition in behavior_conditions:
        indexes[condition] = np.where(exp_names == condition)[0]
  
    return indexes

def get_all_turns(all_files,dx_types):
    
    all_turns = []
    
    for dx_type in dx_types:
        dxs = []
        #print dx_type
        for fil in all_files:
            #print fil
            dx_fil = np.load('./' + fil + '/%s.npz' % dx_type)
            
            x = [np.asarray(dx_fil[key]) for key in dx_fil.keys()] 
            #print('len(x) = ', len(x))
            dxs.append(x)

            dx_fil.close()
        #print('len(dxs) = ', len(dxs))
        
        all_turns.append(np.asarray(dxs))
    
    return np.dstack(all_turns)

def make_turn_dict(all_turns,exp_names,dx_types,dx_keys):

    num_files = all_turns.shape[0]

    all_turn_data = {}

    ##### set up the dictionaries:
    for file_num in range(num_files):
        all_turn_data[exp_names[file_num]] = {}
        for turn_type in range(3): ### dx, dy, dz
            all_turn_data[exp_names[file_num]][dx_types[turn_type]] = {}
            for i in range(4): ## ['y_left', 'y_right', 'X_left', 'X_right']
                all_turn_data[exp_names[file_num]][dx_types[turn_type]][dx_keys[i]] = []
                #print exp_names[file_num],dx_types[turn_type], dx_keys[i], all_turns[file_num,i,turn_type].shape # all_data['dark'][turn_names[]] 

    ##### fill in the dictionaries:
    for file_num in range(num_files):
        
        for turn_type in range(3): ### dx, dy, dz
            
            for i in range(4): ## ['y_left', 'y_right', 'X_left', 'X_right']
                all_turn_data[exp_names[file_num]][dx_types[turn_type]][dx_keys[i]].append(all_turns[file_num,i,turn_type])          


    return all_turn_data


def plot_single_tetrode(all_turn_data,behavior_condition,dx_type,dx_key,tetrode):
    
    f = plt.figure(dpi=600)
    
    f.suptitle([str(behavior_condition),str(dx_type),str(dx_key), 
                'Tetrode %d, %d Turns' % (tetrode,np.concatenate(all_turn_data[behavior_condition][dx_type][dx_key]).shape[0]) ] 
              )
    
    gs = gridspec.GridSpec(3, 20)
  

    ax1 = plt.subplot(gs[0, 0:19])   
    im1 = ax1.pcolormesh(np.concatenate(all_turn_data[behavior_condition][dx_type][dx_key])[:,:,tetrode],vmin=0,vmax=1,cmap='RdPu')
    im1.set_rasterized(True)
    ax1.set_ylabel('Turn Trials')
    #ax1b = plt.subplot(gs[0, 19])   
    
    divider = make_axes_locatable(ax1)
    cax1 = divider.append_axes("right", size="5%", pad=0.05)
    f.colorbar(im1,cax=cax1)
    
  

    ax2 = plt.subplot(gs[1, 0:19],sharex=ax1)
    #ax2b = plt.subplot(gs[1, 19],sharey=ax2)   
    
    im2 = ax2.pcolormesh(np.mean(np.concatenate(all_turn_data[behavior_condition][dx_type][dx_key]),axis=0).T,vmin=0,vmax=.5,cmap='RdPu')
    im2.set_rasterized(True)
    ax2.set_ylabel('Tetrodes')  
    #f.colorbar(im2,ax=ax2b)
    divider = make_axes_locatable(ax2)
    cax2 = divider.append_axes("right", size="5%", pad=0.05)
    f.colorbar(im2, cax=cax2)
    
    
    ax3 = plt.subplot(gs[2, 0:19],sharex=ax1)
    ax3 = sns.tsplot(data=np.concatenate(all_turn_data[behavior_condition][dx_type][dx_key])[:,:,tetrode],
                     color='k',ci=68,linewidth=1)
    
    divider = make_axes_locatable(ax3)
    cax3 = divider.append_axes("right", size="5%", pad=0.05,add_to_figure=False)
 
    ax3.set_ylabel('Tetrode %d \nMean Across Trials' % tetrode)
  
    
    ax2.set_xlabel('Time From Turn Onset (sec)')
    
    return f

if __name__ == "__main__":


    input_file_path = os.getcwd()
    all_files = []

    for file in os.listdir(input_file_path):
            if file.startswith("636"):
                all_files.append(file)
    all_files = np.asarray(all_files)

    """
    Want to get turn files (dx, dy, dz) from each experiment (636...) folder, put them all together, and make one big variable that contains:

    - for each behavioral condition (dark, light, muscimol dark, muscimol light):
        - for each direction of turn (left, right, up, down, CW, CCW):
            - each turn_type dict has a dict with matrices
                - the behavior matrix (turns x window)
                - the ephys matrix (turns x window x tetrode)



    each dx (or dy or dz) contains four elements: two behavior directions and two corresponding ephys matrices. 

    """
    #all_turns = {}
    rat = sys.argv[1]
    tetrodes = sys.argv[2]

    behavior_conditions = ['dark', 'light', 'muscimol_dark', 'muscimol_light']

    dx_types = ['dx','dy','dz']

    dx_keys = ['y_left', 'y_right', 'X_left', 'X_right'] ### these are the keys of each dx.npy (or dy or dz) files. y = behavior, X = ephys, left = one direction, right = the other

    #turn_names = ['left', 'right', 'up', 'down', 'cw', 'ccw']



    exp_names = get_experiment_names(all_files,rat)

    condition_indexes = get_condition_indexes(exp_names,behavior_conditions)

    all_turns = get_all_turns(all_files,dx_types)


    all_turn_data = make_turn_dict(all_turns,exp_names,dx_types,dx_keys)


    for tetrode in tetrodes:
        f = plot_single_tetrode(all_turn_data,behavior_condition='dark',dx_type='dx',dx_key='X_left',tetrode=tetrode)


        f.savefig('./turn_plots/single_tetrode_%d.pdf' % tetrode)


















