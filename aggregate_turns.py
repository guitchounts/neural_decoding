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
            print(fil)
            keys = ['y_left', 'y_right', 'X_left', 'X_right'] ## right order
            dx_fil = np.load('./' + fil + '/%s.npz' % dx_type)
            
            #print('dx_fil.keys() = ', dx_fil.keys())
            x = [np.asarray(dx_fil[key]) for key in keys] 
            print([xx.shape for xx in x])
            #print('len(x) = ', len(x))
            dxs.append(x)

            dx_fil.close()
        print('len(dxs) = ', len(dxs))
        #print([xx.shape for xx in dxs])
        print('np.asarray(dxs).shape = ', np.asarray(dxs).shape)
        all_turns.append(np.asarray(dxs))
    
    return np.dstack(all_turns)


def make_empty_turn_dict(all_turns,exp_names,dx_types,dx_keys):
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

    return all_turn_data

def make_turn_dict(all_turns,exp_names,dx_types,dx_keys):

    num_files = all_turns.shape[0]

    all_turn_data = make_empty_turn_dict(all_turns,exp_names,dx_types,dx_keys)  #{}

    # ##### set up the dictionaries:
    # for file_num in range(num_files):
    #     all_turn_data[exp_names[file_num]] = {}
    #     for turn_type in range(3): ### dx, dy, dz
    #         all_turn_data[exp_names[file_num]][dx_types[turn_type]] = {}
    #         for i in range(4): ## ['y_left', 'y_right', 'X_left', 'X_right']
    #             all_turn_data[exp_names[file_num]][dx_types[turn_type]][dx_keys[i]] = []
    #             #print exp_names[file_num],dx_types[turn_type], dx_keys[i], all_turns[file_num,i,turn_type].shape # all_data['dark'][turn_names[]] 

    ##### fill in the dictionaries:
    for file_num in range(num_files):
        
        for turn_type in range(3): ### dx, dy, dz
            
            for i in range(4): ## ['y_left', 'y_right', 'X_left', 'X_right']
                all_turn_data[exp_names[file_num]][dx_types[turn_type]][dx_keys[i]].append(all_turns[file_num,i,turn_type])          


    return all_turn_data

def make_deviations_dict(all_deviations_data,all_turn_data,behavior_conditions,dx_types,dx_keys):
    

    

    # ##### fill in the dictionaries:
    # for file_num in range(num_files):
        
    #     for turn_type in range(3): ### dx, dy, dz
            
    #         for i in range(2,4): ## ['y_left', 'y_right', 'X_left', 'X_right']
    #             all_deviations_data[exp_names[file_num]][dx_types[turn_type]][dx_keys[i]].append(all_turns[file_num,i,turn_type])          

                    #for session in all_turn_data[behavior][dx_type][left_right]:
                       # get_deviation(session)
    print('all_turn_data keys ==== ', all_turn_data.keys())
    for behavior in [all_turn_data.keys()[0]]: # behavior_conditions:
        for dx_type in dx_types:
            for left_right in ['X_left','X_right']:
                print(behavior, dx_type, left_right)
                for session in all_turn_data[behavior][dx_type][left_right]:
                    print('session.shape = ', session.shape)
                   # get_deviation(session)
                    #all_turn_data[exp_names[file_num]][dx_types[turn_type]][dx_keys[i]].append
                    all_deviations_data[behavior][dx_type][left_right].append(get_deviation(session))


    return all_deviations_data

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


def plot_by_turn_type(all_turn_data,behavior_conditions,dx_type,dx_keys,time,behavior_colors,turn_real_names):

    f = plt.figure(dpi=600)
    gs = gridspec.GridSpec(3, 1)



    ## each ax = plot of all behavioral conditions:

    ### ax1 = behavior:
    for behavior_condition in behavior_conditions:
        ax1 = plt.subplot(gs[0, 0])
        ax1.set_ylabel('Turn')
        ax1.text(0.95, -.75, turn_real_names[dx_type][0],
            verticalalignment='bottom', horizontalalignment='right', color='k', fontsize=15)

        ax1.text(0.95, .75, turn_real_names[dx_type][1],
            verticalalignment='bottom', horizontalalignment='right', color='k', fontsize=15)

        
        for turn_key in range(2): #### this keys either y_left or y_right (i.e. the behavior), the first two options in dx_keys.
            #print(behavior_condition,dx_type,dx_keys[turn_key])
            ax1 = sns.tsplot(np.mean(np.concatenate(all_turn_data[behavior_condition][dx_type][dx_keys[turn_key]]),axis=0).T,
                             time=time,color = behavior_colors[behavior_condition],linewidth=.5,alpha=.75)

            
    ### ax2 = ephys, first turn:
    
        ax2 = plt.subplot(gs[1, 0])
        X_left = np.mean(np.concatenate(all_turn_data[behavior_condition][dx_type][dx_keys[2]]),axis=0).T
        print('X_left.shape = ',X_left.shape)
        ax2.set_ylabel(turn_real_names[dx_type][0])
        ax2 = sns.tsplot(X_left - np.mean(X_left[:,0:51]), ### dx_keys[2] = X_left
                         time=time,color = behavior_colors[behavior_condition])
        
        
    ### ax3 = ephys, second turn:

        ax3 = plt.subplot(gs[2, 0])
        X_left = np.mean(np.concatenate(all_turn_data[behavior_condition][dx_type][dx_keys[3]]),axis=0).T
        print('X_left.shape = ',X_left.shape)
        ax3.set_ylabel(turn_real_names[dx_type][1])
        ax3 = sns.tsplot(X_left - np.mean(X_left[:,0:51]), ### dx_keys[2] = X_left
                         time=time,color = behavior_colors[behavior_condition])

    
    ax1.set_ylim([-1.5,1.5])
    ax2.set_ylim([-0.2,0.2])
    ax3.set_ylim([-0.2,0.2])
    ax1.tick_params(axis='y',which='major',length=10,width=1)
    ax2.tick_params(axis='y',which='major',length=10,width=1)
    ax3.tick_params(axis='y',which='major',length=10,width=1)
    ax3.set_xlabel('Time from Turn Onset (sec)')
    sns.despine(bottom=True,offset=10)
            
            
    return f

def get_deviation(trace):
    ### trace = [trials x time x channels] e.g. 10124 x 201 x 16
    print('trace.shape == ', trace.shape)
    num_tetrodes = trace.shape[2]
    print('num_tetrodes =', num_tetrodes)
    win_start = 90
    win_stop = 111
    
    #peak_idx = np.argmax(abs(trace[:,win_start:win_stop,:]),axis=1)
    tetrode_mean = np.mean(trace,axis=0)
    peak_idx = np.argmax(abs(tetrode_mean[win_start:win_stop,:]),axis=0)
    
    turn_peak = np.empty(num_tetrodes)
    baseline = np.empty(num_tetrodes)
    for i in range(num_tetrodes):

        turn_peak[i] = tetrode_mean[win_start+peak_idx[i],i]
        baseline[i] = np.mean(trace[:,0:51,i])
 
    print('baseline.shape = ',baseline.shape )
    deviation = turn_peak - baseline #( turn_peak - baseline  ) / baseline * 100
    print('deviation.shape = ',deviation.shape) 
    
    return deviation


if __name__ == "__main__":


    input_file_path = os.getcwd()
    all_files = []

    for file in os.listdir(input_file_path):
            if file.startswith("636"):
                if os.path.exists(input_file_path + '/' + file + '/dx.npz'):        
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
    tetrodes = range(9,16)

    time = np.arange(-1.,1.01,0.01)

    behavior_conditions = ['dark', 'light', 'muscimol_dark', 'muscimol_light']

    dx_types = ['dx','dy','dz']

    dx_keys = ['y_left', 'y_right', 'X_left', 'X_right'] ### these are the keys of each dx.npy (or dy or dz) files. y = behavior, X = ephys, left = one direction, right = the other

    turn_real_names = {}

    turn_real_names['dx'] = ['Left','Right']

    turn_real_names['dy'] = ['CW','CCW']

    turn_real_names['dz'] = ['Up','Down']

    behavior_colors = {}
    colors = sns.color_palette("Dark2", 4)
    behavior_colors[behavior_conditions[0]] = colors[2]

    behavior_colors[behavior_conditions[1]] = colors[0]

    behavior_colors[behavior_conditions[2]] = colors[1]

    behavior_colors[behavior_conditions[3]] = "#3498db" #sns.color_palette("hls", 4)[2]


    exp_names = get_experiment_names(all_files,rat)
    #print('exp_names = ',exp_names)

    condition_indexes = get_condition_indexes(exp_names,behavior_conditions)


    #### do one file at a time! ??????
    # for i,fil in enumerate(['636516217500773145', '636516251855668924']):
    #     print(exp_names[i])

    #     all_turns = get_all_turns([fil],dx_types)


    #     all_turn_data = make_turn_dict(all_turns,[exp_names[i]],dx_types,dx_keys) ### dict with keys e.g. all_turn_data['dark']['dx']['X_left']



        #turn_frame = pd.DataFrame.from_dict(all_turn_data)
        #turn_frame.to_csv('./turn_plots/turn_frame.csv')
        #np.save('./turn_plots/turn_frame.npy',all_turn_data)

        # for tetrode in tetrodes:
        #     f = plot_single_tetrode(all_turn_data,behavior_condition='dark',dx_type='dx',dx_key='X_left',tetrode=tetrode)


        #     f.savefig('./turn_plots/single_tetrode_%d.pdf' % tetrode)




        # for dx_type in dx_types:

        #     f = plot_by_turn_type(all_turn_data,behavior_conditions,dx_type,dx_keys,time,behavior_colors,turn_real_names)

        #     f.savefig('./turn_plots/tetrode_avg_%s.pdf' % dx_type)

    all_peaks = {  }

    for i,fil in enumerate(all_files):
        all_peaks[fil] = {'Condition' :  exp_names[i] }

        for dx_type in dx_types:
            
            dx_fil = np.load('./' + fil + '/%s.npz' % dx_type) ### load a dx, dy, dz file from a given 636 fil. 
            
            dx_session = {key : dx_fil[key] for key in dx_fil.keys() } ##### each is a dict with keys = ['y_left', 'y_right', 'X_left', 'X_right'] 

            all_peaks[fil][dx_type] = {'X_left' : [] ,'X_right' : [] }
            #all_turns[fil][dx_type] = dx_session ### for each all_turns[636...][dx, dy, dz], add the session's dict  

            for turn_dir in ['X_left','X_right']:
                all_peaks[fil][dx_type][turn_dir] =  get_deviation(   dx_session[turn_dir]   )



    print(all_peaks)

    pickle.dump( all_peaks, open('all_peaks.p' , "wb" ) )







        # all_deviations_data = make_empty_turn_dict(all_turns,[exp_names[i]],dx_types,dx_keys)
        # all_deviations_data = make_deviations_dict(all_deviations_data,all_turn_data,behavior_conditions,dx_types,dx_keys)
        # print(all_deviations_data)
        # peaks_frame = pd.DataFrame.from_dict(all_deviations_data)
        
        # save_dir = '%s/turn_plots/' % fil
        # if not os.path.exists(save_dir):
        #             os.makedirs(save_dir)

        # peaks_frame.to_csv(save_dir + 'peaks_frame.csv')

        # pickle.dump( all_deviations_data, open(save_dir + 'peaks.p' , "wb" ) )








##########################################################################################################################################





