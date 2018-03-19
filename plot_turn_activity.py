import numpy as np
import pandas as pd
import sys,os
import pylab
from scipy import stats,signal,io
import matplotlib.pyplot as plt
from sklearn import linear_model
import pickle
import seaborn as sns
sns.set_style('white')
import h5py
import matplotlib.gridspec as gridspec
from preprocessing_funcs import get_spikes_with_history

def group_consecutives(vals, step=1):
    """Return list of consecutive lists of numbers from vals (number list)."""
    run = []
    result = [run]
    expect = None
    for v in vals:
        if (v == expect) or (expect is None):
            run.append(v)
        else:
            run = [v]
            result.append(run)
        expect = v + step
    return result

def get_turn_peaks(dx,threshold):
    
    ## ephys = samples x electrode channels
    crossings =  np.where(abs(dx) > threshold)[0]
    peaks = []
    grouped_crossings = group_consecutives(crossings)
    for idx,thing in enumerate(grouped_crossings):
        center = thing[np.argmax(abs(dx[thing]))]
        peaks.append(center)
        
    return peaks

def extract_peak_windows(mua,derivative):
    """
    Take ephys (mua) and head signal (derivative), find peaks from the head signal and extract windowed data aligned with those peaks
    """


    ### standardize ephys and get windows:
    mua_mean=np.nanmean(mua,axis=0)
    mua_std=np.nanstd(mua,axis=0)
    mua=(mua-mua_mean)/mua_std
    X = get_spikes_with_history(mua,100,100)

    d_history = get_spikes_with_history(np.atleast_2d(derivative).T,100,100) ### get windows for the derivative trace too

    ## find peaks in the derivative trace:
    d_peaks = get_turn_peaks(derivative,threshold=1)
    d_peaks = np.asarray(d_peaks)


    ## take only those peaks in the ephys and derivative:
    #y = dy[dy_peaks]
    X_peaks = X[d_peaks,:,:]
    peak_windows = d_history[d_peaks,:,0]



    labels = []
    for peak in d_peaks:
        if derivative[peak] > 0:
            labels.append(1)
        elif derivative[peak] < 0:
            labels.append(-1)
    labels = np.asarray(labels)


    """
    -1 dx = left turns; -1 dy = CW roll; -1 dz = up nod
    """



    ### separate the windowed X and y into left and right (or up/down) components:
    y_left = peak_windows[np.where(labels == -1)[0],:]
    y_right = peak_windows[np.where(labels == 1)[0],:]

    X_left = X_peaks[np.where(labels == -1)[0],:,:]
    X_right = X_peaks[np.where(labels == 1)[0],:,:]

    return y_left,y_right,X_left,X_right

def plot(y_left,y_right,X_left,X_right,head_name,file_name):

    f = plt.figure(dpi=600)


    gs = gridspec.GridSpec(3, 1)

      
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[1, 0])
    ax3 = plt.subplot(gs[2, 0])

    if head_name == 'dx':
        dir1 = 'Left'
        dir2 = 'Right'
    elif head_name == 'dy':
        dir1 = 'CW'
        dir2 = 'CCW'
    elif head_name == 'dz':
        dir1 = 'Up'
        dir2 = 'Down'

    time = np.arange(-1.,1.01,0.01)

    ax1.text(0.95, -.75, dir1,
            verticalalignment='bottom', horizontalalignment='right', color="#2ecc71", fontsize=15)

    ax1.text(0.95, .75, dir2,
            verticalalignment='bottom', horizontalalignment='right', color="#9b59b6", fontsize=15)


    #for i in range(dx_peak_windows.shape[0]):
    #    ax1.plot(time,dx_peak_windows[i,:],c='k',lw=.01)

    sns.tsplot(y_left,time = time,color="#2ecc71",ax=ax1)
    sns.tsplot(y_right,time = time, color="#9b59b6",ax=ax1)
    #ax1 = plt.plot([0,0],ax1.get_ylim(),c='k',alpha=.5)
    ax1.set_ylabel('Mean ' + head_name)
    ax1.axes.xaxis.set_ticklabels([])

    # sns.tsplot(np.mean(X_left,axis=0).T,time = time, color="#2ecc71",ax=ax2)

    # sns.tsplot(np.mean(X_right,axis=0).T,time=time,color="#9b59b6",ax=ax2)

    # ax2.set_ylabel('Z-scored Firing Rate')
    # ax2.set_xlabel('Time from turn onset (sec)')

    mean_X_right = np.mean(X_right,axis=0) ## (trials x time x tetrode) -> (time x tetrode)
    mean_X_left = np.mean(X_left,axis=0)

    norm_X_right = mean_X_right  #/ np.max(mean_X_right,axis=0)
    norm_X_left = mean_X_left  #/ np.max(mean_X_left,axis=0)

    ax2.pcolormesh(time,np.arange(17),norm_X_right.T,cmap='viridis',edgecolors='face')

    ax3.pcolormesh(time,np.arange(17),norm_X_left.T,cmap='viridis',edgecolors='face')

    ax2.axes.xaxis.set_ticklabels([])

    ax2.set_ylabel('Firing Rates \n for Right Turns')
    ax3.set_ylabel('Firing Rates \n for Left Turns')

    ax3.set_xlabel('Time from turn onset (sec)')

    #ax2 = plt.plot([0,0],ax2.get_ylim(),c='k',alpha=.5)


    sns.despine(left=True,bottom=True)

    f.savefig('./' + file_name + '/' + head_name + '.pdf')

def get_X_y(path):

    head_data = h5py.File('./'+ path +'/' + 'all_head_data_100hz.hdf5','r')

    idx_start, idx_stop = [0,9]
    head_signals = np.asarray([np.asarray(head_data[key]) for key in head_data.keys()][0:9]).T[:,idx_start:idx_stop]
    print('head_signals shape: ', head_signals.shape) ## samples x features

    mua_file = h5py.File('./' + path + '/mua_firing_rates_100hz.hdf5','r')

    mua = mua_file['firing_rates'][:]

    mua_file.close()

    limit = int(6e6)
    if mua.shape[0] > limit:
        print('Reducing Data Size Down to %d Samples' % limit)
        mua  = mua[0:limit,:]
        head_signals = head_signals[0:limit,:]

    return mua,head_signals


if __name__ == "__main__":


    ### GRat26:
    # all_files = ['636444372823550001',
    #             '636455956910510001',
    #             '636456486214960001',
    #             '636462665576670001',
    #             '636464313683750001',
    #             '636459234842450001',
    #             '636461899932110001',
    #             '636463456945299197',
    #             '636464513098770001',
    #             '636465322265480001']
    # all_files =  [
    #                 '636461785781685886',
    #                 '636462564120970001',
    #                 '636464402756620001',
    #                 '636465208859170001',

    #             ]

    ### GRat31
    # all_files = ['636426429249996956',
    #              '636427229078062061',
    #              '636427282621202061',
    #              '636428029026710180',
    #              '636428089543470180',
    #              '636428953768193973',
    #              '636429016120323973',
    #              '636429913515267697',
    #              '636430663717571697',
    #              '636431765535543697',
    #              '636432491422312001',
    #              '636438551166665948',
    #              '636438658377315948',
    #              '636439164041965948',
    #              '636439502672505948',
    #              '636440035877005948']

    # input_file_path = os.getcwd()
    # all_files = []
    # for file in os.listdir(input_file_path):
    #         if file.startswith("636"):
    #             all_files.append(file)

    all_files = ['636511715885134658']
    for fil in all_files:

        mua,head_signals = get_X_y(fil)
        head_names = ['dx','dy','dz']
        
        for i in range(3):
            derivative = head_signals[:,3+i]

            y_left,y_right,X_left,X_right = extract_peak_windows(mua,derivative)

            plot(y_left,y_right,X_left,X_right,head_names[i],fil)

            np.savez('./' + fil + '/' + head_names[i] + '.npz',y_left=y_left,y_right=y_right,X_left=X_left,X_right=X_right)


