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
from sklearn.model_selection import train_test_split
import sklearn.linear_model
from sklearn import metrics
from skimage import exposure
from sklearn.linear_model import LogisticRegression,SGDClassifier
plt.rcParams['pdf.fonttype'] = 'truetype'

def get_cdfs(head_signals,bins):
    
          

    hist,edges = np.histogram( head_signals, bins = bins,range=(bins[0],bins[-1]) , normed = True )
    
    dx = edges[1] - edges[0]
    cdf = np.cumsum(hist)*dx

    return cdf,edges[1:]


def plot_traces(head_data,names,fil):
    f, axarr = plt.subplots(4, sharex=True,dpi=600)

    
    axarr[0].plot(head_data[:,0],linewidth=.1,c='k')
    axarr[0].set_ylabel('Yaw')
    axarr[0].tick_params(axis='y',which='major',length=10,width=1)
    axarr[0].set_ylim([0,360])
    axarr[0].axes.yaxis.set_ticks([0,180,360])


    axarr[1].plot(head_data[:,1],linewidth=.1,c='k')
    axarr[1].set_ylabel('Roll')
    axarr[1].tick_params(axis='y',which='major',length=10,width=1)
    axarr[1].set_ylim([-90,90])
    axarr[1].axes.yaxis.set_ticks([-90,0,90])


    axarr[2].plot(head_data[:,2],linewidth=.1,c='k')
    axarr[2].set_ylabel('Pitch')
    axarr[2].tick_params(axis='y',which='major',length=10,width=1)
    axarr[2].set_ylim([-180,180])
    axarr[2].axes.yaxis.set_ticks([-180,0,180])

    axarr[3].plot(head_data[:,3],linewidth=.1,c='k')
    axarr[3].set_ylabel('Total acc')
    axarr[3].tick_params(axis='y',which='major',length=10,width=1)
    axarr[3].set_ylim([0,50])
    axarr[3].axes.yaxis.set_ticks([0,50])



    axarr[3].set_xlabel('Time (sec)')


    sns.despine(bottom=True,offset=5)
    
    f.savefig('./' + fil + '/behavior_traces.pdf')


def plot_histograms(head_data,names,fil):

    f = plt.figure(dpi=600)
    #f.suptitle(model_name, fontsize=10)

    gs = gridspec.GridSpec(2,2,wspace=.5,hspace=.5)
    count = 0

    ranges = [ [0,360],[-90,90],[-180,180],[0,20]   ]
    tick_labels = [ [0,180,360],[-90,0,90],[-180,0,180],[0,10,20]   ]

    for i in range(2):
        for j in range(2):
            
            ax1 = plt.subplot(gs[i,j])
            ax2 = plt.subplot(gs[i,j])
            ax2 = ax1.twinx()

            ax1.set_xlabel(names[count])

            ax1.tick_params(axis='x',direction='out',length=5,width=1,which='major')
            ax1.tick_params(axis='y',direction='out',length=5,width=1,which='major')
            
            #ax1.set_yticks([0,0.5,1.0])
            #ax2.set_ylim([0.0,1.0])
            ax1.set_xticks(tick_labels[count])
            ax1.set_xlim(ranges[count])

            if names[count] == 'Yaw':
                bins = np.linspace(0,360,100)
            elif names[count] == 'Roll':
                bins = np.linspace(-90,90,100)
            elif names[count] == 'Pitch':
                bins = np.linspace(-180,180,100)
            else:
                bins = np.linspace(0,20,100)

            cdfs,cdf_edges = get_cdfs(head_data[:,count],bins)
            print('head_data[:,count] ==== ', head_data[:,count][0:20])
            print('min max of head_data[:,count]: ', head_data[:,count].min(),head_data[:,count].max())
            ax1.hist(head_data[:,count],bins=bins,histtype='stepfilled')
            #ax2.hist(head_data[:,count],bins=bins,histtype='step',cumulative=True, color='r')
            ax2.plot(cdf_edges,cdfs,c='r',lw=1)
            
            
            count+=1

        sns.despine(offset=5)
            
        f.savefig('./' + fil + '/behavior_histograms.pdf')




if __name__ == "__main__":

    input_file_path = os.getcwd()
    all_files = []

    for file in os.listdir(input_file_path):
            if file.startswith("636"):
                all_files.append(file)
    all_files = np.asarray(all_files)

    #all_files = ['636511715885134658']
    for fil in all_files[-3:]:

        head_data = h5py.File('./'+ fil +'/' + 'all_head_data_100hz.hdf5','r')

        idx_start, idx_stop = [0,9]
        head_signals = np.asarray([np.asarray(head_data[key]) for key in head_data.keys()][0:9]).T[:,idx_start:idx_stop]
        print('head_signals shape: ', head_signals.shape) ## samples x features

        ### ox, oy, oz are idx 6,7,8 (last three)

        xyz = np.sqrt(head_signals[:,0]**2 + head_signals[:,1]**2 +  head_signals[:,2]**2)
        print('x y z  === ',head_signals[0:10,0], head_signals[0:10,1], head_signals[0:10,2])
        head_data_to_plot = np.vstack([ head_signals[:,6], head_signals[:,7], head_signals[:,8], xyz    ]).T
        print('head_data_to_plot.shape = ',head_data_to_plot.shape)
        head_names_to_plot = ['Yaw', 'Roll', 'Pitch', 'Total Acc']

        plot_traces(head_data_to_plot,head_names_to_plot,fil)

        plot_histograms(head_data_to_plot,head_names_to_plot,fil)


