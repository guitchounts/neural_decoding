
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
from scipy.stats import pearsonr
import matplotlib.gridspec as gridspec

# In[6]:

def get_R2(y_test,y_test_pred):
    y_mean=np.mean(y_test)
    R2=1-np.sum((y_test_pred-y_test)**2)/np.sum((y_test-y_mean)**2)
    return R2


def get_data(folder):
    y_tests = [] 
    y_predictions = []

    #train_means = []
    #train_stds = []

    R2s = []
    rs = []
    
    y_names = ['ox','oy','oz','dx','dy','dz','ax','ay','az','xyz','theta']
    y_plot_names = ['yaw','roll','pitch','dyaw','droll','dpitch','ML acc','AP acc','DV acc','total acc','heading angle']

    for i in range(len(y_names)):
        
        head_file = np.load(folder + '/%s_LSTM_ypredicted.npz' % y_names[i] )
        
        train_mean =head_file['y_train_mean']
        train_std = head_file['y_train_std']
        print  y_names[i], ' train mean and std are: ', train_mean,train_std

        tmp_test = head_file['y_test']  * train_std + train_mean 
        #print tmp_test.shape
        #print train_std.shape
        #print train_mean.shape
        y_tests.append( tmp_test )
        
        
        tmp_pred = head_file['y_prediction'] * train_std + train_mean 

        y_predictions.append(tmp_pred )

        R2s.append(get_R2(head_file['y_test'],head_file['y_prediction']))
        rs.append(pearsonr(head_file['y_test'],head_file['y_prediction'])[0])

    return y_tests,y_predictions,y_names,y_plot_names,R2s,rs


def plot_results(valids,predictions,y_name,R2s,r,model_name='GRU'):

    num_figs = len(valids)
    

    f = plt.figure(dpi=600,figsize=(7,num_figs))
    
    
    gs = gridspec.GridSpec(num_figs, 7)
    
    for i in range(num_figs):
        y_valid = valids[i]
        y_valid_predicted = predictions[i]
        #print y_valid.shape
        #print y_valid_predicted.shape

        ax1 = plt.subplot(gs[i, 0:4])
        ax2 = plt.subplot(gs[i, 4])
        ax3 = plt.subplot(gs[i, 5:])

        axarr = [ax1,ax2,ax3]

        #axarr[0].set_title(model_name +' Model of %s.' % y_name[i])
        axarr[0].plot(y_valid,linewidth=0.1,c='black')
        axarr[0].set_ylabel(y_name[i])
        axarr[0].plot(y_valid_predicted,linewidth=0.1,color='red')    
        axarr[0].set_title('R^2 = %f. r = %f' % (R2s[i],r[i]),fontsize= 12)
        
        
        axarr[1].scatter(y_valid,y_valid_predicted,alpha=0.05,s=2,marker='o')
           
        axarr[1].axis('equal')
        axarr[1].axes.xaxis.set_ticklabels([])
        axarr[1].axes.yaxis.set_ticklabels([])
        
        axarr[2].hist(y_valid,bins=100,color='black',alpha=.5)
        axarr[2].hist(y_valid_predicted,bins=100,color='red',alpha=.5)
        #axarr[2].set_xlabel(y_name[i])
        #axarr[2].axes.xaxis.set_ticklabels([])
        axarr[2].axes.yaxis.set_ticklabels([])
        #axarr[2].axes.xaxis.set_ticks([])
        #for d in ["left", "top", "bottom", "right"]:
        #    axarr[2].spines[d].set_visible(False)
        
        axarr[2].tick_params(axis="x", which="major", length=5)

        if i == num_figs-1:
            axarr[0].set_xlabel('Time (samples)')
            axarr[1].set_xlabel('Actual')
            axarr[1].set_ylabel('Predicted')
            
        else:
            axarr[0].axes.xaxis.set_ticklabels([])
            
    sns.despine(left=True,bottom=True)
    
    
    plt.tight_layout()
    
    f.savefig(model_name + '.pdf')
    


if __name__ == "__main__":

    folder = os.getcwd() # '/Users/guitchounts/Dropbox (coxlab)/Ephys/Data/GRat32/636397133447345980/'

    y_tests,y_predictions,y_names,y_plot_names,R2s,rs = get_data(folder)

    plot_results(y_tests,y_predictions,y_plot_names,R2s,rs,model_name='RidgeCV')









