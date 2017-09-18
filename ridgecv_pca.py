
# coding: utf-8

# # Examples of all decoders (except Kalman Filter)
# 
# In this example notebook, we:
# 1. Import the necessary packages
# 2. Load a data file (spike trains and outputs we are predicting)
# 3. Preprocess the data for use in all decoders
# 4. Run all decoders and print the goodness of fit
# 5. Plot example decoded outputs
# 
# See "Examples_kf_decoder" for a Kalman filter example. <br>
# Because the Kalman filter utilizes different preprocessing, we don't include an example here. to keep this notebook more understandable

# ## 1. Import Packages
# 
# Below, we import both standard packages, and functions from the accompanying .py files

# In[2]:

#Import standard packages
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import sys,os
from scipy import io
from scipy import stats,signal
import pickle
import h5py
#Import function to get the covariate matrix that includes spike history from previous bins
from preprocessing_funcs import get_spikes_with_history

#Import metrics
from metrics import get_R2
from metrics import get_rho

#Import decoder functions
#from decoders import WienerCascadeDecoder
# from decoders import WienerFilterDecoder
# from decoders import DenseNNDecoder
# from decoders import SimpleRNNDecoder
# from decoders import GRUDecoder
# from decoders import LSTMDecoder
# from decoders import XGBoostDecoder
# from decoders import SVRDecoder



from sklearn.decomposition import PCA
from sklearn import linear_model
from sklearn.externals import joblib

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

import seaborn as sns
sns.set_style('white')

def load_data(folder,spectrogram=0):

    head_data = h5py.File('all_head_data.hdf5','r')
    


    dx = head_data['dx'][:]
    dy = head_data['dy'][:]
    dz = head_data['dz'][:]
    oz = head_data['oz'][:]
    oy = head_data['oy'][:]
    ox = head_data['ox'][:]
    ay = head_data['ay'][:]
    ax = head_data['ax'][:]
    az = head_data['az'][:]

    xyz = np.sqrt(ax**2 + ay**2 + az**2)

    theta = np.rad2deg(np.arctan(ax/ay))

    head_data.close()
    #xy_acc = head_data['xy_acc']
    #theta = head_data['theta']
    #time = head_data['time']


    y = np.vstack([dx,dy,dz,ax,ay,az,ox,oy,oz,xyz,theta]).T
    y_name = ['dx','dy','dz','ax','ay','az','ox','oy','oz','xyz','theta']

    #y = np.vstack([oz,dz,xyz,theta]).T
    #y_name = ['oz','dz','xyz','theta']


    #lfp_file = np.load('lfp_power.npz')
    lfp_file = h5py.File('lfp_power.hdf5','r')

    lfp_power = lfp_file['lfp_power'][:].T

    lfp_file.close()

    print 'Shape of head data = ', y.shape
    print 'Shape of LFP power = ', lfp_power.shape

    #for i in range(len(y_name)):
    #    y[:,i] = signal.medfilt(y[:,i],[9])

    idx = 10000 #int(y.shape[0]/2)
    print 'max idx = ', idx
    return y[0:idx,], lfp_power[0:idx,:],y_name
    


def preprocess(y,neural_data):
    # ## 3. Preprocess Data

    # ### 3A. User Inputs
    # The user can define what time period to use spikes from (with respect to the output).

    # In[25]:

    bins_before=25 #How many bins of neural data prior to the output are used for decoding
    bins_current=1 #Whether to use concurrent time bin of neural data
    bins_after=25 #How many bins of neural data after the output are used for decoding


    # ### 3B. Format Covariates

    # #### Format Input Covariates

    # In[26]:

    # Format for recurrent neural networks (SimpleRNN, GRU, LSTM)
    # Function to get the covariate matrix that includes spike history from previous bins
    X=get_spikes_with_history(neural_data,bins_before,bins_after,bins_current)

    # Format for Wiener Filter, Wiener Cascade, XGBoost, and Dense Neural Network
    #Put in "flat" format, so each "neuron / time" is a single feature
    X_flat=X.reshape(X.shape[0],(X.shape[1]*X.shape[2]))

    ####### reduce dimensionality of the input data (X) from ~8,000 to 100:
    pca = PCA(n_components=100)
    print 'Reducing Dimensionality with PCA'
    
    #pca.fit(X_flat[::10,:])

    #X_flat = pca.transform(X_flat)

    X_flat = pca.fit_transform(X_flat)


    # ### 3C. Split into training / testing / validation sets
    # Note that hyperparameters should be determined using a separate validation set. 
    # Then, the goodness of fit should be be tested on a testing set (separate from the training and validation sets).

    # #### User Options

    # In[32]:

    #Set what part of data should be part of the training/testing/validation sets
    training_range=[0, 0.5]
    #testing_range=[0.7, 0.85]
    valid_range=[0.5,1]


    # #### Split Data

    # In[81]:

    num_examples=X.shape[0]

    print 'Splitting Data into Training and Testing Sets'

    #Note that each range has a buffer of"bins_before" bins at the beginning, and "bins_after" bins at the end
    #This makes it so that the different sets don't include overlapping neural data
    training_set=np.arange(np.int(np.round(training_range[0]*num_examples))+bins_before,np.int(np.round(training_range[1]*num_examples))-bins_after)
    #testing_set=np.arange(np.int(np.round(testing_range[0]*num_examples))+bins_before,np.int(np.round(testing_range[1]*num_examples))-bins_after)
    valid_set=np.arange(np.int(np.round(valid_range[0]*num_examples))+bins_before,np.int(np.round(valid_range[1]*num_examples))-bins_after)

    #Get training data
    X_train=X[training_set,:,:]
    X_flat_train=X_flat[training_set,:]


    y_train=y[training_set,:]

    #Get testing data
    #X_test=X[testing_set,:,:]
    #X_flat_test=X_flat[testing_set,:]

    #y_test=y[testing_set,:]

    #Get validation data
    X_valid=X[valid_set,:,:]
    X_flat_valid=X_flat[valid_set,:]


    y_valid=y[valid_set,:]



    print 'Z-scoring Data'
    X_scaler = StandardScaler().fit(X_flat_valid)
    y_scaler = StandardScaler().fit(y_valid)


    X_flat_train = X_scaler.transform(X_flat_train)
    y_train = y_scaler.transform(y_train)

    X_flat_valid = X_scaler.transform(X_flat_valid)
    y_valid = y_scaler.transform(y_valid)

    return X_flat_train,X_flat_valid,X_train,X_valid,y_train,y_valid

# ## 4. Run Decoders


def run_ridge(X_train,X_test,y_train,y_test,y_name):
    # ### 4D. SVR (Support Vector Regression)

    
    #Declare model
    #Fit model
    

    for head_item in range(len(y_name)):

        ridge_model = linear_model.RidgeCV()
        ### fit one at a time and save/plot the results 
        print '########### Fitting RidgeCV on %s data ###########' % y_name[head_item]

        y_train_item = y_train[:,head_item]
        #y_train_item = np.reshape(y_train_item,[y_train.shape[0],1])
        print 'shape of y_train_item = ', y_train_item.shape

        y_test_item = y_test[:,head_item]
        #y_test_item = np.reshape(y_test_item,[y_test_item.shape[0],1])

        ridge_model.fit(X_train,y_train_item)

        #Get predictions
        y_prediction=ridge_model.predict(X_test)

        #Get metric of fit
        #R2s_svr=get_R2(y_test_item,y_prediction)
        R2 = ridge_model.score(X_test,y_test_item)
        print(y_name[head_item], 'R2:', R2)

        
        print 'saving prediction ...'
        np.savez(y_name[head_item] + '_ridgecv_ypredicted.npz',y_test=y_test_item,y_prediction=y_prediction)
        print 'saving model ...'
        joblib.dump(ridge_model, y_name[head_item] + '_ridgecv.pkl') 
        print 'plotting results...'
        plot_results(y_test_item,y_prediction,y_name[head_item],R2)


def plot_results(y_valid,y_valid_predicted,y_name,R2s,model_name='RidgeCV'):


    f, axarr = plt.subplots(2,dpi=600)
    axarr[0].set_title(model_name +' Model of %s. R^2 = %f ' % (y_name,R2s))


    axarr[0].plot(y_valid,linewidth=0.1)
    axarr[0].set_ylabel('Head Data')

    axarr[0].plot(y_valid_predicted,linewidth=0.1,color='red')
    axarr[0].set_xlabel('Time (samples)')


    
    axarr[1].scatter(y_valid,y_valid_predicted,alpha=0.05,marker='o')
    #axarr[1].set_title('R2 = ' + str(R2s))
    axarr[1].set_xlabel('Actual')
    axarr[1].set_ylabel('Predicted')
    axarr[1].axis('equal')

    sns.despine(left=True,bottom=True)
    f.savefig(model_name + '_%s.pdf' % y_name)




# In[ ]:
if __name__ == "__main__":

    #model_type = sys.argv[1] ## wiener or lstm
    print '############################ Running Ridge Regression with Cross Validation ############################'
    head_data,neural_data,y_name = load_data(os.getcwd())

    X_flat_train,X_flat_valid,X_train,X_valid,y_train,y_valid = preprocess(head_data,neural_data)


    run_ridge(X_flat_train,X_flat_valid,y_train,y_valid,y_name)
    
    
