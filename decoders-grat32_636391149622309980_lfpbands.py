
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
import matplotlib.pyplot as plt
import sys,os
from scipy import io
from scipy import stats
import pickle
import h5py
#Import function to get the covariate matrix that includes spike history from previous bins
from preprocessing_funcs import get_spikes_with_history

#Import metrics
from metrics import get_R2
from metrics import get_rho

#Import decoder functions
from decoders import WienerCascadeDecoder
from decoders import WienerFilterDecoder
from decoders import DenseNNDecoder
from decoders import SimpleRNNDecoder
from decoders import GRUDecoder
from decoders import LSTMDecoder
from decoders import XGBoostDecoder
from decoders import SVRDecoder

def load_data(folder,spectrogram=0):
	# ## 2. Load Data
	# The data file for this example can be downloaded at https://dl.dropboxusercontent.com/u/2944301/Decoding_Data/example_data_s1.pickle. It was recorded by Raeed Chowdhury from Lee Miller's lab at Northwestern.
	# 
	# 
	# The data that we load is in the format described below. We have another example notebook, "Example_format_data", that may be helpful towards putting the data in this format.
	# 
	# Neural data should be a matrix of size "number of time bins" x "number of neurons", where each entry is the firing rate of a given neuron in a given time bin
	# 
	# The output you are decoding should be a matrix of size "number of time bins" x "number of features you are decoding"
	# 
	#  

	# In[3]:

	#folder='/Volumes/Mac HD/Dropbox (coxlab)/Ephys/Data/GRat32/636391149622309980/' #ENTER THE FOLDER THAT YOUR DATA IS IN
	# folder='/home/jglaser/Data/DecData/' 
	# folder='/Users/jig289/Dropbox/Public/Decoding_Data/'

	

	with open(folder+'/sortedspikes_win1step01.pickle','rb') as f:
	#     neural_data,vels_binned=pickle.load(f,encoding='latin1') #If using python 3
	    spike_time_vec,neural_data=pickle.load(f) #If using python 2

    data_file = h5py.File(folder+'/lfpbands_jerkraw.mat','r')
	
	neural_data = data_file['lfp_bands']
	
	jerk = data_file['jerk_dat']
	

	# In[78]:

	#f, axarr = plt.subplots(3, sharex=True,dpi=600)
	#axarr[0].plot(spike_time_vec,neural_data[:,0],linewidth=.25)
	#axarr[1].plot(jerk_time,jerk_power[:,0],linewidth=.25)
	#axarr[2].plot(spike_time_vec,raw_jerk[:,0],linewidth=.25)

	return jerk,neural_data


def preprocess(jerk,neural_data):
	# ## 3. Preprocess Data

	# ### 3A. User Inputs
	# The user can define what time period to use spikes from (with respect to the output).

	# In[25]:

	bins_before=15000 #How many bins of neural data prior to the output are used for decoding
	bins_current=1 #Whether to use concurrent time bin of neural data
	bins_after=15000 #How many bins of neural data after the output are used for decoding


	# ### 3B. Format Covariates

	# #### Format Input Covariates

	# In[26]:

	# Format for recurrent neural networks (SimpleRNN, GRU, LSTM)
	# Function to get the covariate matrix that includes spike history from previous bins
	X=get_spikes_with_history(neural_data,bins_before,bins_after,bins_current)

	# Format for Wiener Filter, Wiener Cascade, XGBoost, and Dense Neural Network
	#Put in "flat" format, so each "neuron / time" is a single feature
	X_flat=X.reshape(X.shape[0],(X.shape[1]*X.shape[2]))



	# #### Format Output Covariates

	# In[79]:

	#Set decoding output
	#y=jerk_power
	y=jerk




	# ### 3C. Split into training / testing / validation sets
	# Note that hyperparameters should be determined using a separate validation set. 
	# Then, the goodness of fit should be be tested on a testing set (separate from the training and validation sets).

	# #### User Options

	# In[32]:

	#Set what part of data should be part of the training/testing/validation sets
	training_range=[0, 0.5]
	testing_range=[0.7, 0.85]
	valid_range=[0.5,1]


	# #### Split Data

	# In[81]:

	num_examples=X.shape[0]

	#Note that each range has a buffer of"bins_before" bins at the beginning, and "bins_after" bins at the end
	#This makes it so that the different sets don't include overlapping neural data
	training_set=np.arange(np.int(np.round(training_range[0]*num_examples))+bins_before,np.int(np.round(training_range[1]*num_examples))-bins_after)
	testing_set=np.arange(np.int(np.round(testing_range[0]*num_examples))+bins_before,np.int(np.round(testing_range[1]*num_examples))-bins_after)
	valid_set=np.arange(np.int(np.round(valid_range[0]*num_examples))+bins_before,np.int(np.round(valid_range[1]*num_examples))-bins_after)

	#Get training data
	X_train=X[training_set,:,:]
	X_flat_train=X_flat[training_set,:]


	y_train=y[training_set,:]

	#Get testing data
	X_test=X[testing_set,:,:]
	X_flat_test=X_flat[testing_set,:]

	y_test=y[testing_set,:]

	#Get validation data
	X_valid=X[valid_set,:,:]
	X_flat_valid=X_flat[valid_set,:]


	y_valid=y[valid_set,:]


	# ### 3D. Process Covariates
	# We normalize (z_score) the inputs and zero-center the outputs.
	# Parameters for z-scoring (mean/std.) should be determined on the training set only, and then these z-scoring parameters are also used on the testing and validation sets.

	# In[ ]:




	# In[82]:

	#Z-score "X" inputs. 
	X_train_mean=np.nanmean(X_train,axis=0)
	X_train_std=np.nanstd(X_train,axis=0)
	X_train=(X_train-X_train_mean)/X_train_std
	X_test=(X_test-X_train_mean)/X_train_std
	X_valid=(X_valid-X_train_mean)/X_train_std


	#Z-score "X_flat" inputs. 
	X_flat_train_mean=np.nanmean(X_flat_train,axis=0)
	X_flat_train_std=np.nanstd(X_flat_train,axis=0)
	X_flat_train=(X_flat_train-X_flat_train_mean)/X_flat_train_std
	X_flat_test=(X_flat_test-X_flat_train_mean)/X_flat_train_std
	X_flat_valid=(X_flat_valid-X_flat_train_mean)/X_flat_train_std

	#Zero-center outputs
	y_train_mean=np.mean(y_train,axis=0)
	y_train=y_train-y_train_mean
	y_test=y_test-y_train_mean
	y_valid=y_valid-y_train_mean

	print 'X_flat_train.shape, y_train.shape = ', X_flat_train.shape,y_train.shape

	return X_flat_train,X_flat_valid,X_train,X_valid,y_train,y_valid

# ## 4. Run Decoders





def Wiener(X_flat_train,X_flat_valid,y_train,y_valid):
	#Declare model
	model_wf=WienerFilterDecoder()

	#Fit model
	model_wf.fit(X_flat_train,y_train)

	#Get predictions
	y_valid_predicted_wf=model_wf.predict(X_flat_valid)

	#Get metric of fit
	R2s_wf=get_R2(y_valid,y_valid_predicted_wf)
	print('R2s:', R2s_wf)

	#plot_results(y_valid,y_valid_predicted_wf)

	return model_wf

def WienerCascade():
	# ### 4B. Wiener Cascade (Linear Nonlinear Model)

	# In[39]:

	#Declare model
	model_wc=WienerCascadeDecoder(degree=3)

	#Fit model
	model_wc.fit(X_flat_train,y_train)

	#Get predictions
	y_valid_predicted_wc=model_wc.predict(X_flat_valid)

	#Get metric of fit
	R2s_wc=get_R2(y_valid,y_valid_predicted_wc)
	print('R2s:', R2s_wc)

def XGBoost():
	# ### 4C. XGBoost (Extreme Gradient Boosting)

	# In[ ]:

	#Declare model
	model_xgb=XGBoostDecoder(max_depth=3,num_round=200,eta=0.3,gpu=-1) 

	#Fit model
	model_xgb.fit(X_flat_train, y_train)

	#Get predictions
	y_valid_predicted_xgb=model_xgb.predict(X_flat_valid)

	#Get metric of fit
	R2s_xgb=get_R2(y_valid,y_valid_predicted_xgb)
	print('R2s:', R2s_xgb)

def SVR():
	# ### 4D. SVR (Support Vector Regression)

	# In[40]:

	#The SVR works much better when the y values are normalized, so we first z-score the y values
	#They have previously been zero-centered, so we will just divide by the stdev (of the training set)
	y_train_std=np.nanstd(y_train,axis=0)
	y_zscore_train=y_train/y_train_std
	y_zscore_test=y_test/y_train_std
	y_zscore_valid=y_valid/y_train_std

	#Declare model
	model_svr=SVRDecoder(C=5, max_iter=4000)

	#Fit model
	model_svr.fit(X_flat_train,y_zscore_train)

	#Get predictions
	y_zscore_valid_predicted_svr=model_svr.predict(X_flat_valid)

	#Get metric of fit
	R2s_svr=get_R2(y_zscore_valid,y_zscore_valid_predicted_svr)
	print('R2s:', R2s_svr)

def DNN():
	# ### 4E. Dense Neural Network

	# In[ ]:

	#Declare model
	model_dnn=DenseNNDecoder(units=400,dropout=0.25,num_epochs=10)

	#Fit model
	model_dnn.fit(X_flat_train,y_train)

	#Get predictions
	y_valid_predicted_dnn=model_dnn.predict(X_flat_valid)

	#Get metric of fit
	R2s_dnn=get_R2(y_valid,y_valid_predicted_dnn)
	print('R2s:', R2s_dnn)

def RNN():
	# ### 4F. Simple RNN

	# In[ ]:

	#Declare model
	model_rnn=SimpleRNNDecoder(units=400,dropout=0,num_epochs=5)

	#Fit model
	model_rnn.fit(X_train,y_train)

	#Get predictions
	y_valid_predicted_rnn=model_rnn.predict(X_valid)

	#Get metric of fit
	R2s_rnn=get_R2(y_valid,y_valid_predicted_rnn)
	print('R2s:', R2s_rnn)

def GRU():
	# ### 4G. GRU (Gated Recurrent Unit)

	# In[ ]:

	#Declare model
	model_gru=GRUDecoder(units=400,dropout=0,num_epochs=5)

	#Fit model
	model_gru.fit(X_train,y_train)

	#Get predictions
	y_valid_predicted_gru=model_gru.predict(X_valid)

	#Get metric of fit
	R2s_gru=get_R2(y_valid,y_valid_predicted_gru)
	print('R2s:', R2s_gru)

def run_LSTM(X_train,X_valid,y_train,y_valid):
	# ### 4H. LSTM (Long Short Term Memory)

	# In[ ]:

	#Declare model
	model_lstm=LSTMDecoder(units=400,dropout=0,num_epochs=5)

	#Fit model
	model_lstm.fit(X_train,y_train)

	#Get predictions
	y_valid_predicted_lstm=model_lstm.predict(X_valid)

	#Get metric of fit
	R2s_lstm=get_R2(y_valid,y_valid_predicted_lstm)
	print('R2s:', R2s_lstm)

	print 'Results: '
	print 'y_valid_predicted_lstm = ', y_valid_predicted_lstm
	print 'y_valid = ', y_valid

	np.savez('lstm_ypredicted.npz',y_valid_predicted_lstm=y_valid_predicted_lstm,y_valid=y_valid)

	plot_results(y_valid,y_valid_predicted_lstm)

	return model_lstm


def plot_results(y_valid,y_valid_predicted_wf):
	fig_x_wf=plt.figure(dpi=600)
	#plt.plot(y_valid[1000:2000,0]+y_train_mean[0],'b')
	#plt.plot(y_valid_predicted_wf[1000:2000,0]+y_train_mean[0],'r')

	plt.scatter(y_valid,y_valid_predicted_wf,alpha=0.1,marker='o')
	plt.axis('equal')
	fig_x_wf.savefig('test_vs_predicted.pdf')


# ## 5. Make Plots

# In[ ]:

#As an example, I plot an example 1000 values of the x velocity (column index 0), both true and predicted with the Wiener filter
#Note that I add back in the mean value, so that both true and predicted values are in the original coordinates

#Save figure
# fig_x_wf.savefig('x_velocity_decoding.eps')


# In[ ]:
if __name__ == "__main__":

	model_type = sys.argv[1] ## wiener or lstm

	jerk,neural_data = load_data(os.getcwd())

	X_flat_train,X_flat_valid,X_train,X_valid,y_train,y_valid = preprocess(jerk,neural_data)

	if model_type == 'lstm':
		data_model = run_LSTM(X_train,X_valid,y_train,y_valid)
	elif model_type == 'wiener':
		data_model = Wiener(X_flat_train,X_flat_valid,y_train,y_valid)

	with open('model_' + model_type + '_rawjerk','wb') as f:
		pickle.dump(data_model,f)
