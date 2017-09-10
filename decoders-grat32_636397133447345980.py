
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
from scipy import stats,signal
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

import h5py


def load_data(folder,spectrogram=0):


#folder = '/Users/guitchounts/Dropbox (coxlab)/Ephys/Data/Grat32/636397133447345980//'




	with open(folder+'/interp_IMUdata.pickle','rb') as f:
	#     neural_data,vels_binned=pickle.load(f,encoding='latin1') #If using python 3
		[splrep_dx,splrep_dy,splrep_dz,splrep_jerkx,splrep_jerky,splrep_jerkz,truncated_lfp_time]=pickle.load(f) #If using python 2



	with open(folder+'/theta_xyspeed.pickle','rb') as f:
	#     neural_data,vels_binned=pickle.load(f,encoding='latin1') #If using python 3
		[theta,xy_speed]=pickle.load(f) #If using python 2



	with open(folder+'/interp_IMU_Oxyz_Axyz.pickle','rb') as f:
	#     neural_data,vels_binned=pickle.load(f,encoding='latin1') #If using python 3
		[splrep_ox,splrep_oy,splrep_oz,splrep_ax,splrep_ay,splrep_az]=pickle.load(f) #If using python 2



	lfp_data = h5py.File(folder+'/lfp_spec.mat','r')
	lfp_spec = lfp_data['lfp_spec'][:]
	lfp_time = lfp_data['t'][:]
	lfp_freq = lfp_data['f'][:]


	###### head data is at 300 Hz now. Decimate to 10 hz to match LFP power data


	#[xy_speed,theta,splrep_ox,splrep_oy,splrep_oz,splrep_ax,splrep_ay,splrep_az]

	decimated_xy_speed = signal.decimate(signal.decimate(xy_speed,10,zero_phase=True),3,zero_phase=True)
	decimated_theta = signal.decimate(signal.decimate(theta,10,zero_phase=True),3,zero_phase=True)

	decimated_ox = signal.decimate(signal.decimate(splrep_ox,10,zero_phase=True),3,zero_phase=True)
	decimated_oy = signal.decimate(signal.decimate(splrep_oy,10,zero_phase=True),3,zero_phase=True)
	decimated_oz = signal.decimate(signal.decimate(splrep_oz,10,zero_phase=True),3,zero_phase=True)

	decimated_ax = signal.decimate(signal.decimate(splrep_ax,10,zero_phase=True),3,zero_phase=True)
	decimated_ay = signal.decimate(signal.decimate(splrep_ay,10,zero_phase=True),3,zero_phase=True)
	decimated_az = signal.decimate(signal.decimate(splrep_az,10,zero_phase=True),3,zero_phase=True)


	start = np.where(np.isclose(lfp_time,truncated_lfp_time[0],rtol=1e-3))[0][0]
	stop = np.where(np.isclose(lfp_time,truncated_lfp_time[-1],rtol=1e-5))[0][0]
	print 'start,stop = ', start,stop

	lfp_spec_time_4aligning = lfp_time[start:stop]
	lfp_spec_4aligning = lfp_spec[:,:,start:stop]



	lfp_power = get_power_bands(lfp_spec_4aligning,lfp_freq)
	lfp_power = lfp_power.T

	
	y = np.vstack([decimated_ax,decimated_ay,decimated_az,decimated_ox,decimated_oy,decimated_oz,decimated_xy_speed,decimated_theta]).T
	y_name = ['ax','ay','az','ox','oy','oz','xy','theta']



	print 'Shape of head data = ', y.shape
	print 'Shape of LFP power = ', lfp_power.shape

	return y, lfp_power,y_name


def get_power_bands(lfp_spec,freqs):


	freq_bands = [ [0,4],[4,8],[8,12],[12,30],[30,60],[60,150] ]

	lfp_power = np.zeros([64*6,lfp_spec.shape[2]])  ## 64 channels x 4 bands

	counter = 0

	for ch in range(64):    
		
		for freq_band in freq_bands:
			power = get_power(lfp_spec[ch,:,:],freq_band,freqs)

			lfp_power[counter,:] = power
			counter += 1
		#power_0_4 = get_power(lfp_spec[ch,:,:],[0,4],freqs)
		#power_4_8 = get_power(lfp_spec[ch,:,:],[4,8],freqs)
		#power_8_12 = get_power(lfp_spec[ch,:,:],[8,12],freqs)
		#power_15_40 = get_power(lfp_spec[ch,:,:],[15,40],freqs)
		#power_40_100 = get_power(lfp_spec[ch,:,:],[40,100],freqs)
		#lfp_power[ch*4:(ch+1)*4,:] = power_0_4,power_5_15,power_15_40,power_40_100


	
	return lfp_power


def get_freq_idx(freqs,desired_freq): # make desired_freq a tuple, e.g. (0,4)
	idx = []
	for counter,value in enumerate(freqs):
		if  desired_freq[0] <= value <= desired_freq[1]:
			#yield counter
			idx.append(counter)
	return idx


def get_power(spec,freq_range,freqs):

	idx = get_freq_idx(freqs,freq_range)

	power = np.mean(spec[idx,:],0)

	return power

def preprocess(jerk,neural_data):
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

def SVR(X_flat_train,X_flat_valid,y_train,y_valid,y_name):
	# ### 4D. SVR (Support Vector Regression)

	# In[40]:

	#The SVR works much better when the y values are normalized, so we first z-score the y values
	#They have previously been zero-centered, so we will just divide by the stdev (of the training set)
	y_train_std=np.nanstd(y_train,axis=0)
	y_zscore_train=y_train/y_train_std
	#y_zscore_test=y_test/y_train_std
	y_zscore_valid=y_valid/y_train_std

	#Declare model
	model_svr=SVRDecoder(C=5, max_iter=10000)

	#Fit model
	

	for head_item in range(len(y_name)):
		### fit one at a time and save/plot the results 
		print '########### Fitting SVR on %s data ###########' % y_name[head_item]

		y_zscore_train_item = np.reshaped(y_zscore_train[:,head_item],y_zscore_train.shape[0],1)

		model_svr.fit(X_flat_train,y_zscore_train_item[:,head_item])

		#Get predictions
		y_zscore_valid_predicted_svr=model_svr.predict(X_flat_valid)

		#Get metric of fit
		R2s_svr=get_R2(y_zscore_valid[:,head_item],y_zscore_valid_predicted_svr)
		print(y_name[head_item], 'R2:', R2s_svr)

		np.savez(y_name[head_item] + '_svr_ypredicted.npz',y_zscore_valid=y_zscore_valid,y_zscore_valid_predicted_svr=y_zscore_valid_predicted_svr)


		plot_results(y_zscore_valid[:,head_item],y_zscore_valid_predicted_svr,y_name[head_item],R2s_svr)

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

	plot_results(y_valid,y_valid_predicted_lstm)

	return model_lstm


def plot_results(y_valid,y_valid_predicted,y_name,R2s):


	f, axarr = plt.subplots(2,dpi=600)
	axarr[0].set_title('SVR Model of %s. R^2 = %d ' % (y_name,R2s))


	axarr[0].plot(y_zscore_valid,linewidth=0.1)
	axarr[0].set_ylabel('Head Data')

	axarr[0].plot(y_zscore_valid_predicted_svr,linewidth=0.1,color='red')


	axarr[1].scatter(y_valid,y_valid_predicted,alpha=0.05,marker='o')
	axarr[1].set_title('R2 = ' + str(R2s_svr[0]))
	axarr[1].set_xlabel('Actual')
	axarr[1].set_ylabel('Predicted')
	axarr[1].axis('equal')

	sns.despine(left=True,bottom=True)
	f.savefig('svr_%s_.pdf' % y_name)




# In[ ]:
if __name__ == "__main__":

	model_type = sys.argv[1] ## wiener or lstm

	hea_data,neural_data,y_name = load_data(os.getcwd())

	X_flat_train,X_flat_valid,X_train,X_valid,y_train,y_valid = preprocess(hea_data,neural_data)

	if model_type == 'lstm':
		data_model = run_LSTM(X_train,X_valid,y_train,y_valid)
	elif model_type == 'wiener':
		data_model = Wiener(X_flat_train,X_flat_valid,y_train,y_valid)
	elif model_type == 'svr':
		data_model = SVR(X_flat_train,X_flat_valid,y_train,y_valid,y_name)

	#with open('model_' + model_type + '_rawjerk','wb') as f:
	#	pickle.dump(data_model,f)
