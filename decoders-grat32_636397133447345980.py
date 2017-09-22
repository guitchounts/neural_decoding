
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
from decoders import WienerCascadeDecoder
from decoders import WienerFilterDecoder
from decoders import DenseNNDecoder
from decoders import SimpleRNNDecoder
from decoders import GRUDecoder
from decoders import LSTMDecoder
#from decoders import XGBoostDecoder
from decoders import SVRDecoder

import h5py
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model

import seaborn as sns
sns.set_style('white')

from scipy import stats,signal


def filter(ephys,freq_range,filt_order = 4,filt_type='bandpass',fs=10.):
	
    # design Elliptic filter:

    [b,a] = signal.butter(filt_order,[freq/fs for freq in freq_range],btype=filt_type)
    
    filtered_trace = signal.filtfilt(b,a,ephys,axis=0)
    return filtered_trace

def load_data(folder):

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


	y = np.vstack([xyz,oz,dx,dy,dz,ax,ay,az,ox,oy,theta]).T
	y_name = ['xyz','oz','dx','dy','dz','ax','ay','az','ox','oy','theta']

	#y = np.vstack([ox]).T
	#y_name = ['ox']

	#y = np.unwrap(np.unwrap(np.deg2rad(y)))

	#y = np.vstack([ox,oy,dx,dy,ax,ay,az]).T
	#y_name = ['ox','oy','dx','dy','ax','ay','az']


	#lfp_file = np.load('lfp_power.npz')
	lfp_file = h5py.File('lfp_power.hdf5','r')

	lfp_power = lfp_file['lfp_power'][:].T

	lfp_file.close()

	print 'Shape of head data = ', y.shape
	print 'Shape of LFP power = ', lfp_power.shape

	#for i in range(len(y_name)):
		#y[:,i] = signal.medfilt(y[:,i],[9])
	#	y[:,i] = filter(y[:,i],[1.],filt_type='lowpass')



	idx = 1000 #int(y.shape[0]/2)
	print 'max idx = ', idx
	return y[0:idx,:], lfp_power[0:idx,:],y_name
	
	
	

def preprocess(jerk,neural_data):
	# ## 3. Preprocess Data

	# ### 3A. User Inputs
	# The user can define what time period to use spikes from (with respect to the output).

	# In[25]:

	bins_before=10 #How many bins of neural data prior to the output are used for decoding
	bins_current=1 #Whether to use concurrent time bin of neural data
	bins_after=10 #How many bins of neural data after the output are used for decoding


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
	training_range=[0.2, 1]
	testing_range=[0.7, 0.85]
	valid_range=[0, 0.2]


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

	#Z-score  outputs
	y_train_mean=np.mean(y_train,axis=0)

	y_train_std=np.nanstd(y_train,axis=0)


	#### 
	y_train=(y_train-y_train_mean)/y_train_std
	y_test=(y_test-y_train_mean)/y_train_std
	y_valid=(y_valid-y_train_mean)/y_train_std

	return X_flat_train,X_flat_valid,X_train,X_valid,y_train,y_valid,y_train_mean,y_train_std

# ## 4. Run Decoders

def BayesianRidge_model(X_train,X_valid,y_train,y_test,y_name, y_train_mean,y_train_std):
 
 	model_name = 'BayesianRidge'
	print 'head items to fit are: ', y_name
		# In[ ]:
	for head_item in range(len(y_name)):

		y_train_item = y_train[:,head_item]
		y_train_item = np.reshape(y_train_item,[y_train.shape[0],1])

		y_test_item = y_test[:,head_item]
		y_test_item = np.reshape(y_test_item,[y_test_item.shape[0],1])
		print '********************************** Fitting %s on %s Data **********************************' % (model_name,y_name[head_item])
		#Declare model
		model = linear_model.BayesianRidge(compute_score=True)

		

		#Fit model
		model.fit(X_train,y_train_item)

		#Get predictions
		y_valid_predicted=model.predict(X_valid)


		training_prediction=model.predict(X_train)

		R2s_training=get_R2(y_train_item,training_prediction)
		print 'R2 on training set = ', R2s_training

		#Get metric of fit
		R2s=get_R2(y_test_item,y_valid_predicted)
		print('R2s:', R2s)
		print 'saving prediction ...'
		np.savez(y_name[head_item] + '_%s_ypredicted.npz' % model_name,y_test=y_test_item,y_prediction=y_valid_predicted,
			y_train_=y_train_item,training_prediction=training_prediction,
			y_train_mean=y_train_mean[head_item],y_train_std=y_train_std[head_item])
		#print 'saving model ...'
		joblib.dump(model, y_name[head_item] + '_%s.pkl' % model_name) 
		print 'plotting results...'
		plot_results(y_test_item,y_valid_predicted,y_name[head_item],R2s,model_name=model_name)

	return model




def ridgeCV_model(X_train,X_valid,y_train,y_test,y_name, y_train_mean,y_train_std):
 
	print 'head items to fit are: ', y_name
		# In[ ]:
	for head_item in range(len(y_name)):

		y_train_item = y_train[:,head_item]
		y_train_item = np.reshape(y_train_item,[y_train.shape[0],1])

		y_test_item = y_test[:,head_item]
		y_test_item = np.reshape(y_test_item,[y_test_item.shape[0],1])
		print '********************************** Fitting RidgeCV on %s Data **********************************' % y_name[head_item]
		#Declare model
		model = linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0],normalize=True,fit_intercept=True)

		

		#Fit model
		model.fit(X_train,y_train_item)

		#Get predictions
		y_valid_predicted=model.predict(X_valid)


		training_prediction=model.predict(X_train)

		R2s_training=get_R2(y_train_item,training_prediction)
		print 'R2 on training set = ', R2s_training

		#Get metric of fit
		R2s=get_R2(y_test_item,y_valid_predicted)
		print('R2s:', R2s)
		print 'saving prediction ...'
		np.savez(y_name[head_item] + '_RidgeCV_ypredicted.npz',y_test=y_test_item,y_prediction=y_valid_predicted,
			y_train_=y_train_item,training_prediction=training_prediction,
			y_train_mean=y_train_mean[head_item],y_train_std=y_train_std[head_item])
		#print 'saving model ...'
		joblib.dump(model, y_name[head_item] + '_Ridge.pkl') 
		print 'plotting results...'
		plot_results(y_test_item,y_valid_predicted,y_name[head_item],R2s,model_name='RidgeCV')

	return model



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

def WienerCascade(X_train,X_valid,y_train,y_test,y_name, y_train_mean,y_train_std):
	# ### 4B. Wiener Cascade (Linear Nonlinear Model)

	print 'head items to fit are: ', y_name
		# In[ ]:
	for head_item in range(len(y_name)):

		y_train_item = y_train[:,head_item]
		y_train_item = np.reshape(y_train_item,[y_train.shape[0],1])

		y_test_item = y_test[:,head_item]
		y_test_item = np.reshape(y_test_item,[y_test_item.shape[0],1])
		print '********************************** Fitting WienerCascade on %s Data **********************************' % y_name[head_item]
		#Declare model
		model = WienerCascadeDecoder(degree=3)

		

		#Fit model
		model.fit(X_train,y_train_item)

		#Get predictions
		y_valid_predicted=model.predict(X_valid)


		training_prediction=model.predict(X_train)

		R2s_training=get_R2(y_train_item,training_prediction)
		print 'R2 on training set = ', R2s_training

		#Get metric of fit
		R2s=get_R2(y_test_item,y_valid_predicted)
		print('R2s:', R2s)
		print 'saving prediction ...'
		np.savez(y_name[head_item] + '_WienerCascade_ypredicted.npz',y_test=y_test_item,y_prediction=y_valid_predicted,
			y_train_=y_train_item,training_prediction=training_prediction,
			y_train_mean=y_train_mean[head_item],y_train_std=y_train_std[head_item])
		#print 'saving model ...'
		joblib.dump(model, y_name[head_item] + '_WienerCascade.pkl') 
		print 'plotting results...'
		plot_results(y_test_item,y_valid_predicted,y_name[head_item],R2s,model_name='WienerCascade')

	return model

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
	model_svr=SVRDecoder(C=.1, max_iter=10000,gamma=1e-5)
	
	#Fit model
	

	for head_item in range(len(y_name)):
		### fit one at a time and save/plot the results 
		print '########### Fitting SVR on %s data ###########' % y_name[head_item]

		y_zscore_train_item = y_zscore_train[:,head_item]
		y_zscore_train_item = np.reshape(y_zscore_train_item,[y_zscore_train.shape[0],1])
		print 'shape of y_zscore_train_item = ', y_zscore_train_item.shape

		y_zscore_valid_item = y_zscore_valid[:,head_item]
		y_zscore_valid_item = np.reshape(y_zscore_valid_item,[y_zscore_valid_item.shape[0],1])

		model_svr.fit(X_flat_train,y_zscore_train_item)

		#Get predictions
		y_zscore_valid_predicted_svr=model_svr.predict(X_flat_valid)

		#Get metric of fit
		R2s_svr=get_R2(y_zscore_valid_item,y_zscore_valid_predicted_svr)
		print(y_name[head_item], 'R2:', R2s_svr)

		np.savez(y_name[head_item] + '_svr_ypredicted.npz',y_zscore_valid=y_zscore_valid_item,y_zscore_valid_predicted_svr=y_zscore_valid_predicted_svr)


		plot_results(y_zscore_valid_item,y_zscore_valid_predicted_svr,y_name[head_item],R2s_svr)

def DNN(X_train,X_valid,y_train,y_test,y_name):
	# ### 4E. Dense Neural Network

	
	print 'head items to fit are: ', y_name
	# In[ ]:
	for head_item in range(len(y_name)):

		y_train_item = y_train[:,head_item]
		y_train_item = np.reshape(y_train_item,[y_train.shape[0],1])

		y_test_item = y_test[:,head_item]
		y_test_item = np.reshape(y_test_item,[y_test_item.shape[0],1])
		print '********************************** Fitting DNN on %s Data **********************************' % y_name[head_item]
		#Declare model
		model_dnn=DenseNNDecoder(units=[128,64,32],num_epochs=15)

		#Fit model
		model_dnn.fit(X_train,y_train_item)

		#Get predictions
		y_valid_predicted=model_dnn.predict(X_valid)

		#Get metric of fit
		R2s=get_R2(y_test_item,y_valid_predicted)
		print('R2s:', R2s)
		print 'saving prediction ...'
		np.savez(y_name[head_item] + '_DNN_ypredicted.npz',y_test=y_test_item,y_prediction=y_valid_predicted)
		#print 'saving model ...'
		#joblib.dump(model_dnn, y_name[head_item] + '_LSTM.pkl') 
		print 'plotting results...'
		plot_results(y_test_item,y_valid_predicted,y_name[head_item],R2s,model_name='DNN')

	return model_dnn

def RNN(X_train,y_train,X_valid,y_valid,y_name):
	model_name = 'RNN'
	# ### 4F. Simple RNN
	print '############################# RUNNING RNN #############################'
	# In[ ]:

	#Declare model
	model_rnn=SimpleRNNDecoder(units=400,dropout=0,num_epochs=100)

	for head_item in range(len(y_name)):
		### fit one at a time and save/plot the results 
		print '########### Fitting RNN on %s data ###########' % y_name[head_item]

		y_train_item = y_train[:,head_item]
		y_train_item = np.reshape(y_train_item,[y_train.shape[0],1])
		print 'shape of y_train_item = ', y_train_item.shape

		y_valid_item = y_valid[:,head_item]
		y_valid_item = np.reshape(y_valid_item,[y_valid_item.shape[0],1])

		model_rnn.fit(X_train,y_train_item)

		#Get predictions
		y_valid_predicted_rnn=model_rnn.predict(X_valid)

		#Get metric of fit
		R2s_rnn=get_R2(y_valid_item,y_valid_predicted_rnn)
		print(y_name[head_item], 'R2:', R2s_rnn)

		np.savez(y_name[head_item] + '_rnn_ypredicted.npz',y_valid=y_valid_item,y_valid_predicted_rnn=y_valid_predicted_rnn)


		plot_results(y_valid_item,y_valid_predicted_rnn,y_name[head_item],R2s_rnn,model_name)


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

def run_LSTM(X_train,X_valid,y_train,y_test,y_name, y_train_mean,y_train_std):
	# ### 4H. LSTM (Long Short Term Memory)
	print 'head items to fit are: ', y_name
	# In[ ]:
	for head_item in range(len(y_name)):

		y_train_item = y_train[:,head_item]
		y_train_item = np.reshape(y_train_item,[y_train.shape[0],1])

		y_test_item = y_test[:,head_item]
		y_test_item = np.reshape(y_test_item,[y_test_item.shape[0],1])
		print '********************************** Fitting Deep Net on %s Data **********************************' % y_name[head_item]
		#Declare model
		model_lstm=LSTMDecoder(dropout=0.25,num_epochs=5)

		model_lstm.get_means(y_train_mean,y_train_std) ### for un-zscoring during loss calculation ??? 

		#Fit model
		model_lstm.fit(X_train,y_train_item)

		#Get predictions
		y_valid_predicted_lstm=model_lstm.predict(X_valid)


		training_prediction=model_lstm.predict(X_train)

		R2s_training=get_R2(y_train_item,training_prediction)
		print 'R2 on training set = ', R2s_training

		#Get metric of fit
		R2s_lstm=get_R2(y_test_item,y_valid_predicted_lstm)
		print('R2s:', R2s_lstm)
		print 'saving prediction ...'
		np.savez(y_name[head_item] + '_LSTM_ypredicted.npz',y_test=y_test_item,y_prediction=y_valid_predicted_lstm,
			y_train_=y_train_item,training_prediction=training_prediction,
			y_train_mean=y_train_mean[head_item],y_train_std=y_train_std[head_item])
		#print 'saving model ...'
		#joblib.dump(model_lstm, y_name[head_item] + '_LSTM.pkl') 
		print 'plotting results...'
		plot_results(y_test_item,y_valid_predicted_lstm,y_name[head_item],R2s_lstm,model_name='LSTM')

	return model_lstm


def plot_results(y_valid,y_valid_predicted,y_name,R2s,params='_',model_name='SVR'):
	print 'y_valid shape = ',y_valid.shape
	print 'y_valid_predicted shape = ', y_valid_predicted.shape
    print stats.pearsonr(y_valid,y_valid_predicted)
    f, axarr = plt.subplots(2,dpi=600)
    axarr[0].set_title(model_name +' Model of %s. R^2 = %f. r = %f ' % (y_name,R2s,stats.pearsonr(y_valid,y_valid_predicted)[0] ))


    axarr[0].plot(y_valid,linewidth=0.1)
    axarr[0].set_ylabel('Head Data')

    axarr[0].plot(y_valid_predicted,linewidth=0.1,color='red')

    axarr[1].set_title(params)
    axarr[1].scatter(y_valid,y_valid_predicted,alpha=0.05,marker='o')
    #axarr[1].set_title('R2 = ' + str(R2s))
    axarr[1].set_xlabel('Actual')
    axarr[1].set_ylabel('Predicted')
    axarr[1].axis('equal')

    sns.despine(left=True,bottom=True)
    f.savefig(model_name + '_%s.pdf' % y_name)




# In[ ]:
if __name__ == "__main__":

	model_type = sys.argv[1] ## wiener or lstm

	head_data,neural_data,y_name = load_data(os.getcwd())

	X_flat_train,X_flat_valid,X_train,X_valid,y_train,y_valid, y_train_mean,y_train_std = preprocess(head_data,neural_data)

	
	if model_type == 'lstm':
		data_model = run_LSTM(X_train,X_valid,y_train,y_valid,y_name, y_train_mean,y_train_std)
	elif model_type == 'wiener':
		data_model = Wiener(X_flat_train,X_flat_valid,y_train,y_valid)
	elif model_type == 'svr':
		data_model = SVR(X_flat_train,X_flat_valid,y_train,y_valid,y_name)
	elif model_type == 'rnn':
		RNN(X_train,y_train,X_valid,y_valid,y_name,y_name)
	elif model_type == 'dnn':
		data_model = DNN(X_flat_train,X_flat_valid,y_train,y_valid,y_name)
	elif model_type == 'ridge':
		data_model = ridgeCV_model(X_flat_train,X_flat_valid,y_train,y_valid,y_name, y_train_mean,y_train_std)
	elif model_type == 'WienerCascade':
		data_model = WienerCascade(X_flat_train,X_flat_valid,y_train,y_valid,y_name, y_train_mean,y_train_std)
	elif model_type == 'BayesianRidge':
		data_model = BayesianRidge_model(X_flat_train,X_flat_valid,y_train,y_valid,y_name, y_train_mean,y_train_std)
	

	#with open('model_' + model_type + '_rawjerk','wb') as f:
	#	pickle.dump(data_model,f)
