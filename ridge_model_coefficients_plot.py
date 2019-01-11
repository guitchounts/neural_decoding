import numpy as np
import pandas as pd
import sys,os
import pylab
from scipy import stats,signal,io
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
%matplotlib inline
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
from sklearn.externals import joblib



def get_coef(file):

    with open(os.getcwd() + '/' + dark_files[0] + '/120617_lfp/jerk_power_Ridge.pkl','rb') as f:
        print f
        stuff = joblib.load(f)
    
    lfp_bands_within384 = np.mod(range(384),6)
    bands = np.tile(lfp_bands_within384,21)
    
    #tetrodes_within384 = np.repeat(np.arange(16),24)
    #tetrodes = np.tile(tetrodes_within384,21)
    #timebins = np.repeat(np.arange(-10,11),384)

    
    time_vec = np.arange(-10.,11.)/10


    #### make DataFrame for LFP bands:
    band_coef_dict = dict()
    for i in range(6):
        band_idx = np.where(bands == i )[0]
        band_coef_dict[lfp_names[i]] = stuff.coef_[0,band_idx] #mean_coef_byband[i]

    band_coef_frame = pd.DataFrame.from_dict(band_coef_dict)


    #### make DataFrame for time bins:
    time_coef_dict = dict()    
    for i in range(21):
        band_idx = np.where(bands == i )[0]
        time_coef_dict[time_vec[i]] = stuff.coef_[0,384*i:384*(1+i)] #mean_coef_byband[i]

    time_coef_frame = pd.DataFrame.from_dict(time_coef_dict)

    return band_coef_frame,time_coef_frame


all_files = ['636444372823550001',
'636455956910510001',
'636456486214960001',
'636462665576670001',
'636464313683750001',
             
'636459234842450001',
'636461899932110001', #'636463456945299197',
'636464513098770001',
'636465322265480001',

'636461785781685886',
'636462564120970001',
'636464402756620001',
'636465208859170001']

    
dark_files = all_files[0:5]
muscimol_files = all_files[5:9]  ## skip 636463456945299197 !!!! 
light_files = all_files[9:]


lfp_names = ['delta','theta','alpha','beta','low_gamma','high_gamma']
