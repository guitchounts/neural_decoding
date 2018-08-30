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
from functools import reduce

def load_npz(fil,d): ### d = dx, dy, or dz
    dx_fils = []
    for dir_item in os.listdir('./' + fil):
        if dir_item.startswith("%s_" % d):
            if dir_item.endswith(".npz"):
                print(dir_item)
                dx_fil = np.load('./%s/%s' % (fil,dir_item))
                dx_fils.append(dx_fil)
                #dx_fil.close()

    return dx_fils

def get_exp_names(all_files,rat_name):
	exp_names = []
	for fil in all_files:
	    for exp in os.listdir('./' + fil + '/'):
	        if exp.startswith('%s_0' % rat_name): ## e.g. grat47
	            exp_names.append(exp[exp.find('m_')+2:exp.find('.txt')])
	            
	exp_names = np.asarray(exp_names)

	return exp_names

if __name__ == "__main__":

	input_file_path = os.getcwd()
	all_files = []

	for file in os.listdir(input_file_path):
	        if file.startswith("636"):
	            all_files.append(file)
	all_files = np.asarray(all_files)

	rat_name = os.getcwd()[os.getcwd().find('GRat'):os.getcwd().find('GRat')+6].lower()

	exp_names = get_exp_names(all_files,rat_name)

	dark_idx = np.where(exp_names == 'dark')[0]

	light_idx = np.where(exp_names == 'light')[0]



	for condition in ['dark','light']:
	    for head_variable in ['dx','dy','dz']:

	        ## load the npz files
	        d_files = []
	        for fil in all_files[eval('%s_idx' % condition)]:    ## go thru only dark or only light files
	            d_files.append(load_npz(fil,head_variable))

	        ## make dict with X or y and left or right fields
	        d_items = {var :  {direction : [] for direction in ['left','right'] } for var in ['X', 'y']  }

	        save_dir = './Turns/%s/%s' % (head_variable,condition)
	        if not os.path.exists(save_dir):
	            os.makedirs(save_dir)
	        
	        for var in ['X', 'y']:
	            for direction in ['left','right']:
	                d_item = np.concatenate([d['%s_%s' % (var,direction)] for dx in d_files for d in dx])
	                d_items[var][direction] = d_item
	                np.save('%s/%s_%s_%s_%s.npy' % (save_dir,head_variable,condition,var,direction),d_item)
	                print(d_item.shape)
