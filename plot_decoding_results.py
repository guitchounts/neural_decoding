
# coding: utf-8

# In[1]:

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
plt.rcParams['pdf.fonttype'] = 'truetype'
from sklearn import metrics

def get_all_files(res_folder):
    input_file_path = os.getcwd()
    all_files = []

    for file in os.listdir(input_file_path):
            if file.startswith("636"):
                if os.path.exists(file + res_folder):
                    if os.listdir(file + res_folder) !=[]:
                        all_files.append(file)


    return np.asarray(all_files)

if __name__ == "__main__":

    ##### This script is specifically for updating the yaw signal w/ complex value decoding results.
    ##### Run decoding_results_plot.py for the regular saving of results as a csv file.


    input_file_path = os.getcwd()
    print(input_file_path)
    rat = input_file_path[input_file_path.find('GRat'):input_file_path.find('GRat')+6]
    res_folder = sys.argv[1] # '/082118_ridge_mua/'
    #res_folder_yaw = sys.argv[2] # '/101718_ridge_mua/'

    all_files = []

    for file in os.listdir(input_file_path):
            if file.startswith("636"):
                all_files.append(file)

    all_files = np.asarray(all_files)




    results_normal_names = ['pitch_abs', 'roll_abs','yaw_abs', 'total_acc']
    results_file_names = ["""['pitch_abs']_results.npz""",  """['roll_abs']_results.npz""", """['yaw_abs']_results.npz""", """['total_acc']_results.npz"""]
    file_keys = ['y_valids','y_hats']

    results_normal_names_yaw = ['yaw_real', 'yaw_imag']
    results_file_names_yaw = ["""['yaw_real']_results.npz""",  """['yaw_imag']_results.npz"""]
    r2_dict_yaw = {results_normal_name : [] for results_normal_name in results_normal_names_yaw }
    pearson_dict_yaw = {results_normal_name : [] for results_normal_name in results_normal_names_yaw }


    pearson_dict = {results_normal_name : [] for results_normal_name in results_normal_names }
    r2_dict = {results_normal_name : [] for results_normal_name in results_normal_names }







    experiment_paths = []
    experiment_paths_yaw = []

    exp_names = []

    print('RAT = ', rat)

    for i,fil in enumerate(all_files):
        print('Going through %s' % fil)
        for exp in os.listdir('./' + fil + '/'): ### e.g. ./636596531772835142
            if exp.startswith('%s_0' % rat.lower()):
                exp_name = exp[exp.find('m_')+2:exp.find('.txt')]
                print('Found exp_name = ', exp_name)
            elif exp.startswith('%s_1' % rat.lower()):
                exp_name = exp[exp.find('m_')+2:exp.find('.txt')]
                print('Found exp_name = ', exp_name)

        for chunk in  os.listdir('./' + fil + '/'  + res_folder + '/'): # + '/' + 
            if len(chunk) < 2: ### i.e. check for short things like 0's, 1's etc; ignore "ds_store" and xxx'd folders

                print('./' + fil + res_folder + chunk + '/',exp_name)
                experiment_paths.append('./' + fil + res_folder + chunk + '/')
                #experiment_paths_yaw.append('./' + fil + res_folder_yaw + chunk + '/')
        
                exp_names.append(exp_name)

    exp_names = np.asarray(exp_names)
    experiment_paths = np.asarray(experiment_paths)
    experiment_paths_yaw = np.asarray(experiment_paths_yaw)

    print('Gathered the following files and exp names: ', all_files,exp_names)
    print('experiment_paths_yaw =', experiment_paths_yaw)
    print('experiment_paths =', experiment_paths)


    # for i,fil in enumerate(experiment_paths_yaw): #### all_files !!!!!
    #     #print fil
    #     #for hem in hemispheres:
    #     #print ('going thru experiment_paths_yaw', fil)
    #     for res in results_normal_names_yaw:
    #         tmp_fil = np.load(fil + '/' + """['%s']_results.npz""" %  res)
    #         #print(fil + res_folder_yaw + """['%s']_results.npz""" %  res)

    # #             #for key in file_keys:
    # #             #print fil,res, len(results_dict[res]),tmp_fil
    #         tmp_pearson = stats.pearsonr(tmp_fil[file_keys[0]],tmp_fil[file_keys[1]])[0]
    #         pearson_dict_yaw[res].append(tmp_pearson) # ['Pearson']

    #         if np.any(np.isnan(tmp_fil[file_keys[0]])) == True:
    #             tmp_r2 = 0.0#np.array([0.0])
    #         else:
    #             tmp_r2 = metrics.r2_score(tmp_fil[file_keys[0]],tmp_fil[file_keys[1]]) #np.array([metrics.r2_score(tmp_fil[file_keys[0]],tmp_fil[file_keys[1]])])
    #         r2_dict_yaw[res].append(tmp_r2)

    #         #print fil,res_folder,res, file_keys
    #         #results_dict[res].append(stats.pearsonr(tmp_fil[file_keys[0]], tmp_fil[file_keys[1]]))
    #         tmp_fil.close()


    #print('r2_dict_yaw = ', r2_dict_yaw)
    # In[16]:

    for i,fil in enumerate(experiment_paths): #### all_files !!!!!
        #print fil
        #for hem in hemispheres:

        for res in results_normal_names:
            tmp_fil = np.load(fil + '/' + """['%s']_results.npz""" %  res)
            print(fil + res_folder + """['%s']_results.npz""" %  res)

    #             #for key in file_keys:
    #             #print fil,res, len(results_dict[res]),tmp_fil
            tmp_pearson = stats.pearsonr(tmp_fil[file_keys[0]],tmp_fil[file_keys[1]])[0]
            pearson_dict[res].append(tmp_pearson) # ['Pearson']

            if np.any(np.isnan(tmp_fil[file_keys[0]])) == True:
                tmp_r2 = 0.0#np.array([0.0])
            else:
                tmp_r2 = metrics.r2_score(tmp_fil[file_keys[0]],tmp_fil[file_keys[1]]) #np.array([metrics.r2_score(tmp_fil[file_keys[0]],tmp_fil[file_keys[1]])])
            r2_dict[res].append(tmp_r2)

            #print fil,res_folder,res, file_keys
            #results_dict[res].append(stats.pearsonr(tmp_fil[file_keys[0]], tmp_fil[file_keys[1]]))
            tmp_fil.close()

    print('r2_dict = ', r2_dict)

    #for hem in hemispheres:
    # for res in results_normal_names_yaw:
    #     print(res,len(r2_dict_yaw[res])) #,len(results_dict[res]['R2'])
    #     pearson_dict_yaw[res] = np.asarray(pearson_dict_yaw[res])
    #     r2_dict_yaw[res] = np.asarray(r2_dict_yaw[res])
    #     #results_dict[res]['R2'] = np.concatenate(results_dict[res]['R2'])


    # In[21]:

    #for hem in hemispheres:
    for res in results_normal_names:
        print(res,len(pearson_dict[res])) #,len(results_dict[res]['R2'])
        pearson_dict[res] = np.asarray(pearson_dict[res])
        r2_dict[res] = np.asarray(r2_dict[res])
        #results_dict[res]['R2'] = np.concatenate(results_dict[res]['R2'])


    pearson_dict_new = dict(pearson_dict)
    #pearson_dict_new.update(pearson_dict_yaw)

    r2_dict_new = dict(r2_dict)
    #r2_dict_new.update(r2_dict_yaw)


    pearson_frame = pd.DataFrame.from_dict(pearson_dict_new)
    r2_frame = pd.DataFrame.from_dict(r2_dict_new)
    print('pearson_frame.shape,len(exp_names) = ', pearson_frame.shape,len(exp_names))
    print('r2_frame.shape,len(exp_names) = ', r2_frame.shape,len(exp_names))
    pearson_frame['exp_names'] = exp_names
    r2_frame['exp_names'] = exp_names


    pearson_frame = pd.melt(pearson_frame,id_vars='exp_names',value_name='Pearson')
    r2_frame = pd.melt(r2_frame,id_vars='exp_names',value_name='R2')


    all_data_frame = pearson_frame #pd.concat([pearson_frame,r2_frame],axis=1,keys=['Pearson','R2'])
    all_data_frame['R2'] = r2_frame['R2']



    all_data_frame.to_csv('%s.csv' % res_folder[1:-1])




    hue_order = ['dark','light']
    new_dark_pal = [sns.color_palette("Dark2", 3)[2],
                    sns.color_palette("Dark2", 3)[0],
    #                 sns.color_palette("Dark2", 3)[1],
    #                "#3498db"
                   ]




    f= plt.figure(dpi=600)
    gs = gridspec.GridSpec(2, 1)

    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[1, 0])
    axs = [ax1,ax2]

    plot_keys = ['Pearson', 'R2']
    for i in range(2):

        g = sns.barplot(x='variable',y=plot_keys[i],hue='exp_names', ci=68,capsize=.1,#errwidth=1
                    #order = ['yaw_abs','yaw_real','yaw_imag'],
                        hue_order=hue_order, data=all_data_frame,
                        palette=new_dark_pal,ax=axs[i])
        #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        g.legend_.remove()
        sns.despine(bottom=True,offset=5)
    g.axes.legend(loc=(0,-.25),  ncol=4) #bbox_to_anchor=(1.3, 0.5)

    f.savefig('%s_decoding_results.pdf' % rat)
