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

def get_exp_names(rat,all_files):
    exp_names = []
    for fil in all_files:
        for exp in os.listdir('./' + fil + '/'):
            if exp.startswith('%s_1' % rat):
                exp_names.append(exp[exp.find('m_')+2:exp.find('.txt')])

    return np.asarray(exp_names)

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


    # pitch_name = """['pitch_abs']_results.npz"""
    # roll_name = """['roll_abs']_results.npz"""
    # yaw_name = """['yaw_abs']_results.npz"""

    rat = 'grat28'
    res_folder =   '/060518_ridge_mua/' # '/031718_ridge_mua/' # ## ' #/031918_mua/' # '/022618_mua/' # '/031618_mua/' #'/031718_ridge_mua/'  # '/022618_mua/' # #

    all_files = get_all_files(res_folder)
    exp_names = get_exp_names(rat,all_files)
    print(all_files,exp_names)
    
    # exp_info = pd.DataFrame.from_csv('111817_mua_files.csv')
    # all_files = exp_info['Files'].values
    # exp_names = exp_info['Conditions'].values
    # res_folder = exp_info['Experiment'][0] + '/'
    # print all_files,exp_names,res_folder
    
    results_normal_names = ['pitch_abs', 'roll_abs','yaw_abs', 'total_acc']
    results_file_names = ["""['pitch_abs']_results.npz""",  """['roll_abs']_results.npz""", """['yaw_abs']_results.npz""", """['total_acc']_results.npz"""]
    file_keys = ['y_valids','y_hats']


    pearson_dict = {results_normal_name : [] for results_normal_name in results_normal_names }
    r2_dict = {results_normal_name : [] for results_normal_name in results_normal_names }


    for i,fil in enumerate(all_files): #### all_files !!!!!
        
        for res in results_normal_names:
            print(type(fil),type(res_folder),type(res))
            tmp_file_dir = './' +str(fil) + res_folder + """['%s']_results.npz""" %  res
            
            if os.path.exists(tmp_file_dir):
                tmp_fil = np.load(tmp_file_dir)
                        
                #print fil,res, len(results_dict[res]),tmp_fil
                tmp_pearson = stats.pearsonr(tmp_fil[file_keys[0]],tmp_fil[file_keys[1]])[0]
                
                if np.any(np.isnan(tmp_fil[file_keys[0]])) == True:
                    tmp_r2 = np.array([0.0])
                else:               
                    tmp_r2 = np.array([metrics.r2_score(tmp_fil[file_keys[0]],tmp_fil[file_keys[1]])])
                

                


                tmp_fil.close()
            
            else:
                print('skipping file ', tmp_file_dir)
                tmp_r2 = np.array([0.0])
                tmp_pearson = np.array([0.0])

            pearson_dict[res].append(tmp_pearson)
                
            r2_dict[res].append(tmp_r2)



    for res in results_normal_names:
        pearson_dict[res] = np.concatenate(pearson_dict[res])
        r2_dict[res] = np.concatenate(r2_dict[res])



    pearson_frame = pd.DataFrame.from_dict(pearson_dict)
    r2_frame = pd.DataFrame.from_dict(r2_dict)

    pearson_frame['exp_names'] = exp_names
    r2_frame['exp_names'] = exp_names

    pearson_frame = pearson_frame.melt(id_vars='exp_names',value_name='Pearson')
    r2_frame = r2_frame.melt(id_vars='exp_names',value_name='R2')
    all_data_frame = pearson_frame 
    all_data_frame['R2'] = r2_frame['R2']


    hue_order = ['dark','light','dark_muscimol','light_muscimol']
    new_dark_pal = [sns.color_palette("Dark2", 3)[2],
                    sns.color_palette("Dark2", 3)[0],
                    sns.color_palette("Dark2", 3)[1],
                   "#3498db"
                   ]

    f= plt.figure(dpi=600)
    gs = gridspec.GridSpec(2, 1)

    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[1, 0])
    axs = [ax1,ax2]
        
    plot_keys = ['Pearson', 'R2']
    for i in range(2):
        
        g = sns.barplot(x='variable',y=plot_keys[i],hue='exp_names',ci=68,errwidth=1,capsize=.1,
                    hue_order=hue_order, data=all_data_frame,ax=axs[i],palette=new_dark_pal)
        #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        g.legend_.remove()
        sns.despine(bottom=True,offset=5)
    g.axes.legend(loc=(0,-.25),  ncol=4) #bbox_to_anchor=(1.3, 0.5)

    f.savefig('./decoding_results/%s.pdf' % res_folder[1:-1])


    all_data_frame.to_csv('./decoding_results/%s.csv' % res_folder[1:-1])


