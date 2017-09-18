from scipy.interpolate import splev, splrep




head_variables = np.vstack([head_data.ox,head_data.oy,head_data.oz,head_data.ax,head_data.ay,head_data.az]).T

head_names = ['ox','oy','oz','ax','ay','az']


start = np.where(np.isclose(lfp_time,head_data.converted_times.iloc[0],atol=1e-1))[0][1]
stop = np.where(np.isclose(head_data.converted_times,lfp_time[-1],atol=1e-1))[0][-1]


truncated_head_data = head_variables[0:stop,:]


interp_head = np.empty([interp_time_axis.shape[0],truncated_head_data.shape[1]])
for i in range(truncated_head_data.shape[1]):

    interp_head[:,i] = get_interp_data(truncated_head_data[:,i],head_data.converted_times.iloc[0:stop],interp_time_axis)


decimated_head = signal.decimate(interp_head.T,10,zero_phase=True).T


def get_interp_data(data,times_original,times_interp):

    f_splrep = interpolate.splrep(times_original,data)

    splrep = interpolate.splev(times_interp, f_splrep)
    return splrep


#### get the derivatives of head angles:

dz = signal.decimate(np.gradient(interp_head[:,2]),10,zero_phase=True)

dy = signal.decimate(np.gradient(interp_head[:,1]),10,zero_phase=True)

dx = np.zeros(len(interp_head[:,0]))

for idx in range(1,len(interp_head[:,0])):
    
    # left movements:
    
    tmp_diff = interp_head[idx,0] - interp_head[idx-1,0]
    if tmp_diff > 180:
        tmp_diff -= 360
    elif tmp_diff < -180:
        tmp_diff += 360
    dx[idx] = tmp_diff

dx = signal.decimate(dx,10,zero_phase=True)