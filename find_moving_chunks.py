def group_consecutives(vals, step=1):
    """Return list of consecutive lists of numbers from vals (number list)."""
    run = []
    result = [run]
    expect = None
    for v in vals:
        if (v == expect) or (expect is None):
            run.append(v)
        else:
            run = [v]
            result.append(run)
        expect = v + step
    return result


def get_moving_resting_chunks(neural_data,chunks_idx):
print 'neural_data.shape = ', neural_data.shape
all_lfps = []
 

for thing in chunks_idx:    
    all_lfps.append(neural_data[thing[0]:thing[-1],:])

    
return np.concatenate([chunked_lfp for chunked_lfp in all_lfps])


resting_light = np.where(total_acc_light < 1)[0]
moving_light = np.where(total_acc_light > 1)[0]

resting_dark = np.where(total_acc_dark < 1)[0]
moving_dark = np.where(total_acc_dark > 1)[0]


moving_chunks_idx_light = group_consecutives(moving_light)
resting_chunks_idx_light  = group_consecutives(resting_light)

moving_chunks_idx_dark = group_consecutives(moving_dark)
resting_chunks_idx_dark  = group_consecutives(resting_dark)


moving_lfps_light = get_moving_resting_chunks(lfp_power_light,moving_chunks_idx_light)
moving_lfps_dark = get_moving_resting_chunks(lfp_power_dark,moving_chunks_idx_dark)
resting_lfps_light = get_moving_resting_chunks(lfp_power_light,resting_chunks_idx_light)
resting_lfps_dark = get_moving_resting_chunks(lfp_power_dark,resting_chunks_idx_dark)