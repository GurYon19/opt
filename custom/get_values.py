import numpy as np
import pandas as pd

def get_label(window_seg):
    ws = window_seg.shape[0]
    float_ones = np.ones(ws)
    float_ones[:] = 1.0
    try:
        norm_window_seg = 2*window_seg-float_ones # replace 0 by -1

    except:
        return np.array([0, 1])

    a = np.sum(norm_window_seg)/ws
    b = norm_window_seg[0] * np.sqrt(1-a**2)
    return np.array([a,b])

'''
def get_labeled_data(df):
    
    # SNC
    snc1, snc2, snc3 = df.Ulnar.dropna().to_numpy(), df.Median.dropna().to_numpy(), df.Radial.dropna().to_numpy()
    snc = np.transpose(np.vstack([snc1, snc2, snc3]))

    # IMU (acc)
    acc1, acc2, acc3 = df.RAW1.dropna().to_numpy(), df.RAW2.dropna().to_numpy(), df.RAW3.dropna().to_numpy()
    acc = np.transpose(np.vstack([acc1, acc2, acc3]))

    seg = df.SNC_FLAG.dropna().to_numpy()
    print('file wrong type', type(seg))
    seg = 2*seg - np.ones(len(seg)) # replace 0 by -1
    
    return snc, acc, seg
'''    


def get_snc_acc_and_seg(path):
    df = pd.read_csv(path)
    snc1, snc2, snc3 = df.Ulnar.dropna().to_numpy(), df.Median.dropna().to_numpy(), df.Radial.dropna().to_numpy()
    
    #df['SNC_FLAG'] = df['SNC_FLAG'].astype('int')
    seg = df.SNC_FLAG.dropna().to_numpy()
    #seg = np.array([float(segm) for segm in seg])

    acc1, acc2, acc3 = df.RAW1.dropna().to_numpy(), df.RAW2.dropna().to_numpy(), df.RAW3.dropna().to_numpy()
    try:
        snc = np.vstack((snc1, snc2, snc3), dtype=np.float32)
        acc = np.vstack((acc1,acc2,acc3), dtype=np.float32)
    except:
        print('BAD FILE ' + path)
        
    snc = np.transpose(snc)
    acc = np.transpose(acc)
    return snc, acc, seg


def tap_impact(x,c = 50):
    if x<0:
        result = max(0,(x+3)/3)
    else:
        result =  1 - np.tanh(x/c)   
    return  result

def get_file_tap_function(file_path):

    # Vectorize the tap_impact function
    tap_impact_vec = np.vectorize(tap_impact, otypes=[np.float64])

    snc, acc, seg = get_snc_acc_and_seg(file_path)
    len_seg = seg.shape[0]
    x = np.linspace(0,len_seg-1,len_seg)
    sum_tap = np.zeros(len_seg)
    for i in range(len_seg-1):
        if [seg[i],seg[i+1]] == [0,1]:
            sum_tap+=tap_impact_vec(x-i-1)

    return sum_tap        