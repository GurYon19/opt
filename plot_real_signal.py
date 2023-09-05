

import numpy as np
import os
import scipy.io.wavfile
import pandas as pd

from scipy import signal

import matplotlib.pyplot as plt
import kymatio
from kymatio.numpy import Scattering1D
from kymatio.datasets import fetch_fsdd
def plot_signal(path,J,Q):
        
    df = pd.read_csv(path)
    
    upper_bound =4700   
    lower_bound = 4500
    data_len_nominal = upper_bound - lower_bound
    x = df.Acc2.to_numpy()[lower_bound :upper_bound]
    ###############################################################################
    # Once the recording is in memory, we normalize it.

    ###############################################################################
    # We are now ready to set up the parameters for the scattering transform.
    # First, the number of samples, `T`, is given by the size of our input `x`.
    # The averaging scale is specified as a power of two, `2**J`. Here, we set
    # `J = 6` to get an averaging, or maximum, scattering scale of `2**6 = 64`
    # samples. Finally, we set the number of wavelets per octave, `Q`, to `16`.
    # This lets us resolve frequencies at a resolution of `1/16` octaves.


    T = x.shape[-1]
    
    
    Sx = None
    ###############################################################################
    # Finally, we are able to create the object which computes our scattering
    # transform, `scattering`.

    scattering = Scattering1D( J,T, Q, average=False, out_type='dict', max_order=2)

    ###############################################################################
    # Compute and display the scattering coefficients
    # -----------------------------------------------
    # Computing the scattering transform of a signal is achieved using the
    # `__call__` method of the `Scattering1D` class. The output is an array of
    # shape `(C, T)`. Here, `C` is the number of scattering coefficient outputs,
    # and `T` is the number of samples along the time axis. This is typically much
    # smaller than the number of input samples since the scattering transform
    # performs an average in time and subsamples the result to save memory.

    Sx = scattering(x)

    ###############################################################################
    # To display the scattering coefficients, we must first identify which belong
    # to each order (zeroth, first, or second). We do this by extracting the `meta`
    # information from the scattering object and constructing masks for each order.

    meta = scattering.meta()
    order0 = np.where(meta['order'] == 0)
    order1 = np.where(meta['order'] == 1)
    order2 = np.where(meta['order'] == 2)

    ###############################################################################
    # We now plot the zeroth-order scattering coefficient, which is simply an
    # average of the original signal at the scale `2**J`.
    fig = plt.figure(figsize=(8, 8))
    gs = fig.add_gridspec(4, 1)
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(x,'r-')
    ax.set_title('Original signal')
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(Sx[()],'b-')
    ax.set_title('Zeroth-order scattering')
    
    ###############################################################################
    # We then plot the first-order coefficients, which are arranged along time
    # and log-frequency.
    o1_expanded_bands = []
    o2_expanded_bands = []
    for band,value in Sx.items():
        if len(band) ==2:
            data_len = len(value)
            o2_expanded_bands.append(np.expand_dims(signal.resample(value, data_len_nominal), axis=0))
        else:
            data_len = len(value)
            o1_expanded_bands.append(np.expand_dims(signal.resample(value, data_len_nominal), axis=0))
        #i+=1
    S1 = np.concatenate(tuple(o1_expanded_bands), axis=0)[1:, :]
    try:
        S2 = np.concatenate(tuple(o2_expanded_bands), axis=0)[1:, :]
    except Exception as e:
        print(e)
        S2 = np.zeros(S1.shape)

    ax = fig.add_subplot(gs[2, 0])
    ax.plot(S1,'g-')
    ax.set_title('First-order scattering')
    plt.imshow(S1, aspect='auto')
    plt.title(f'First-order scattering.  J is {J}, Q is {Q}')
    ax = fig.add_subplot(gs[3, 0])
    ax.plot(S2,'r-')
    ax.set_title('Second-order scattering')
    plt.imshow(S2, aspect='auto')


    plt.tight_layout()
    plt.show()

    #info_dataset = fetch_fsdd(verbose=True)

for J in [2,4,8]:
    for Q in [1,2]:
        #file_path = '/media/wld/HDD/Data_2023/DB_AirTouch_v3/LeeorLanger /tapCycle_0.csv'
        #plot_signal(file_path,J,Q)
        noise_file_path = 'Data/Train/CSV/8tapX2_11.csv'
        plot_signal(noise_file_path,J,Q)