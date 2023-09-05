import matplotlib.pyplot as plt
# from custom.data_generator import DataGenerator, get_filtered_snc_and_fsr, lin_res,get_slope
import pandas as pd
from scipy import signal
import tensorflow as tf
import numpy as np
import math
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
#from custom.data_generator import get_snc_acc_and_seg
from custom.get_values2 import get_snc_acc_and_seg
def get_acc_filter(acc):
     sos = signal.butter(4,1,'hp',output = 'sos', fs = 1125)
     acc_filter = signal.sosfilt(sos, acc, axis = 1)
     return acc_filter

def layer_output(model, x, layer_name):
    feature_extractor_output = tf.keras.Model(
        inputs=model.inputs,
        # outputs=[layer.output for layer in model.layers],
        # otputs = model.get_layer(name = 'layer1').output,
        outputs=model.get_layer(name=layer_name).output,
    )
    return feature_extractor_output(x)

def get_seg(vector, window_size):
    #vector = vector.numpy()
    a = vector[0]
    b = vector[1]
    seg = np.zeros(window_size)
    #n1 = int(window_size*(a+1)/2)
    n1 = tf.cast(window_size * (a + 1) / 2, dtype=tf.int32) #total number of '1'
    print('IKUGHKIUHK')
    print('a',a)
    print('seg',seg)
    print('n1',n1)
    if b>0:
        seg[:n1] = 1
    else:
        seg[-n1:] = 1
    return seg


def get_seg_tf(vector,window_size):
    a = vector[0]
    b = vector[1]
    
    #n1 = int(window_size*(a+1)/2)
    n1 = tf.cast(window_size * (a + 1) / 2, dtype=tf.int32)

    # Create two tensors: one filled with ones, another filled with zeros
    seg_ones = tf.ones(n1, dtype=tf.float32)
    seg_zeros = tf.zeros(window_size - n1, dtype=tf.float32)
    
    if b>0:
        seg = tf.concat([seg_ones, seg_zeros], axis=0)

    else:
        seg = tf.concat([seg_zeros, seg_ones], axis=0)

    return seg

def get_aprox_seg(vector, window_size):
    a = vector[0]
    b = vector[1]
    seg = np.zeros(window_size)
    #n1 = tf.cast(window_size * (a + 1) / 2, dtype=tf.int32)
    n1 = int(window_size*(a+1)/2)

    if n1>window_size/4:
        if b > 0:
            seg[:n1] = 1
        else:
            seg[-n1:] = 1
    if n1>3*window_size/4:
        seg[:] = 1
    return seg

def get_aprox_seg_tf(vector, window_size):
    a = vector[0]
    b = vector[1]
    
    #n1 = int(window_size*(a+1)/2)
    n1 = tf.cast(window_size * (a + 1) / 2, dtype=tf.int32)
    window_size_int32 = tf.cast(window_size, dtype=tf.int32)

    condition = n1 < tf.cast(window_size_int32 / 4, dtype=tf.int32)
    def true_fn():
        return tf.constant(0)

    def false_fn():
        return n1

    n1 = tf.cond(condition, true_fn, false_fn)

    condition = n1 > tf.cast(window_size*3 / 4, dtype=tf.int32)
    def true_fn():
        return window_size

    def false_fn():
        return n1
    n1 = tf.cond(condition, true_fn, false_fn)
    '''
    if n1 < window_size/4:
        n1 = tf.constant(0.0)
    if n1 > window_size*3/4:
        n1 = window_size   
    '''

    # Create two tensors: one filled with ones, another filled with zeros
    seg_ones = tf.ones(n1, dtype=tf.float32)
    seg_zeros = tf.zeros(window_size - n1, dtype=tf.float32)

    
    if b>0:
        seg = tf.concat([seg_ones, seg_zeros], axis=0)

    else:
        seg = tf.concat([seg_zeros, seg_ones], axis=0)

    return seg

def get_windows_and_flag(file_path, window_size_snc = 306,
                         memory_size_snc  = 216,
                         window_size_acc = 320,
                         #memory_size_acc = 220,
                         step = 100,
                         ratio = 11125/1042):

    #df = pd.read_csv(file_path)
    snc, acc, seg, file_path = get_snc_acc_and_seg(file_path)

    #print('len',snc.shape,seg.shape)
    snc_len = snc.shape[0]
    acc_len = acc.shape[0]
    max_start_point = min(snc_len-window_size_snc,int((acc_len-window_size_acc-1)/ratio))
    windowed_line = [np.arange(start_point+memory_size_snc, start_point+window_size_snc,1)
                     for start_point in range(0,max_start_point,step)]

    windowed_seg = [seg[start_point+memory_size_snc: start_point+window_size_snc]
                    for start_point in range(0,max_start_point,step)]

    windowed_snc = [snc[start_point_snc: start_point_snc+window_size_snc,:]
                    for start_point_snc in range(0,max_start_point,step)]
    windowed_acc = [acc[int(start_point_snc*ratio): int(start_point_snc*ratio) + window_size_acc,:]
                    for start_point_snc in range(0,max_start_point,step)]
    return windowed_line, windowed_snc, windowed_acc, windowed_seg


def compare_true_and_pred(file_path, models, 
                          ratio = 1125/1042,
                          window_size_snc = 306,
                         memory_size_snc  = 216,
                         window_size_acc = 320,
                         #memory_size_acc = 220,
                         step = 100,
                         model_format = 'h5',
                         colors = ['red']):
    '''
    mosels = list of models
    '''
    df = pd.read_csv(file_path)
    snc, acc, seg = get_snc_acc_and_seg(file_path)
    acc = get_acc_filter(acc)
                         
    windowed_line,windowed_snc, windowed_acc, windowed_seg = get_windows_and_flag(file_path, window_size_snc = window_size_snc,
                         memory_size_snc  = memory_size_snc,
                         window_size_acc = window_size_acc,
                         #memory_size_acc = memory_size_acc,
                         step = step, ratio=ratio)
    
    #model = models[0]
    
    plt.subplot(3, 1, 1)
    plt.title(str(file_path)[-20:])
    #plt.text(0,0,str(file_path)[-10:])

    #if model_format == 'h5':
        #predictions = [model.predict([np.expand_dims(windowed_snc[i],axis = 0),np.expand_dims(windowed_acc[i],axis = 0)])
        #                for i in range(len(windowed_line))]
    #if model_format == 'mlmodel':
    #    print('Model prediction is only supported on macOS version 10.13 or later.')    
    #pred_seg = [get_aprox_seg(pred[0],window_size_snc-memory_size_snc) for pred in predictions]


    plt.subplot(3,1,3)
    #plt.plot(windowed_line[s],windowed_seg[s],'.',linewidth = 5, color = 'black')
    for i in range(3):
        plt.plot(snc[:,i],alpha = 0.7)

    plt.plot(seg, color ='black')
    plt.legend(['snc1','snc2','snc3','seg'])
    plt.subplot(3, 1, 2)
    #plt.plot(windowed_line[s],pred_seg[s],'.')
    x = np.linspace(0,snc.shape[0],acc.shape[0])
    for i in range(3):
        plt.plot(x,acc[:,i])
    plt.subplot(3, 1, 1)
    plt.plot(np.linspace(0,memory_size_snc,memory_size_snc+1),np.zeros(memory_size_snc+1), color = 'white')
    

    for s in range(len(windowed_line)):
            plt.plot(windowed_line[s], windowed_seg[s], linewidth=3, color='black', alpha = 0.5)
    for i,model in enumerate(models):
        predictions = [model.predict([np.expand_dims(windowed_snc[i],axis = 0),np.expand_dims(windowed_acc[i],axis = 0)])
                        for i in range(len(windowed_line))]
        pred_seg = [get_aprox_seg(pred[0],window_size_snc-memory_size_snc) for pred in predictions]


    
        for s in range(len(windowed_line)):
            #plt.plot(windowed_line[s], windowed_seg[s], linewidth=3, color='black', alpha = 0.5)

           # plt.plot(windowed_line[s], pred_seg[s]*(1-i*0.05), color = colors[i%len(colors)] )
           plt.plot(windowed_line[s], pred_seg[s]-i*0.05, color = colors[i%len(colors)] )
        
        
        #print(pred_seg[s].shape)



    return plt



