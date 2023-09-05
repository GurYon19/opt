import tensorflow as tf
import numpy as np
from kymatio.keras import Scattering1D
#from kymatio.tensorflow import Scattering1D
import optuna
from optuna import trial
from keras.layers import Dense,Dropout,Lambda,LSTM
from keras.models import Model
from tensorflow import keras
from keras.optimizers import RMSprop
from keras.utils import to_categorical
from tfrecord_generator import create_tfrecord_dataset
from keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization, Concatenate
from create_ds import DS_WINDOW_SIZE

from custom.losses import *
from custom.callbacks import f1_fp
CLASSES = 2
WINDOW_SIZE = DS_WINDOW_SIZE
CHANNELS = 6
JBANK = [6,7,8]# number of scales
QBANK = [2,3,4] #number of wavelets per scale
TBANK = [128,256] #length
def conv_block4(trial,inputs):
    print('Applying conv block 4....')
    input_shape = inputs[0].shape
    conved_inputs = []
    for input in inputs:
        x = Conv1D(filters=trial.suggest_categorical("conv_filter4", [128]),
                        kernel_size = 3,
                activation='relu',input_shape= input_shape)(input)
        # x = Conv1D(filters=trial.suggest_categorical("filters", [8,16]),
        #             kernel_size = trial.suggest_categorical("kernel_size", [3, 5]),
        #             activation='relu')(x)
        try:
            x = MaxPooling1D(pool_size=2)(x)
        except Exception as e:
            print(e)
        conved_inputs.append(x) 
    # x = BatchNormalization()(x)
    # x = Dropout(trial.suggest_float("conv_dropout_1",0.1, 0.6))(x)
    #convert x a to tensorflow tensor
    print('Finished conv block')
    return conved_inputs
def conv_block3(trial,inputs):
    print('Applying conv block 3....')
    input_shape = inputs[0].shape
    conved_inputs = []
    for input in inputs:
        x = Conv1D(filters = trial.suggest_categorical("conv_filter3", [64,128]),
                        kernel_size = 3,
                activation='relu',input_shape= input_shape)(input)
        # x = Conv1D(filters=trial.suggest_categorical("filters", [8,16]),
        #             kernel_size = trial.suggest_categorical("kernel_size", [3, 5]),
        #             activation='relu')(x)
        try:
            x = MaxPooling1D(pool_size=trial.suggest_categorical("MaxPooling_3", [3,4]))(x)
        except Exception as e:
            print(e)
        conved_inputs.append(x) 
    # x = BatchNormalization()(x)
    # x = Dropout(trial.suggest_float("conv_dropout_1",0.1, 0.6))(x)
    #convert x a to tensorflow tensor
    print('Finished conv block')
    return conved_inputs
def conv_block2(trial,inputs,input_shape =None):
    print('Applying conv block 2 ....')
    input_shape = inputs[0].shape
    conved_inputs = []
    for input in inputs:
        #conv_fiilter2 == 128 is not bad
        x = Conv1D(filters=trial.suggest_categorical("conv_filter2", [128]),
                    kernel_size = trial.suggest_categorical("kernel_size_2", [3]),
                    activation='relu',input_shape= input_shape)(input)
        # try:
        #     x = MaxPooling1D(pool_size=trial.suggest_categorical("MaxPooling_2", [1,2,3]))(x)
        # except Exception as e:
        #     print(e)
        conved_inputs.append(x)
    
    print('Finished conv block 2')
    return conved_inputs
def conv_block1(trial,inputs):
    print('Applying conv block 1 ....')
    input_shape = inputs[0].shape
    conved_inputs = []
    for input in inputs:
        #conv_fiilter1 == 32 is not bad
        x = Conv1D(filters=trial.suggest_categorical("conv_filter1", [24]),
                    kernel_size = trial.suggest_categorical("kernel_size_1", [3]),
                    activation='relu',input_shape= input_shape)(input)
        # x = Conv1D(filters=trial.suggest_categorical("filters", [8,16]),
        #             kernel_size = trial.suggest_categorical("kernel_size", [3, 5]),
        #             activation='relu')(x)
        x = MaxPooling1D(pool_size=trial.suggest_categorical("MaxPooling_1", [3]))(x)
        conved_inputs.append(x)
        # x = BatchNormalization()(x)
        # x = Dropout(trial.suggest_float("conv_dropout_1",0.1, 0.6))(x)
        #convert x a to tensorflow tensor
    
    print('Finished conv block 1')
    return conved_inputs
def apply_scattering_1d(trial,input_layer,input_shape):
    channels = Lambda(lambda x: tf.unstack(x, axis=-1))(input_layer)
    #input_layer_shape = input_layer.shape
    scattering_1d = Scattering1D(J=trial.suggest_categorical("J", JBANK),T=trial.suggest_categorical("T", TBANK),Q=trial.suggest_categorical("Q", QBANK))
    #scatters = scattering_1d(input_layer)
    scatters = []
    print('Starting scattering....')
    for channel in channels:
        s = scattering_1d(channel)
        scatters.append(s)
    print('Finished scattering....')
    # channels = []
    # for s in scatters:
    #     channels.append(s[:,:,0])
    # channels = tf.stack(channels,axis=2)
    return scatters
    
#
def create_model(trial):
    input_shape = (WINDOW_SIZE,CHANNELS)
    input_layer = Input(shape=input_shape, name='acc_gyr')
    scatters = apply_scattering_1d(trial,input_layer,input_shape)#for each channel
    # Create the convolutional block for the single input
    x = conv_block1(trial,scatters)
    x = conv_block2(trial,x)
    #x = conv_block3(trial,x)
    #x = conv_block4(trial,x)
    concat_layer = Concatenate(axis=2)
    x= concat_layer(x)
    #x= conv_block3(trial,x)
    

    x = Flatten()(x)
    x = Dense(512,activation='relu')(x)
    x = Dense(256,activation='relu')(x)
    x = Dense(128,activation='relu')(x)
    x = Dense(32,activation='relu')(x)
    x = Dense(8,activation='relu')(x)
    # Output layer
    output_layer = Dense(1,activation='sigmoid')(x)


    # Create the model with single input and one output
    model = Model(inputs=input_layer, outputs=output_layer)
    learning_rate = 0.0035
    # optimizer = RMSprop(learning_rate=learning_rate)
    optz = keras.optimizers.Adam(learning_rate=learning_rate)
    #try to use hinge,focal,logistic loss
    model.compile(loss='binary_crossentropy', optimizer=optz, metrics=[f1_score])
    return model




 