import tensorflow as tf
import numpy as np
from kymatio.keras import Scattering1D
import optuna
from keras.layers import Dense,Dropout,Lambda
from keras.models import Model
from tensorflow import keras
from keras.optimizers import RMSprop
from keras.utils import to_categorical
from keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization, Concatenate
from keras.callbacks import ModelCheckpoint
from tfrecord_generator.create_ds import get_datasets,get_inputs
from models import create_model, CLASSES
from keras import backend as K



N_TRAIN_EXAMPLES = 3000
N_VALID_EXAMPLES = 1000
BATCHSIZE = 16
EPOCHS = 12
CLASSES = CLASSES
WINDOW_SIZE = 256
CHANNELS = 6


def f1_score(y_true, y_pred): 
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val


def create_conv_block(inputs,inputs_shape =None):
    print('Applying conv block....')
    conved_inputs = []
    for input in inputs:
        x = Conv1D(filters=12,
                    kernel_size = 5,
                    activation='relu')(input)
        x = MaxPooling1D(pool_size=2)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.21908)(x)
        conved_inputs.append(x)
        #convert x a to tensorflow tensor
    concat_layer = Concatenate(axis=2)
    cx = concat_layer(conved_inputs)
    print('Finished conv block')
    return cx


def apply_scattering_1d(input_layer):
    channels = Lambda(lambda x: tf.unstack(x, axis=-1))(input_layer)
    scattering_1d = Scattering1D(J=6,T=WINDOW_SIZE,Q=3)
    scatters = []
    print('Starting scattering....')
    for channel in channels:
        s = scattering_1d(channel)
        scatters.append(s)
        new_shape = s.shape
    print('Finished scattering....')
    channels = []
    for s in scatters:
        channels.append(s[:,:,0])
    channels = tf.stack(channels,axis=2)
    return scatters, new_shape
    

def create_model():
    input_shape = (WINDOW_SIZE,CHANNELS)
    input_layer = Input(shape=input_shape, name='acc_gyr')
   
    scatters, scatters_shape = apply_scattering_1d(input_layer)#for each channel
    single_input_conv_block = create_conv_block(scatters, scatters_shape)
    x = Flatten()(single_input_conv_block)
 
    # Fully connected layers
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4251386110952534)(x)

    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4278129185021247)(x)

    # Output layer
    output_layer = Dense(1, activation='sigmoid')(x)


    # Create the model with single input and one output
    model = Model(inputs=input_layer, outputs=output_layer)
    learning_rate = 0.003124287817877256
    _optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    # optimizer = RMSprop(learning_rate=learning_rate)
    model.compile(loss='binary_crossentropy' , optimizer=_optimizer, metrics=[f1_score])
    return model




if __name__ == "__main__":
    
    inputs = get_inputs()
    train_ds,val_ds,test_ds = get_datasets(inputs)

    model = create_model()
    #train model and save best epoch
    # Fit the model on the training data.
    # The KerasPruningCallback checks for pruning condition every epoch.

    model.fit(
        train_ds,
        batch_size=BATCHSIZE,
        callbacks=[ModelCheckpoint(filepath = '/tmp/checkpoint',monitor=f1_score,save_best_only=True,mode='max')],
        epochs=EPOCHS,
        validation_data=val_ds,
        verbose=1,
    )
    loss,f1_score = model.evaluate(test_ds, verbose=0)
    print(f1_score )
    print(loss)

    model.save('Yonis_model.h5')