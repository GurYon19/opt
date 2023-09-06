import datetime
import re
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
EPOCHS = 1
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
    #input_layer_shape = input_layer.shape
    scattering_1d = Scattering1D(J=7,T=WINDOW_SIZE,Q=3)
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

def conv_block2(inputs,input_shape =None):
    print('Applying conv block 2 ....')
    input_shape = inputs[0].shape
    conved_inputs = []
    for input in inputs:
        x = Conv1D(filters=128,
                    kernel_size = 3,
                    activation='relu',input_shape= input_shape)(input)
        conved_inputs.append(x)
    
    print('Finished conv block 2')
    return conved_inputs

def conv_block1(inputs):
    print('Applying conv block 1 ....')
    input_shape = inputs[0].shape
    conved_inputs = []
    for input in inputs:
        x = Conv1D(filters=24,
                    kernel_size = 3,
                    activation='relu',input_shape= input_shape)(input)

        x = MaxPooling1D(pool_size=3)(x)
        conved_inputs.append(x)
    
    print('Finished conv block 1')
    return conved_inputs

@keras.saving.register_keras_serializable(package="My_Layers")
class CustomLayer(keras.layers.Layer):
    def __init__(self,J):
        super().__init__()
        self.J = J
    def call(self, inputs):
        return apply_scattering_1d(inputs)

    def get_config(self):
        return {"J": self.J}

# @keras.saving.register_keras_serializable(package="my_package", name="custom_scat")
# def custom_scat(x):
#     return apply_scattering_1d(x)


@keras.utils.register_keras_serializable(package="my_package", name="f1_score")
class F1Score(keras.metrics.Metric):
    def __init__(self, name="f1_score", **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight("true_positives", initializer="zeros")
        self.false_positives = self.add_weight("false_positives", initializer="zeros")
        self.false_negatives = self.add_weight("false_negatives", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(tf.math.round(y_pred), tf.float32)

        true_positives = tf.reduce_sum(y_true * y_pred)
        false_positives = tf.reduce_sum((1 - y_true) * y_pred)
        false_negatives = tf.reduce_sum(y_true * (1 - y_pred))

        self.true_positives.assign_add(true_positives)
        self.false_positives.assign_add(false_positives)
        self.false_negatives.assign_add(false_negatives)

    def result(self):
        precision = self.true_positives / (self.true_positives + self.false_positives + tf.keras.backend.epsilon())
        recall = self.true_positives / (self.true_positives + self.false_negatives + tf.keras.backend.epsilon())

        f1 = 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())
        return f1


def create_model_2conv():
    input_shape = (WINDOW_SIZE,CHANNELS)
    input_layer = Input(shape=input_shape, name='acc_gyr')
    custom_scattering1d =CustomLayer(7)
    scatters = custom_scattering1d(input_layer)
    #scatters = apply_scattering_1d(input_layer)#for each channel
    # Create the convolutional block for the single input
    x = conv_block1(scatters)
    x = conv_block2(x)
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
    model.compile(loss='binary_crossentropy', optimizer=optz,  metrics=['accuracy', F1Score()])
    return model


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
    model.compile(loss='binary_crossentropy' , optimizer=_optimizer, metrics=[F1Score])
    return model


def _load_model(train_ds,val_ds,test_ds):
    #import load_model
    from keras.models import load_model
    #load Yonis_model.h5
    model_path = 'Yonis_model.h5'
    #load the model
    model = load_model(model_path,custom_objects={'f1_score':F1Score,'CustomLayer':CustomLayer})
    #model.summary()
    #model.fit(train_ds,epochs=1)
    #continue training
    model.fit(train_ds,
        batch_size=BATCHSIZE,
        epochs=EPOCHS,
        validation_data=val_ds,
        verbose=1,
        )
    loss,acc,f_one = model.evaluate(test_ds, verbose=0)
    print(acc)
    print(f_one)
    print(loss)

if __name__ == "__main__":
    
    data_path, batch_size,max_window_size,new_label_name,new_label_func,chosen_features ,label_name = get_inputs()
    train_ds,val_ds,test_ds = get_datasets(data_path, batch_size,max_window_size,new_label_name,new_label_func,chosen_features ,label_name)
    #train_ds,val_ds,test_ds = get_datasets(data_path)
    
    model = create_model_2conv()
    #train model and save best epoch
    # Fit the model on the training data.
    # The KerasPruningCallback checks for pruning condition every epoch.
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    model.fit(
        train_ds,
        batch_size=BATCHSIZE,
        callbacks=[ModelCheckpoint(filepath = '/tmp/checkpoint',monitor=f1_score,save_best_only=True,mode='max'),tensorboard_callback],
        epochs=EPOCHS,
        validation_data=val_ds,
        verbose=1,
    )
    loss,acc,f_one = model.evaluate(test_ds, verbose=0)
    print(f_one)
    print(loss)

    model.save('Yonis_model.h5')
    _load_model(train_ds,val_ds,test_ds)