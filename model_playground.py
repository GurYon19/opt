from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import tensorboard
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
from create_ds import DS_BATCH_SIZE, DS_WINDOW_SIZE, get_train_val_test_dataset

from custom.losses import *
from custom.callbacks import f1_fp
from keras.models import load_model

train_ds,val_ds,test_ds = get_train_val_test_dataset()

CLASSES = 2
WINDOW_SIZE = DS_WINDOW_SIZE

N_TRAIN_EXAMPLES = 3000
N_VALID_EXAMPLES = 1000
# BATCHSIZE = 1 
BATCHSIZE = DS_BATCH_SIZE
EPOCHS = 1


CHANNELS = 6
JBANK = 7 #7
QBANK = 4 # 4 number of wavelets per scale
TBANK = 128 #length
def conv_block4(inputs):
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
def conv_block3(inputs):
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
def conv_block2(inputs):
    print('Applying conv block 2 ....')
    input_shape = inputs[0].shape
    conved_inputs = []
    for input in inputs:
        #conv_fiilter2 == 128 is not bad
        x = Conv1D(filters=128,
                    kernel_size = 3,
                    activation='relu',input_shape= input_shape)(input)
        conved_inputs.append(x)
    
    print('Finished conv block 2')
    return conved_inputs
def conv_block1(inputs):
    #print('Applying conv block 1 ....')
    input_shape = inputs[0].shape
    conved_inputs = []
    for input in inputs:
        #conv_fiilter1 == 32 is not bad
        x = Conv1D(filters=24,
                    kernel_size = 3,
                    activation='relu',input_shape= input_shape)(input)
        # x = Conv1D(filters=trial.suggest_categorical("filters", [8,16]),
        #             kernel_size = trial.suggest_categorical("kernel_size", [3, 5]),
        #             activation='relu')(x)
        x = MaxPooling1D(pool_size=3)(x)
        conved_inputs.append(x)
        # x = BatchNormalization()(x)
        # x = Dropout(trial.suggest_float("conv_dropout_1",0.1, 0.6))(x)
        #convert x a to tensorflow tensor
    
    #print('Finished conv block 1')
    return conved_inputs
def apply_scattering_1d(input_layer):
    channels = Lambda(lambda x: tf.unstack(x, axis=-1))(input_layer)
    #input_layer_shape = input_layer.shape
    scattering_1d = Scattering1D(JBANK,T=TBANK,Q=QBANK)
    #scatters = scattering_1d(input_layer)
    scatters = []
    print('Starting scattering....')
    for channel in channels:
        s = scattering_1d(channel)
        scatters.append(s)
    print('Finished scattering....')
    return scatters
    
def plot_pca(features,labels):
    #reduce features dim from (16,1,128) to (16,128). features is a list
    features = np.array(features)
    labels = np.array(labels)
    features = features.squeeze(axis=1)
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(features)
    plt.figure()
    colors = ['b', 'r']
    target_names = [0, 1]

    for color, i, target_name in zip(colors, [0, 1], target_names):
        plt.scatter(principal_components[labels == i, 0], 
                    principal_components[labels == i, 1], 
                    color=color, 
                    label=target_name)

    plt.legend(loc='best')
    plt.title('PCA of Dataset')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()


def create_model():
    input_shape = (WINDOW_SIZE,CHANNELS)
    input_layer = Input(shape=input_shape, name='acc_gyr')
    scatters = apply_scattering_1d(input_layer)#for each channel
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
    model = Model(inputs=scatters, outputs=output_layer)
    learning_rate = 0.0035
    # optimizer = RMSprop(learning_rate=learning_rate)
    optz = keras.optimizers.Adam(learning_rate=learning_rate)
    #try to use hinge,focal,logistic loss
    model.compile(loss='binary_crossentropy', optimizer=optz, metrics=[f1_score])
    return model

def extract_features(tfrecord_dataset,feature_extractor):
    #iterate over tfrecord_dataset and extract features
    # iterator = tfrecord_dataset.make_one_shot_iterator()
    # features = iterator.get_next()
    # loop on tfrecord_dataset and extract using tfrecord_dataset.take()
    features_list = []
    labels_list = [] 
    j = 0
    while j//BATCHSIZE < 8:
        try:
            one_batch_dataset = tfrecord_dataset.take(1)
            for features_batch, labels_batch in one_batch_dataset:
                # batch is now a single batch from your dataset
                # batch['feature_name'] will be a tensor containing the 'feature_name' feature for all elements in this batch
                for i in range(BATCHSIZE):
                    # For each element in the batch, extract and process the features as needed
                    single_element_features = features_batch[i]
                    #turn single_element_features with shape (256,6) to (None,256,6)
                    single_element_features = tf.expand_dims(single_element_features, axis=0)
                    features = feature_extractor(single_element_features)
                    features = features.numpy()
                    single_element_label = labels_batch[i]
                    label  = single_element_label.numpy()
                    #convert label to int
                    label = int(label)
                    features_list.append(features)
                    labels_list.append(label)

        except tf.errors.OutOfRangeError:
            break
        j = j+ 1
    return features_list,labels_list



if __name__ == "__main__":

    #save model with best trial
    # model = create_model()
    
    # logdir = './logs'
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    # model.fit(
    #     train_ds,
    #     batch_size=BATCHSIZE,
    #     epochs=EPOCHS,
    #     # callbacks=[tensorboard_callback,tensorboard_callback(trial, monitor = f1_fp)],
    #     validation_data=val_ds,
    #     verbose=10,
    # )
    # # Evaluate the model accuracy on the validation set.
    # #f1_score,fp = model.evaluate(test_ds, verbose=0)
    # model.save('just_model.h5')
    model  = load_model('just_model.h5',custom_objects={'Scattering1D':Scattering1D},compile=False)
    loss,f1 = model.evaluate(test_ds, verbose=0)
    feature_extractor = Model(inputs=model.input, outputs=model.layers[-4].output)
    features_list,labels_list = extract_features(test_ds,feature_extractor) #keys are features, values are labels
    
    plot_pca(features_list,labels_list)
    print('f1_score: ',f1)
    # model.save('Yonis_best_model.h5')
    # from keras.models import load_model
    # model_path = 'Yonis_model.h5'
    # #load the model
    # model = load_model(model_path)




 