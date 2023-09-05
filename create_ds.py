import os 
from tfrecord_generator.create_tfrecord_dataset import create_tfrecord_dataset 
from tfrecord_generator.dg_utils.label_utils import two_d_to_one_d_label

DS_BATCH_SIZE = 16
DS_WINDOW_SIZE = 224
def get_inputs():
    curr_dir = os.getcwd()
    data_path = os.path.join(curr_dir,'Data')
    max_window_size = DS_WINDOW_SIZE # based on the column with most rows in the CSV.

    batch_size=DS_BATCH_SIZE

    #list of features to be extracted from the csv INCLUDING LABEL
    feature_list = ['Gyro1','Gyro2','Gyro3','Acc1','Acc2','Acc3','GyrButton','AccButton']  #MUST BE A LIST 
    label_name = ['GyrButton','AccButton'] #name of the label column. MUST BE A LIST 
    
    new_label_func = [two_d_to_one_d_label] #the label calculation will be when calculating the window. MUST RETURN A LIST OR NONE
    new_label_name = None if new_label_func is None else ['label'] #name of the new label column.  MUST BE A LIST OR NONE
    assert len(new_label_name) == len(new_label_func), f'length of label_name is {len(new_label_name)} and new_label_func is {len(new_label_func)} should be equal'
    chosen_features = feature_list # 
    #assert label_name in chosen_features, f'label name {label_name} should be in the chosen features'
    return data_path, batch_size,max_window_size,chosen_features ,label_name,new_label_name,new_label_func



def get_train_val_test_dataset():
    data_path, batch_size,max_window_size,chosen_features ,label_name,new_label_name,new_label_func = get_inputs()
    print("\n Starting TFRecord Dataset creation.....\n ")
    train_dataset,val_dataset, test_dataset,_ = create_tfrecord_dataset(data_path, batch_size,max_window_size,new_label_name,new_label_func,chosen_features ,label_name)
    print("\n Finished TFRecord Dataset.....\n ")
    return train_dataset,val_dataset, test_dataset 

if __name__ == "__main__":
    get_train_val_test_dataset()