import numpy as np
from custom.get_values import get_label, get_snc_acc_and_seg
from custom.compare import get_seg, get_aprox_seg, get_seg_tf, get_aprox_seg_tf
import tensorflow as tf

def mismatch(seg1,seg2):
    if (np.all(seg1 == 0) and np.all(seg2 == 1)) or (np.all(seg1 == 1) and np.all(seg2 == 0)):
        return 1
    return 0

def mismatch_tf(seg1,seg2):
    condition = (
    (tf.reduce_all(tf.equal(seg1, 0)) and tf.reduce_all(tf.equal(seg2, 1))) or 
    (tf.reduce_all(tf.equal(seg1, 1)) and tf.reduce_all(tf.equal(seg2, 0))))
    if condition:
        return 1.0
    return 0.0

def mismatch_1_tf(seg1,seg2):
    ''' returns 1  seg1 is equal to 0 or 1 everywhere and an other one not'''
    condition = (
    (tf.reduce_all(tf.equal(seg1, 0)) and not tf.reduce_all(tf.equal(seg2, 0))) or 
    (tf.reduce_all(tf.equal(seg1, 1)) and  not tf.reduce_all(tf.equal(seg2, 1))))
    if condition:
        return 1.0
    return 0.0

def mean_mismatch_tf(true_seq, pred_seq):
    cond = tf.not_equal(tf.shape(true_seq)[0], tf.shape(pred_seq)[0])
    def true_fn():
        return tf.constant(0, dtype=tf.float32)
    
    def false_fn():
        # Calculate the sum of mismatches
        mismatches = tf.map_fn(lambda i: mismatch_tf(true_seq[i], pred_seq[i]), tf.range(tf.shape(true_seq)[0]), dtype=tf.float32)

        # Calculate the average of mismatches
        average_mismatch = tf.reduce_sum(mismatches) / tf.cast(tf.shape(true_seq)[0], dtype=tf.float32)
        return average_mismatch
    result = tf.cond(cond, true_fn, false_fn)
    
    return result

def mean_mismatch_1_tf(true_seq, pred_seq):
    cond = tf.not_equal(tf.shape(true_seq)[0], tf.shape(pred_seq)[0])
    def true_fn():
        return tf.constant(0, dtype=tf.float32)
    
    def false_fn():
        # Calculate the sum of mismatches
        mismatches = tf.map_fn(lambda i: mismatch_1_tf(true_seq[i], pred_seq[i]), tf.range(tf.shape(true_seq)[0]), dtype=tf.float32)

        # Calculate the average of mismatches
        average_mismatch = tf.reduce_sum(mismatches) / tf.cast(tf.shape(true_seq)[0], dtype=tf.float32)
        return average_mismatch
    result = tf.cond(cond, true_fn, false_fn)
    
    return result
     
def my_accuracy_tf(true,pred, window_prediction_size = 54):
    '''true and pred are the list of arrays [a,b] (corresponding to one-length vector)'''
    #pred_seg = np.array([get_aprox_seg[pred_vect, window_prediction_size] for pred_vect in pred])
    #true_seg = np.array([get_seg[true_vect, window_prediction_size] for true_vect in true])
    
    #pred_seg = [get_aprox_seg(pred_vect, window_prediction_size) for pred_vect in pred]
    #true_seg = [get_seg(true_vect, window_prediction_size) for true_vect in true]
    '''true and pred are the tf (corresponding to one-length vector)'''
   
    pred_seg = tf.map_fn(lambda x: get_aprox_seg_tf(x, window_prediction_size), pred, dtype=tf.float32)
    true_seg = tf.map_fn(lambda x: get_seg_tf(x, window_prediction_size), true, dtype=tf.float32)

    return mean_mismatch_tf(true_seg, pred_seg)

def my_accuracy_1_tf(true,pred, window_prediction_size = 54):
   
    pred_seg = tf.map_fn(lambda x: get_aprox_seg_tf(x, window_prediction_size), pred, dtype=tf.float32)
    true_seg = tf.map_fn(lambda x: get_seg_tf(x, window_prediction_size), true, dtype=tf.float32)

    return mean_mismatch_1_tf(true_seg, pred_seg)

class MyMetric(tf.keras.metrics.Metric):

    def __init__(self, window_prediction_size, name='MyMetric', **kwargs):
        super(MyMetric, self).__init__(name=name, **kwargs)
        self.window_prediction_size = window_prediction_size

    def get_config(self):
        config = super(MyMetric, self).get_config()
        config.update({"window_prediction_size": self.window_prediction_size})
        return config
     
    def update_state(self, y_true, y_pred, sample_weight=None):
        self.mean_mis =  my_accuracy_tf(y_true,y_pred,self.window_prediction_size)

    def result(self):
        return 1-self.mean_mis
    
class MyMetric_1(tf.keras.metrics.Metric):

    def __init__(self, window_prediction_size, name='MyMetric_1', **kwargs):
        super(MyMetric_1, self).__init__(name=name, **kwargs)
        self.window_prediction_size = window_prediction_size

    def get_config(self):
        config = super(MyMetric_1, self).get_config()
        config.update({"window_prediction_size": self.window_prediction_size})
        return config
     
    def update_state(self, y_true, y_pred, sample_weight=None):
        self.mean_mis_1 =  my_accuracy_1_tf(y_true,y_pred,self.window_prediction_size)

    def result(self):
        return 1-self.mean_mis_1    
    
def e_dist(x,y):
    """Returns squared eucl distance matrix"""
    '''x  of (None,:,:)'''
    
    dists = -2 * tf.tensordot(x[0,:,:], tf.transpose(y[0,:,:]), axes=1) + \
            tf.math.reduce_sum(y[0,:,:] ** 2, axis=1) + \
            tf.math.reduce_sum(x[0,:,:] ** 2, axis=1)[:, tf.newaxis]
    return dists

def similarity(m):
    dists = e_dist(m,m)
    return tf.reduce_sum(dists)
