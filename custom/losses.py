from keras import backend as K
import tensorflow as tf


def f1_score(y_true, y_pred): 
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

def focal_loss(y, s, alpha=0.25, gamma=2):
    """Calculates the focal loss for binary classification.

    Args:
    y: The ground truth labels.
    s: The predicted scores.
    alpha: The modulating factor.
    gamma: The power factor.

    Returns:
    The focal loss.
    """
    import math
    p = math.sigmoid(s)
    q = 1 - p
    modulating_factor = alpha * (1 - p) ** gamma
    loss = -modulating_factor * y * math.log(p) - (1 - modulating_factor) * (1 - y) * math.log(q)
    return loss                                             

def hinge_loss(y_true, y_pred):
    """Calculates the hinge loss for binary classification.

    Args:
    y: The ground truth labels.
    s: The predicted scores.

    Returns:
    The hinge loss.
    """

    margin = 1
    loss = max(0, margin - y_true * y_pred)
    return loss

def logistic_loss(y_true, y_pred):
  """Calculates the logistic loss."""

  # Get the predictions and labels from the model.
  predictions = y_pred
  labels = y_true

  # Calculate the logistic loss.
  logistic_loss = -tf.math.log(predictions * labels + (1 - predictions) * (1 - labels))

  # Return the logistic loss.
  return logistic_loss



def false_positives(y_true, y_pred):
    return K.sum(K.round(K.clip(y_pred - y_true, 0, 1)))

def false_positives_loss(y_true, y_pred):
    return K.sum(K.square(K.clip(y_pred - y_true, 0, 1)))
