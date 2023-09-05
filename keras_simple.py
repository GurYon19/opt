"""
Optuna example that optimizes a neural network classifier configuration for the
MNIST dataset using Keras.

In this example, we optimize the validation accuracy of MNIST classification using
Keras. We optimize the filter and kernel size, kernel stride and layer activation.

"""
import urllib
import warnings

import optuna
from optuna.integration import KerasPruningCallback
from tensorflow import keras
from keras.backend import clear_session
from keras.datasets import mnist
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.models import Sequential
from keras.optimizers import RMSprop
from create_ds import get_train_val_test_dataset
from models import create_model, CLASSES

# TODO(crcrpar): Remove the below three lines once everything is ok.
# Register a global custom opener to avoid HTTP Error 403: Forbidden when downloading MNIST.
opener = urllib.request.build_opener()
opener.addheaders = [("User-agent", "Mozilla/5.0")]
urllib.request.install_opener(opener)


N_TRAIN_EXAMPLES = 3000
N_VALID_EXAMPLES = 1000
BATCHSIZE = 128
CLASSES = 10
EPOCHS = 10


def objective(trial):
    # Clear clutter from previous session graphs.
    keras.backend.clear_session()

    train_ds,val_ds,test_ds = get_train_val_test_dataset()
    num_classes = 2
    input_length = 288
    input_channels = 6

    # Generate our trial model.
    model = create_model(trial)

    # Fit the model on the training data.
    # The KerasPruningCallback checks for pruning condition every epoch.
    model.fit(
        train_ds,
        batch_size=BATCHSIZE,
        callbacks=[KerasPruningCallback(trial, "val_accuracy")],
        epochs=EPOCHS,
        validation_data=val_ds,
        verbose=1,
    )

    # Evaluate the model accuracy on the validation set.
    score = model.evaluate(test_ds, verbose=0) 
    return score[1]


if __name__ == "__main__":
    warnings.warn(
        "Recent Keras release (2.4.0) simply redirects all APIs "
        "in the standalone keras package to point to tf.keras. "
        "There is now only one Keras: tf.keras. "
        "There may be some breaking changes for some workflows by upgrading to keras 2.4.0. "
        "Test before upgrading. "
        "REF:https://github.com/keras-team/keras/releases/tag/2.4.0"
    )
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100, timeout=600)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
