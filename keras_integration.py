import urllib
import warnings
from keras.callbacks import TensorBoard
import optuna
from optuna.integration import TFKerasPruningCallback
from optuna.trial import TrialState
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.datasets import mnist
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.utils import to_categorical
from create_ds import get_train_val_test_dataset,DS_BATCH_SIZE
from models import create_model, CLASSES
from custom.callbacks import f1_fp



N_TRAIN_EXAMPLES = 3000
N_VALID_EXAMPLES = 1000
BATCHSIZE = DS_BATCH_SIZE
EPOCHS = 78

train_ds,val_ds,test_ds = get_train_val_test_dataset()

def objective(trial):
    # Clear clutter from previous session graphs.
    keras.backend.clear_session()

    #train_ds,val_ds,test_ds = get_train_val_test_dataset()
    #print(train_ds.cardinality)
    # for x,y in train_ds.take(1):
    #     print(x)
    #     print(y.numpy())
    # Generate our trial model.
    model = create_model(trial)
    logdir = './logs'
    tensorboard_callback = TensorBoard(log_dir=logdir)
    model.fit(
        train_ds,
        batch_size=BATCHSIZE,
        callbacks=[tensorboard_callback,TFKerasPruningCallback(trial, monitor = f1_fp)],
        epochs=EPOCHS,
        validation_data=val_ds,
        verbose=1,
    )
    # Evaluate the model accuracy on the validation set.
    #f1_score,fp = model.evaluate(test_ds, verbose=0)
    loss,f1 = model.evaluate(test_ds, verbose=0)

    #f1_fp_score = f1_fp(f1,fp)
    return f1


if __name__ == "__main__":
    warnings.warn(
        "Recent Keras release (2.4.0) simply redirects all APIs "
        "in the standalone keras package to point to tf.keras. "
        "There is now only one Keras: tf.keras. "
        "There may be some breaking changes for some workflows by upgrading to keras 2.4.0. "
        "Test before upgrading. "
        "REF: https://github.com/keras-team/keras/releases/tag/2.4.0. "
        "There is an alternative callback function that can be used instead: "
        ":class:`~optuna.integration.TFKerasPruningCallback`",
    )
    #TODO:  study = optuna.create_study(direction='maximize',storage='sqlite:///db.sqlite3'
    study = optuna.create_study(direction='maximize',storage='sqlite:///db.sqlite3', pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=600)
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    # Save the best epoch for each trial.
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    #save model with best trial
    model = create_model(trial)
    model.save('Yonis_best_model.h5')
    from keras.models import load_model
    model_path = 'Yonis_model.h5'
    #load the model
    model = load_model(model_path)

    #take random from test_ds and predict
    for x,y in test_ds.take(1):
        print(x)
        print(y.numpy())
        print(model.predict(x))
        #plot x, where x is 6 channels signal
        #for i in range(6):
        #    plt.plot(x[i])
