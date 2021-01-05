import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.utils import class_weight
from ray import tune
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.integration.keras import TuneReportCallback
import pandas as pd

from models import get_compiled_model, variable_model
from data_pipelines import gal_tvt_split

# Data loading and splitting
X_train, y_train, X_val, y_val, X_test, y_test = gal_tvt_split(fracs=(0.9, 0.05, 0.05))

# Categories coding
cat = y_train.cat.categories
y_train = y_train.cat.codes
y_val = y_val.cat.codes
y_test = y_test.cat.codes

# Class weights initialization
class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = dict(enumerate(class_weights))
val_weights = y_val.map(class_weights_dict)


# Model initialization
def train_model(config):
    # config: batch, learning_rate, n_layers, n_neurons
    # Setting callbacks
    cp_name = f"lr({config['learning_rate']:.4f})b({config['batch']})" + '.h5'
    cp_callback = keras.callbacks.ModelCheckpoint(filepath='trained-models/' + cp_name,
                                                  monitor='val_loss',
                                                  mode='min',
                                                  save_best_only=True,
                                                  verbose=1, )

    tb_log_dir = "logs/fit/" + cp_name
    tb_callback = keras.callbacks.TensorBoard(log_dir=tb_log_dir, histogram_freq=1)

    es_callback = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                mode='min',
                                                verbose=1,
                                                patience=10,
                                                )

    tune_callback = TuneReportCallback({"val_loss": "val_loss"})

    callback_list = [cp_callback, tb_callback, es_callback, tune_callback]

    # Dividing dataset into mini-batches
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train.values, y_train.values)).batch(config['batch'])

    # Training model
    # model = get_compiled_model(learning_rate=config['learning_rate'])
    model = variable_model(
        learning_rate=config['learning_rate'],
        n_layers=config['n_layers'],
        n_neurons=config['n_neurons']
    )

    fit_history = model.fit(train_dataset,
                            validation_data=(X_val, y_val, val_weights.values),
                            epochs=150,
                            callbacks=callback_list,
                            class_weight=class_weights_dict)

    pd.DataFrame(fit_history.history).to_csv("logs/histories/history" + cp_name[:-3] + ".csv")


# Running hyperparameters tuning
analysis = tune.run(
    train_model,
    name="HII classification",
    metric="val_loss",
    mode="min",
    search_alg=HyperOptSearch(metric="val_loss", mode="min"),
    num_samples=15,
    stop={"training_iteration": 150},
    config={
        "n_layers": tune.randint(1, 5),
        "n_neurons": tune.choice([6, 12, 24, 48]),
        "learning_rate": 0.001,
        "batch": 512,
    },
    resources_per_trial={"cpu": 4}) #Due to memory limitations

print("Best hyperparameters found were: ", analysis.best_config)

analysis.results_df.to_csv('logs/anasysis_df_models.csv')
