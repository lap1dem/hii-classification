import tensorflow as tf
from tensorflow import keras
import pickle
import numpy as np
import datetime

import config as c
from models import get_compiled_model
from data_pipelines import gal_tt_split

X_train, y_train, X_test, y_test = gal_tt_split()

cat = y_train.cat.categories
y_train = y_train.cat.codes
y_test = y_test.cat.codes

pickle.dump(X_test, open("xtest.p", "wb"))
pickle.dump(y_test, open("ytest.p", "wb"))
pickle.dump(cat, open("categories.p", "wb"))

train_dataset = tf.data.Dataset.from_tensor_slices((X_train.values, y_train.values)).batch(512)

cp_callback = keras.callbacks.ModelCheckpoint(filepath=c.checkpoint_path,
                                              save_weights_only=True,
                                              verbose=1)

model = get_compiled_model()

# ----------------
from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = dict(enumerate(class_weights))
print(class_weights_dict)
# ----------------

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


model.fit(train_dataset, epochs=50, callbacks=[cp_callback, tensorboard_callback], class_weight=class_weights_dict)
