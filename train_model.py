import tensorflow as tf
from tensorflow import keras
import pickle

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

train_dataset = tf.data.Dataset.from_tensor_slices((X_train.values, y_train.values)).batch(64)

cp_callback = keras.callbacks.ModelCheckpoint(filepath=c.checkpoint_path,
                                              save_weights_only=True,
                                              verbose=1)

model = get_compiled_model()
model.fit(train_dataset, epochs=50, callbacks=[cp_callback])
