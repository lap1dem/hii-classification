import os

from visualization import *
from models import get_compiled_model
import config as c
from data_pipelines import gal_tvt_split

model = get_compiled_model()
model.load_weights(c.checkpoint_path)

X_train, y_train, X_val, y_val, X_test, y_test = gal_tvt_split(fracs=(0.9, 0.05, 0.05))
cat = y_train.cat.categories
y_train = y_train.cat.codes
y_val = y_val.cat.codes
y_test = y_test.cat.codes

loss, acc = model.evaluate(X_test, y_test, verbose=2)
print(f"Restored model, accuracy: {100 * acc:3.2f}%")

y_pred = np.argmax(model.predict(X_test), 1)

if not os.path.exists('figures'):
    os.makedirs('figures')
plot_confusion_matrix(y_test, y_pred, cat, save_path='figures/confusion.png')

non_hii_index = y_test[y_test != 1].index
X_test = X_test.drop(non_hii_index)
y_test = y_test.drop(non_hii_index)
hii_pred = np.argmax(model.predict(X_test), 1)
hii_precision = np.sum(hii_pred == 1) / len(hii_pred)
print(f"Restored model, HII precision: {100 * hii_precision:3.2f}%")