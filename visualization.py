import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
import numpy as np


def plot_confusion_matrix(y_test, y_pred, classes, save_path,
                          normalize=False, title='Confusion matrix',
                          cmap=plt.cm.Blues, dpi=500,
                          ):
    cm = confusion_matrix(y_test, y_pred)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
