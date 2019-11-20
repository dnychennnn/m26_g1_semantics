import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

from training.postprocessing import make_classification_map


def compute_confusion_matrix(semantic_output_batch, semantic_target_batch):
    """Note: The input will be a batch.
    """
    predicted_semantic_labels = make_classification_map(semantic_output_batch).cpu().detach().numpy()
    actual_semantic_labels = semantic_target_batch.cpu().detach().numpy()

    valid_pixels = np.where(actual_semantic_labels!=3) # do not use pixels labeled as ignored (index=3)
    predicted_semantic_labels = predicted_semantic_labels[valid_pixels]
    actual_semantic_labels = actual_semantic_labels[valid_pixels]

    return confusion_matrix(predicted_semantic_labels.flatten(),
                            actual_semantic_labels.flatten())


def compute_accuracy_from_confusion_matrix():
  pass


def compute_precision_recall_from_confusion_matrix():
  pass


def compute_iou_from_confusion_matrix():
  pass


def plot_confusion_matrix(path,
                          confusion_matrix,
                          class_names,
                          title,
                          normalize=False,
                          color_map=plt.cm.Blues):
    """Reference: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """

    accuracy = np.trace(confusion_matrix)/np.sum(confusion_matrix).astype('float')
    misclass = 1-accuracy

    if normalize:
        confusion_matrix = confusion_matrix.astype('float')/(confusion_matrix.sum(axis=1)[:, np.newaxis]+1e-10)

    fig, ax = plt.subplots()
    im = ax.imshow(confusion_matrix, interpolation='nearest', cmap=color_map)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(confusion_matrix.shape[1]),
           yticks=np.arange(confusion_matrix.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=class_names, yticklabels=class_names,
           title='Confusion Matrix: '+title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", va="center",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = confusion_matrix.max() / 2.
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            ax.text(j, i, format(confusion_matrix[i, j], fmt),
                    ha="center", va="center",
                    color="white" if confusion_matrix[i, j] > thresh else "black")

    confusion_matrix_dir_path = path/'confusion_matrix'
    if not confusion_matrix_dir_path.exists():
        confusion_matrix_dir_path.mkdir()
    confusion_matrix_path = confusion_matrix_dir_path/(str(title)+'.png')

    ax.set_ylim(len(confusion_matrix)-0.5, -0.5)
    fig.savefig(str(confusion_matrix_path), bbox_inches='tight')
    plt.close('all')
