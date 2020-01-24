import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import yaml
from pathlib import Path
import torch
from training.postprocessing.semantic_labeling import make_classification_map


def compute_stem_metrics(stem_position_output, stem_position_target, tolerance_radius):
    """Compute metrics for evaluation of the stem detection.

    Count a true positive for each predicted stem if an actual stem is within tolerance_radius.
    Count a false positive for each predicted stem if no actual stem is within tolerance_radius.
    Count a false negative for each actual stem if no predicted stem is within tolerance_radius.

    For all true positives sum up the deviation from the actual stem position.

    Returns:
        A tuple of confusion matrix and accumulated deviation.
    """

    stem_confusion_matrix = np.zeros((2, 2,), dtype=np.int)
    accum_deviation = 0
    batch_size = len(stem_position_output)

    # compute metrics for each batch
    for index_in_batch in range(batch_size):
        # get count of predicted and actual stems
        output_count = stem_position_output[index_in_batch].shape[0]
        target_count = stem_position_target[index_in_batch].shape[0]

        # bring stem position to numpy
        stem_output_coords = stem_position_output[index_in_batch].cpu().detach().numpy()
        stem_target_coords = stem_position_target[index_in_batch].cpu().detach().numpy()

        if output_count>0 and target_count>0:
            # using numpy broadcasting
            differences = stem_output_coords[:, None, :]-stem_target_coords[None, :, :]
            distances = np.linalg.norm(differences, axis=-1)
            min_distances_per_output = np.amin(distances, axis=1)
            min_distances_per_target = np.amin(distances, axis=0)

            # calculate deviation for true positives

            is_true_positive = min_distances_per_output<=tolerance_radius
            accum_deviation += np.sum(min_distances_per_output[is_true_positive])

            true_positives = np.sum(is_true_positive)
            false_positives = np.sum(min_distances_per_output>tolerance_radius) # the stems we hit wrongly
            false_negatives = np.sum(min_distances_per_target>tolerance_radius) # the stems we missed
            # true_negatives not well defined
        elif output_count>0:
            # we do only have false positives
            true_positives = 0
            false_positives = output_count
            false_negatives = 0
        elif target_count>0:
            # we do only have false negatives
            true_positives = 0
            false_positives = 0
            false_negatives = target_count
        else:
            # we have nothing except true_negatives which are not well defined
            true_positives = 0
            false_positives = 0
            false_negatives = 0

        stem_confusion_matrix += np.array([[true_positives,  false_positives], [false_negatives,  0]])

    return stem_confusion_matrix, accum_deviation


def compute_confusion_matrix(semantic_output_batch, semantic_target_batch, sugar_beet_threshold, weed_threshold):
    """Note: The input will be a batch.
    """
    predicted_semantic_labels = make_classification_map(semantic_output_batch,
            sugar_beet_threshold, weed_threshold).cpu().detach().numpy()
    actual_semantic_labels = semantic_target_batch.cpu().detach().numpy()

    valid_pixels = np.where(actual_semantic_labels!=3) # do not use pixels labeled as ignored (index=3)
    predicted_semantic_labels = predicted_semantic_labels[valid_pixels]
    actual_semantic_labels = actual_semantic_labels[valid_pixels]

    return confusion_matrix(predicted_semantic_labels.flatten(),
                            actual_semantic_labels.flatten(),
                            labels=[0, 1, 2])


def compute_metrics_from_confusion_matrix(confusion_matrix, eps=1e-6):
  """Get several class-wise metrics given a confusion matrix.

  Args:
      confusion_matrix (numpy.array): A square matrix.

  Returns:
      A python dictionary with keys 'accuracy', 'precision', 'recall', 'f1_score', 'iou'.
  """
  total = np.sum(confusion_matrix)
  true_positives = np.diag(confusion_matrix)
  false_negatives = np.sum(confusion_matrix, axis=0)-true_positives
  false_positives = np.sum(confusion_matrix, axis=1)-true_positives
  true_negatives = total-true_positives-false_negatives-false_positives

  accuracy = (true_positives+true_negatives)/(total+eps)
  precision = true_positives/(true_positives+false_positives+eps)
  recall = true_positives/(true_positives+false_negatives+eps)
  f1_score = 2.0*precision*recall/(precision+recall+eps)
  iou = true_positives/(true_positives+false_negatives+false_positives+eps)

  metrics = {}
  metrics['accuracy'] = accuracy.tolist()
  metrics['precision'] = precision.tolist()
  metrics['recall'] = recall.tolist()
  metrics['f1_score'] = f1_score.tolist()
  metrics['iou'] = iou.tolist()

  return metrics


def plot_confusion_matrix(path,
                          confusion_matrix,
                          filename,
                          class_names=['background', 'weed', 'sugar_beet'],
                          normalize=False,
                          color_map=plt.cm.Blues,
                          eps=1e-6):
    """Reference: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """

    accuracy = np.trace(confusion_matrix)/(np.sum(confusion_matrix).astype('float')+eps)
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
           title='Confusion Matrix: '+Path(filename).stem,
           ylabel='Actual label',
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
    confusion_matrix_path = confusion_matrix_dir_path/filename

    ax.set_ylim(len(confusion_matrix)-0.5, -0.5)
    fig.savefig(str(confusion_matrix_path), bbox_inches='tight')
    plt.close('all')


def precision_recall_curve_and_average_precision(semantic_outputs, semantic_targets, path, filename):
    average_precisions = []

    valid_pixels = semantic_targets!=3
    for class_label, class_name in [(1, 'weed'), (2, 'sugar_beet')]:
        valid_outputs = semantic_outputs[:, class_label][valid_pixels]
        valid_targets = semantic_targets[valid_pixels]==class_label
        average_precision = _single_class_precision_recall(valid_outputs, valid_targets, class_name, path, filename)
        average_precisions.append(average_precision)

    return average_precisions


def _single_class_precision_recall(semantic_outputs, target_mask, class_name, path, filename, eps=1e-6):
    num_confidence_thresholds = 101
    confidence_thresholds = np.linspace(0.0, 1.0, num_confidence_thresholds)

    precisions = np.zeros(num_confidence_thresholds, dtype=np.float)
    recalls = np.zeros(num_confidence_thresholds, dtype=np.float)
    ious = np.zeros(num_confidence_thresholds, dtype=np.float)

    for index, confidence_threshold in enumerate(confidence_thresholds):
        predicted_mask = semantic_outputs>=confidence_threshold

        true_positives_mask = np.logical_and(target_mask, predicted_mask)
        false_positives_mask = np.logical_and(np.logical_not(target_mask), predicted_mask)
        false_negatives_mask = np.logical_and(target_mask, np.logical_not(predicted_mask))

        true_positives = np.sum(true_positives_mask)
        false_positives = np.sum(false_positives_mask)
        false_negatives = np.sum(false_negatives_mask)

        precisions[index] = true_positives/(true_positives+false_positives+eps)
        recalls[index] = true_positives/(true_positives+false_negatives+eps)
        ious[index] = true_positives/(true_positives+false_negatives+false_positives+eps)

    num_recall_thresholds = 101
    recall_thresholds = np.linspace(0.0, 1.0, num_recall_thresholds)

    sum_precision = 0.0
    for recall_threshold in recall_thresholds:
        above_or_equal_threshold = recalls>=recall_threshold

        if not np.any(above_or_equal_threshold):
            # precision is zero
            continue

        # maximum precision with a recall equal or above threshold does contribute to average precision
        sum_precision += np.max(precisions[above_or_equal_threshold])

    average_precision = sum_precision/num_recall_thresholds

    f1_scores = 2.0*precisions*recalls/(precisions+recalls+eps)
    index_max_f1 = np.argmax(f1_scores)
    index_max_iou = np.argmax(ious)

    print('average precision ({}): {:.4f}'.format(class_name, average_precision))
    print('confidence threshold for maximum F1 {:.4f} ({}): {:.2f}'.format(f1_scores[index_max_f1], class_name, confidence_thresholds[index_max_f1]))
    print('confidence threshold for maximum IoU {:.4f} ({}): {:.2f}'.format(ious[index_max_iou], class_name, confidence_thresholds[index_max_iou]))

    # plotting
    indices = np.argsort(recalls)
    recalls = recalls[indices]
    confidence_thresholds = confidence_thresholds[indices]
    precisions = precisions[indices]

    # set all obtained precisions to max precision for given recall threshold
    for index in range(num_confidence_thresholds):
        precisions[index] = np.max(precisions[index:])

    plt.ylim(0.0, 1.1)
    plt.ylabel('precision')
    plt.xlim(0.0, 1.0)
    plt.xlabel('recall')
    plt.grid(True, color=(0.5, 0.5, 0.5))

    # plot as bars with certain width
    # shifted_recalls = np.zeros_like(recalls)
    # shifted_recalls[0] = recalls[0]
    # shifted_recalls[1:] = recalls[:-1]
    # widths = shifted_recalls-recalls
    # plt.bar(recalls, precisions, align='edge', width=widths, color=(0.7, 0.7, 0.7), linewidth=2)

    # plot as single points
    plt.scatter(recalls, precisions, s=5, c=np.array([0.2, 0.2, 0.8]).reshape(1, -1))

    # put confidence threshold as text
    for index in range(0, num_confidence_thresholds, 10):
        text = '{:.02f}'.format(confidence_thresholds[index])
        plt.annotate(text, (recalls[index], precisions[index]))

    path = path/("precision_recall/{}_{}.png".format(filename, class_name))
    path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(str(path), bbox_inches='tight')
    plt.close('all')

    return average_precision.item()

