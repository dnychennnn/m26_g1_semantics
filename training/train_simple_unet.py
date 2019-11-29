"""Train test model.

Author: Jan Quakernack

Note: This module contains parts, which were written for other student projects
conducted by the author.
"""

import torch
from torch import nn
import datetime
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt

from training.models.model import Model
from training.dataloader import SugarBeetDataset
from training.losses import StemClassificationLoss, StemRegressionLoss
from training import vis
from training import LOGS_DIR, MODELS_DIR, CUDA_DEVICE_NAME, load_config
from training.evalmetrics import compute_confusion_matrix, compute_metrics_from_confusion_matrix, plot_confusion_matrix, write_metrics_to_file


def main():
    config = load_config('training.yaml')

    learning_rate = config['learning_rate']
    batch_size = config['batch_size']
    num_epochs = config['num_epochs']

    # weighting of losses
    semantic_loss_weight = config['semantic_loss_weight']
    stem_loss_weight = config['stem_loss_weight']
    stem_classification_loss_weight = config['stem_classification_loss_weight']
    stem_regression_loss_weight = config['stem_regression_loss_weight']

    # class weights for semantic segmentation
    weight_background = config['weight_background']
    weight_weed = config['weight_weed']
    weight_sugar_beet = config['weight_sugar_beet']


    # class weights for stem keypoint detection
    weight_stem_background = config['weight_stem_background']
    weight_stem = config['weight_stem']

    size_test_set = config['size_test_set']

    path_to_weights_file = 'simple_unet.pth' # config['path_to_weights_file']
    architecture_name = 'simple_unet' # config['architecture_name']

    device = torch.device(CUDA_DEVICE_NAME) if torch.cuda.is_available() else torch.device('cpu')

    dataset = SugarBeetDataset.from_config()
    # split into train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset_train = torch.utils.data.Subset(dataset, indices[:-size_test_set])
    dataset_test = torch.utils.data.Subset(dataset, indices[-size_test_set:])

    # define training and validation data loaders
    data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False)

    model = Model.by_name(architecture_name, phase='training', path_to_weights_file=path_to_weights_file, verbose=True).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    semantic_loss_function = nn.CrossEntropyLoss(ignore_index=3, weight=torch.Tensor([weight_background, weight_weed, weight_sugar_beet])).to(device)

    stem_classification_loss_function = StemClassificationLoss(weight_background=weight_stem_background, weight_stem=weight_stem).to(device)
    stem_regression_loss_function = StemRegressionLoss()

    now = datetime.datetime.now()
    run_time_string = now.strftime('%b_%d_%Y_%Hh%M')
    run_name = '{}_{}'.format('simple_unet', run_time_string)
    log_dir = LOGS_DIR/run_name
    if not log_dir.exists():
        log_dir.mkdir()

    for epoch_index in range(num_epochs):

        accumulated_confusion_matrix_train = np.zeros((3, 3,), dtype=np.long)

        for batch_index, batch in enumerate(data_loader_train):
            # skip trainin
            break
            print('Train batch {}/{} in epoch {}/{}.'.format(batch_index, len(data_loader_train), epoch_index, num_epochs))

            model.train()
            optimizer.zero_grad()

            # get input an bring to device
            input_batch, semantic_target_batch, stem_keypoint_target_batch, stem_offset_target_batch = batch

            # debug overfit first image in dataset
            # input_batch, semantic_target_batch, stem_keypoint_target_batch, stem_offset_target_batch = dataset[0]

            # input_batch = input_batch[None, ...]
            # semantic_target_batch = semantic_target_batch[None, ...]
            # stem_keypoint_target_batch = stem_keypoint_target_batch[None, ...]
            # stem_offset_target_batch = stem_offset_target_batch[None, ...]

            input_batch = input_batch.to(device)
            semantic_target_batch = semantic_target_batch.to(device)
            stem_keypoint_target_batch = stem_keypoint_target_batch.to(device)
            stem_offset_target_batch = stem_offset_target_batch.to(device)

            # foward pass
            semantic_output_batch, stem_keypoint_output_batch, stem_offset_output_batch = model(input_batch)

            # compute losses
            semantic_loss = semantic_loss_weight*\
                            semantic_loss_function(semantic_output_batch,
                                                   semantic_target_batch)

            stem_classification_loss = stem_loss_weight*stem_classification_loss_weight*\
                                       stem_classification_loss_function(stem_keypoint_output_batch=stem_keypoint_output_batch,
                                                                         stem_keypoint_target_batch=stem_keypoint_target_batch)

            stem_regression_loss = stem_loss_weight*stem_regression_loss_weight*\
                                   stem_regression_loss_function(stem_offset_output_batch=stem_offset_output_batch,
                                                                 stem_keypoint_target_batch=stem_keypoint_target_batch,
                                                                 stem_offset_target_batch=stem_offset_target_batch)

            stem_loss = stem_classification_loss+stem_regression_loss
            loss = semantic_loss+stem_loss

            print('  Loss: {:04f}'.format(loss.item()))
            print('  Semantic loss: {:04f}'.format(semantic_loss.item()))
            print('  Stem loss: {:04f}'.format(stem_loss.item()))
            print('  Stem classification loss: {:04f}'.format(stem_classification_loss.item()))
            print('  Stem regression loss: {:04f}'.format(stem_regression_loss.item()))

            loss.backward()
            optimizer.step()

            # accumulate confusion matrix
            accumulated_confusion_matrix_train += compute_confusion_matrix(semantic_output_batch, semantic_target_batch)


        # compute IoU and accuracy at the end of the epoch over the last batch
        # mIoU, acc = compute_mIoU_and_Acc(semantic_output_batch, semantic_target_batch, 3)
        # print('[Training] mIoU: {:04f}, Accuracy: {:04f}'.format(mIoU, acc))

        # end of epoch
        print('End of epoch. Make checkpoint. Evaluate.')
        make_checkpoint(run_name,
                        log_dir,
                        epoch_index,
                        model,
                        accumulated_confusion_matrix_train,
                        dataset,
                        data_loader_test)


def make_checkpoint(run_name, log_dir, epoch, model, accumulated_confusion_matrix_train, dataset, data_loader_test):
    cp_name = '{}_cp_{:06d}'.format(run_name, epoch)

    cp_dir = log_dir/cp_name
    if not cp_dir.exists():
        cp_dir.mkdir()

    # save model weights
    # TODO optional: only save best model to save some space
    weights_path = cp_dir/(cp_name+'.pth')
    torch.save(model.state_dict(), str(weights_path))

    # save accumulated confusion matrix
    print('Save confusion matrix of training.')
    plot_confusion_matrix(cp_dir, accumulated_confusion_matrix_train, normalize=False, filename=cp_name+'_test.png')

    # calculate metrics on accumulated confusion matrix
    metrics_train = compute_metrics_from_confusion_matrix(accumulated_confusion_matrix_train)
    print('Calculate metrics of training.')
    write_metrics_to_file(cp_dir, metrics_train, filename=cp_name+'_test.yaml')

    evaluate_on_checkpoint(model, dataset, data_loader_test, epoch, cp_dir, cp_name)



def evaluate_on_checkpoint(model, dataset, data_loader_test, epoch, cp_dir, cp_name):
    device = torch.device(CUDA_DEVICE_NAME) if torch.cuda.is_available() else torch.device('cpu')

    model = model.to(device)
    model.eval()

    # folder to save some examples
    examples_dir = cp_dir/'examples'
    if not examples_dir.exists():
        examples_dir.mkdir()

    accumulated_confusion_matrix_test = np.zeros((3, 3), np.long)

    for batch_index, batch in enumerate(data_loader_test):
        input_batch, semantic_target_batch, stem_keypoint_target_batch, stem_offset_target_batch = batch

        input_batch = input_batch.to(device)
        semantic_target_batch = semantic_target_batch.to(device)
        stem_keypoint_target_batch = stem_keypoint_target_batch.to(device)
        stem_offset_target_batch = stem_offset_target_batch.to(device)

        # foward pass
        semantic_output_batch, stem_keypoint_output_batch, stem_offset_output_batch, stem_voting_output_batch = model(input_batch)

        path_for_plots = examples_dir/'sample_{:02d}'.format(batch_index)
        save_plots(path_for_plots,
                   input_slice=input_batch[0],
                   semantic_output=semantic_output_batch[0],
                   stem_keypoint_output=stem_keypoint_output_batch[0],
                   stem_offset_output=stem_offset_output_batch[0],
                   keypoint_radius=dataset.keypoint_radius,
                   mean_rgb=dataset.mean_rgb,
                   std_rgb=dataset.std_rgb,
                   mean_nir=dataset.mean_nir,
                   std_nir=dataset.std_nir)

        # compute IoU and Accuracy over every batch
        accumulated_confusion_matrix_test += compute_confusion_matrix(semantic_output_batch, semantic_target_batch)

    print('Save confusion matrix of test.')
    plot_confusion_matrix(cp_dir, accumulated_confusion_matrix_test, normalize=False, filename=cp_name+'_test.png')

    # calculate metrics on accumulated confusion matrix
    metrics_test = compute_metrics_from_confusion_matrix(accumulated_confusion_matrix_test)
    print('Calculate metrics of test.')
    write_metrics_to_file(cp_dir, metrics_test, filename=cp_name+'_test.yaml')

    mean_accuracy = np.mean(np.asarray(metrics_test['accuracy'])[1:]) # without background
    mean_iou = np.mean(np.asarray(metrics_test['iou'])[1:]) # without background

    print('  Mean IoU (without background): {:04f}'.format(mean_iou))
    print("  IoU 'background': {:04f}".format(metrics_test['iou'][0]))
    print("  IoU 'weed': {:04f}".format(metrics_test['iou'][1]))
    print("  IoU 'sugar beet': {:04f}".format(metrics_test['iou'][2]))
    print('  Mean accuracy (without background): {:04f}'.format(mean_accuracy))
    print("  Accuracy 'background': {:04f}".format(metrics_test['accuracy'][0]))
    print("  Accuracy 'weed': {:04f}".format(metrics_test['accuracy'][1]))
    print("  Accuracy 'sugar beet': {:04f}".format(metrics_test['accuracy'][2]))


def save_plots(path, input_slice, semantic_output, stem_keypoint_output, stem_offset_output, keypoint_radius, **normalization):
    """Make plots and write images.
    """
    image_bgr = vis.tensor_to_bgr(input_slice[:3],
                                  mean_rgb=normalization['mean_rgb'],
                                  std_rgb=normalization['std_rgb'])

    image_nir = vis.tensor_to_single_channel(input_slice[3],
                                             mean=normalization['mean_nir'],
                                             std=normalization['std_nir'])

    image_false_color = vis.tensor_to_false_color(input_slice[:3], input_slice[3], **normalization)


    plot_semantics = vis.make_plot_from_semantic_output(input_rgb=input_slice[:3],
                                                        input_nir=input_slice[3],
                                                        semantic_output=semantic_output,
                                                        apply_softmax=True, **normalization)

    plot_stems = vis.make_plot_from_stem_output(input_rgb=input_slice[:3],
                                                input_nir=input_slice[3],
                                                stem_keypoint_output=stem_keypoint_output,
                                                stem_offset_output=stem_offset_output,
                                                keypoint_radius=keypoint_radius,
                                                apply_sigmoid=True,
                                                apply_tanh=True,
                                                **normalization)

    path_rgb = path.parent/(path.name+'_rgb.jpg')
    path_nir = path.parent/(path.name+'_nir.jpg')
    path_false_color = path.parent/(path.name+'_false_color.jpg')
    path_semantics = path.parent/(path.name+'_semantics.jpg')
    path_stems = path.parent/(path.name+'_stems.jpg')

    # debug output
    # cv2.imshow('plot_semantics', plot_semantics)
    # cv2.imshow('plot_stems', plot_stems)
    # cv2.imshow('image_bgr', image_bgr)
    # cv2.imshow('image_nir', image_nir)
    # cv2.imshow('image_false_color', image_false_color)
    # cv2.waitKey(0)

    cv2.imwrite(str(path_rgb), (255.0*image_bgr).astype(np.uint8))
    cv2.imwrite(str(path_nir), (255.0*image_nir).astype(np.uint8))
    cv2.imwrite(str(path_false_color), (255.0*image_false_color).astype(np.uint8))
    cv2.imwrite(str(path_semantics), (255.0*plot_semantics).astype(np.uint8))
    cv2.imwrite(str(path_stems), (255.0*plot_stems).astype(np.uint8))


if __name__=='__main__':
    main()

