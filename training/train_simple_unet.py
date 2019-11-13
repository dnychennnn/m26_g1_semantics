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

from training.model.simple_unet import SimpleUnet
from training.dataloader import SugarBeetDataset
from training.losses import StemClassificationLoss, StemRegressionLoss
from training import vis
from training import LOGS_DIR, MODELS_DIR

from utils import intersection_and_union, accuracy, make_classification_map, compute_mIoU_and_Acc, compute_confusion_matrix, plot_confusion_matrix
import matplotlib.pyplot as plt


def main():
    learning_rate = 0.001
    batch_size = 5
    num_epochs = 500

    # weighting of losses
    semantic_loss_weight = 0.3
    stem_loss_weight = 0.7
    stem_classification_loss_weight = 0.05
    stem_regression_loss_weight = 0.95

    # class weights for semantic segmentation
    weight_background = 0.05
    weight_weed = 0.8
    weight_sugar_beet = 0.15


    # class weights for stem keypoint detection
    weight_stem_background = 0.1
    weight_stem = 0.9

    size_test_set = 50
    weights_path = None
    # weights_path = "simple_unet.pth"

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    dataset = SugarBeetDataset.from_config()
    # split into train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset_train = torch.utils.data.Subset(dataset, indices)

    # use some images from the train set for testing
    # just for testing, we will probably overfit at some stage
    dataset_test = torch.utils.data.Subset(dataset, indices[-size_test_set:])

    # define training and validation data loaders
    data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False)

    model = SimpleUnet.from_config().to(device)
    if weights_path:
        model = load_weights(model, weights_path)

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

        accumulated_confusion_matrix = np.zeros((4,4))

        for batch_index, batch in enumerate(data_loader_train):
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
            
            print('  loss: {:04f}'.format(loss.item()))
            print('  semantic_loss: {:04f}'.format(semantic_loss.item()))
            print('  stem_loss: {:04f}'.format(stem_loss.item()))
            print('  stem_classification_loss: {:04f}'.format(stem_classification_loss.item()))
            print('  stem_regression_loss: {:04f}'.format(stem_regression_loss.item()))

            loss.backward()
            optimizer.step()


            # accumulate confusion matrix
            accumulated_confusion_matrix += compute_confusion_matrix(semantic_output_batch, semantic_target_batch)

        # save accumulated cm
        plot_confusion_matrix(accumulated_confusion_matrix, 3, normalization=True, title=str(epoch_index))
        
	    #compute IoU and Accuracy at the end of the epoch over the last batch
        mIoU, acc = compute_mIoU_and_Acc(semantic_output_batch, semantic_target_batch, 3)

        print('[Training] mIoU: {:04f}, Accuracy: {:04f}'.format(mIoU, acc))

        # end of epoch
        print('End of epoch. Make checkpoint.')
        cp_dir = make_checkpoint(run_name, log_dir, epoch_index, model)

        # pass test set through network and save example images
        examples_dir = cp_dir/'examples'
        if not examples_dir.exists():
            examples_dir.mkdir()

        averaged_mIoU = 0
        averaged_acc = 0

        test_accumulated_confusion_matrix = np.zeros((4,4))
        for batch_index, batch in enumerate(data_loader_test):
            model.eval()

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

            #compute IoU and Accuracy over every batch
            mIoU, acc = compute_mIoU_and_Acc(semantic_output_batch, semantic_target_batch, 3)
            averaged_mIoU += mIoU
            averaged_acc  += acc
            #accumulate confusion matrix
            test_accumulated_confusion_matrix += compute_confusion_matrix(semantic_output_batch, semantic_target_batch)
        
        plot_confusion_matrix(test_accumulated_confusion_matrix, 3, nomalize=True, title=str(epoch))
        print('[Testing] Averaged mIoU: {:04f}, Averaged Accuracy: {:04f}'.format(np.mean(averaged_mIoU), np.mean(averaged_acc)))


def make_checkpoint(run_name, log_dir, epoch, model):
    cp_dir = log_dir/'{}_cp_{:06d}'.format(run_name, epoch)
    if not cp_dir.exists():
        cp_dir.mkdir()

    weights_path = cp_dir/'{}_{:06d}.pth'.format(run_name, epoch)
    torch.save(model.state_dict(), str(weights_path))

    return cp_dir


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


def load_weights(model, path):
    """Load model weights from .pth file.

    If path is relative, assume weights are in MODELS_DIR/path.

    Note: This funcion contains parts, which were written for other student projects
    conducted by the author.
    """
    path = Path(path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    if not path.is_absolute():
      path = MODELS_DIR/path

    print('Load weights from {}.'.format(path))

    model_dict = torch.load(path, map_location=device)
    model.load_state_dict(model_dict)

    trainable_parameters = filter(lambda parameter: parameter.requires_grad, model.parameters())
    num_trainable_parameters = sum([np.prod(parameter.size()) for parameter in trainable_parameters])
    print('Number of trainable model parameters: {}'.format(num_trainable_parameters))

    return model


if __name__=='__main__':
    main()

