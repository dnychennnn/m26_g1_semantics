"""
Authors: Yung-Yu Chen, Jan Quakernack

Note: Parts adapted from code originally written for MGE-MSR-P-S.
"""
import torch
from torch import nn
import datetime
import numpy as np
import cv2
import yaml
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from training.models.model import Model
from training.dataloader import SugarBeetDataset
from training.losses import StemClassificationLoss, StemRegressionLoss
from training import vis
from training import LOGS_DIR, MODELS_DIR, CUDA_DEVICE_NAME, load_config
from training.evalmetrics import (compute_confusion_matrix,
                                  compute_stem_metrics,
                                  compute_metrics_from_confusion_matrix,
                                  plot_confusion_matrix)
from training.postprocessing.stem_extraction import StemExtraction
from training.postprocessing.semantic_labeling import make_classification_map

class Trainer:

    @classmethod
    def from_config(cls, architecture_name):
        config = load_config(architecture_name+'.yaml')

        # additional parameters for training
        training_config = load_config('training.yaml')
        config.update(training_config)

        dataset_train = SugarBeetDataset.from_config(architecture_name, 'train')
        dataset_val = SugarBeetDataset.from_config(architecture_name, 'val')

        model = Model.by_name(architecture_name=architecture_name,
                              phase='training',
                              verbose=True)

        trainer_parameters = {**config}

        trainer_parameters['dataset_train'] = dataset_train
        trainer_parameters['dataset_val'] = dataset_val
        trainer_parameters['model'] = model

        return Trainer(**trainer_parameters)


    def __init__(self,
                 architecture_name,
                 learning_rate,
                 batch_size,
                 num_epochs,
                 model,
                 dataset_train,
                 dataset_val,
                 input_channels,
                 input_height,
                 input_width,
                 target_height,
                 target_width,
                 semantic_loss_weight,
                 stem_loss_weight,
                 stem_classification_loss_weight,
                 stem_regression_loss_weight,
                 weight_background,
                 weight_weed,
                 weight_sugar_beet,
                 weight_stem_background,
                 weight_stem,
                 keypoint_radius,
                 stem_inference_device_option,
                 stem_inference_kernel_size_votes,
                 stem_inference_kernel_size_peaks,
                 stem_inference_threshold_votes,
                 stem_inference_threshold_peaks,
                 sugar_beet_threshold,
                 weed_threshold,
                 tolerance_radius,
                 **extra_arguments):

        self.architecture_name = architecture_name
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.dataset_train = dataset_train
        self.dataset_val = dataset_val
        self.input_channels = input_channels
        self.input_height = input_height
        self.input_width = input_width
        self.target_height = target_height
        self.target_width = target_width
        self.keypoint_radius = keypoint_radius
        self.tolerance_radius = tolerance_radius
        self.sugar_beet_threshold = sugar_beet_threshold
        self.weed_threshold = weed_threshold

        # init loss weights
        loss_norm = semantic_loss_weight+stem_loss_weight
        stem_loss_norm = stem_classification_loss_weight+stem_regression_loss_weight

        self.semantic_loss_weight = semantic_loss_weight/loss_norm
        self.stem_classification_loss_weight = stem_classification_loss_weight/stem_loss_norm/loss_norm
        self.stem_regression_loss_weight = stem_regression_loss_weight/stem_loss_norm/loss_norm

        # init semantic segmentation class weights
        semantic_norm = weight_background+weight_weed+weight_sugar_beet

        self.weight_background = weight_background/semantic_norm
        self.weight_weed = weight_weed/semantic_norm
        self.weight_sugar_beet = weight_sugar_beet/semantic_norm

        # init stem classification weights
        stem_norm = weight_stem_background+weight_stem

        self.weight_stem_background = weight_stem_background/stem_norm
        self.weight_stem = weight_stem/stem_norm

        # init model
        self.device = torch.device(CUDA_DEVICE_NAME) if torch.cuda.is_available() else torch.device('cpu')
        self.model = model.to(self.device)

        # init losses
        self.semantic_loss_function = nn.CrossEntropyLoss(ignore_index=3,
                weight=torch.Tensor([self.weight_background, self.weight_weed, self.weight_sugar_beet])).to(self.device)
        self.stem_classification_loss_function = StemClassificationLoss(weight_background=weight_stem_background, weight_stem=weight_stem).to(self.device)
        self.stem_regression_loss_function = StemRegressionLoss().to(self.device)

        # init opzimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # init postprocessing

        self.stem_inference_module = StemExtraction(
                input_width=self.target_width,
                input_height=self.target_height,
                keypoint_radius=self.keypoint_radius,
                threshold_votes=stem_inference_threshold_votes,
                threshold_peaks=stem_inference_threshold_peaks,
                kernel_size_votes=stem_inference_kernel_size_votes,
                kernel_size_peaks=stem_inference_kernel_size_peaks,
                device_option=stem_inference_device_option).to(self.device)

        # TODO (optional) have a way to resume training from a given checkpoint


    def train(self, test_run=False, only_eval=False, overfit=False):
        """
        Args:
            test_run (bool): Only do a few steps an show some output to see everything is working.
            overfit (bool): Only use the first training batch to see everything is working.
        """
        # init run name
        now = datetime.datetime.now()
        run_time_string = now.strftime('%b_%d_%Y_%Hh%M')
        self.run_name = '{}_{}'.format(self.architecture_name, run_time_string)

        # init data loader for each split
        self.data_loader_train = torch.utils.data.DataLoader(self.dataset_train,
            batch_size=self.batch_size, shuffle=(False if overfit else True))
        self.data_loader_val = torch.utils.data.DataLoader(self.dataset_val, batch_size=1, shuffle=False)

        if test_run:
            print('Test run. Do not trust metrics!')

        if only_eval:
            # init logs directory for only eval
            self.run_name += '_only_eval'
            self.log_dir = LOGS_DIR/self.run_name

            # checkpoint dir same as logs dir as there is only one checkpint
            self.current_checkpoint_index = 0
            self.current_checkpoint_dir = self.log_dir/self.run_name
            self.current_checkpoint_name = self.current_checkpoint_dir.name
            if not self.current_checkpoint_dir.exists():
                self.current_checkpoint_dir.mkdir(parents=True)

            # init tensorboard summary writer
            self.summary_writer = SummaryWriter(str(self.log_dir))

            # evaluate
            self.evaluate_on_checkpoint(test_run)

            return

        if overfit:
            print('Overfit. Do not trust metrics!')
            self.run_name += '_overfit'
            first_batch = next(iter(self.data_loader_train))

        # init logs directory for training
        self.log_dir = LOGS_DIR/self.run_name
        if not self.log_dir.exists():
            self.log_dir.mkdir()

        # start at the first checkpoint
        self.current_checkpoint_index = 0
        self.current_checkpoint_name = None
        self.current_checkpoint_dir = None

        # init tensorboad summary writer
        self.summary_writer = SummaryWriter(str(self.log_dir))

        for epoch_index in range(self.num_epochs):
            accumulated_losses_train = {}

            for batch_index, (input_batch, target_batch) in enumerate(self.data_loader_train):
                print('Train batch {}/{} in epoch {}/{}.'.format(batch_index, len(self.data_loader_train), epoch_index, self.num_epochs))

                self.model.train()
                self.optimizer.zero_grad()

                if overfit:
                    # use the first batch instead of the current
                    input_batch, target_batch = first_batch

                # unpack batch
                semantic_target_batch = target_batch['semantic']
                stem_keypoint_target_batch = target_batch['stem_keypoint']
                stem_offset_target_batch = target_batch['stem_offset']

                # bring to device
                input_batch = input_batch.to(self.device)
                semantic_target_batch = semantic_target_batch.to(self.device)
                stem_keypoint_target_batch = stem_keypoint_target_batch.to(self.device)
                stem_offset_target_batch = stem_offset_target_batch.to(self.device)

                # debug output
                # input_image = vis.tensor_to_false_color(input_batch[0, :3], input_batch[0, 3],
                                                        # **self.dataset_train.normalization_rgb_dict,
                                                        # **self.dataset_train.normalization_nir_dict)
                # cv2.imshow('input', input_image)
                # cv2.waitKey()

                # foward pass
                semantic_output_batch, stem_keypoint_output_batch, stem_offset_output_batch = self.model(input_batch)

                # backward pass
                losses = self.compute_losses(semantic_output_batch=semantic_output_batch,
                                             stem_keypoint_output_batch=stem_keypoint_output_batch,
                                             stem_offset_output_batch=stem_offset_output_batch,
                                             semantic_target_batch=semantic_target_batch,
                                             stem_keypoint_target_batch=stem_keypoint_target_batch,
                                             stem_offset_target_batch=stem_offset_target_batch)
                losses['loss'].backward()
                self.optimizer.step()

                for key, value in losses.items():
                    print("  Loss '{}': {:04f}".format(key, value.item()))

                self.accumulate_losses(losses, accumulated_losses_train)

                if overfit:
                    # show the overfitted output so we have a clue if things are working
                    self.show_images_for_debugging(input_slice=input_batch[0],
                        semantic_output=semantic_output_batch[0],
                        semantic_target=semantic_target_batch[0],
                        stem_keypoint_output=stem_keypoint_output_batch[0],
                        stem_offset_output=stem_offset_output_batch[0])
                    cv2.waitKey(1)

                if test_run or only_eval:
                    break

            # end of epoch
            print('End of epoch. Make checkpoint. Evaluate.')
            self.make_checkpoint(accumulated_losses_train=accumulated_losses_train,
                                 test_run=test_run)

            if test_run:
                break


    def compute_losses(self,
                       semantic_output_batch,
                       stem_keypoint_output_batch,
                       stem_offset_output_batch,
                       semantic_target_batch,
                       stem_keypoint_target_batch,
                       stem_offset_target_batch):

        # compute losses
        semantic_loss = (self.semantic_loss_weight
                         *self.semantic_loss_function(semantic_output_batch,
                                                      semantic_target_batch))

        stem_classification_loss = (self.stem_classification_loss_weight
                                    *self.stem_classification_loss_function(stem_keypoint_output_batch=stem_keypoint_output_batch,
                                                                            stem_keypoint_target_batch=stem_keypoint_target_batch))

        stem_regression_loss = (self.stem_regression_loss_weight
                                *self.stem_regression_loss_function(stem_offset_output_batch=stem_offset_output_batch,
                                                                    stem_keypoint_target_batch=stem_keypoint_target_batch,
                                                                    stem_offset_target_batch=stem_offset_target_batch))

        stem_loss = stem_classification_loss+stem_regression_loss
        loss = semantic_loss+stem_loss

        return {'semantic_loss': semantic_loss,
                'stem_classification_loss': stem_classification_loss,
                'stem_regression_loss': stem_regression_loss,
                'stem_loss': stem_loss,
                'loss': loss}


    def accumulate_losses(self, losses, accumulated_losses):
        for key, value in losses.items():
            if not key in accumulated_losses:
                accumulated_losses[key] = 0.0
            accumulated_losses[key] += value.item()


    def show_images_for_debugging(self, input_slice, semantic_output, semantic_target, stem_keypoint_output, stem_offset_output):
        image_false_color = vis.tensor_to_false_color(input_slice[:3], input_slice[3],
            **self.dataset_train.normalization_rgb_dict, **self.dataset_train.normalization_nir_dict)
        plot_semantics = vis.make_plot_from_semantic_output(input_rgb=input_slice[:3],
                                                            input_nir=input_slice[3],
                                                            semantic_output=semantic_output,
                                                            semantic_target=semantic_target,
                                                            apply_softmax=True,
                                                            **self.dataset_train.normalization_rgb_dict,
                                                            **self.dataset_train.normalization_nir_dict)

        plot_stem_keypoint_offset = vis.make_plot_from_stem_keypoint_offset_output(input_rgb=input_slice[:3],
                                                                                   input_nir=input_slice[3],
                                                                                   stem_keypoint_output=stem_keypoint_output,
                                                                                   stem_offset_output=stem_offset_output,
                                                                                   keypoint_radius=self.keypoint_radius,
                                                                                   apply_sigmoid=True,
                                                                                   apply_tanh=False,
                                                                                   **self.dataset_train.normalization_rgb_dict,
                                                                                   **self.dataset_train.normalization_nir_dict)

        cv2.imshow('input', image_false_color)
        cv2.imshow('semantics', plot_semantics)
        cv2.imshow('stem_keypoint_offsets', plot_stem_keypoint_offset)


    def make_checkpoint(self,
                        accumulated_losses_train,
                        test_run):
        """
        Args:
            accumulated_confusion_matrix_train (np.array): From training phase.
            test_run (bool): Only do a few steps an show some output to see everything is working.
        """
        self.current_checkpoint_name = '{}_checkpoint_{:06d}'.format(self.run_name, self.current_checkpoint_index)

        self.current_checkpoint_dir = self.log_dir/self.current_checkpoint_name
        if not self.current_checkpoint_dir.exists():
            self.current_checkpoint_dir.mkdir()

        # save model weights
        # TODO (optional) only save best model to save some space
        weights_path = self.current_checkpoint_dir/(self.current_checkpoint_name+'.pth')
        torch.save(self.model.state_dict(), str(weights_path))

        # average and save train losses
        losses_train_norm = len(self.data_loader_train)+1e-6
        losses_train = {key: value/losses_train_norm for key, value in accumulated_losses_train.items()}

        print('Write average losses to tensorboard log.')
        for key, value in losses_train.items():
            self.summary_writer.add_scalar('{}/train'.format(key),
                                           value,
                                           global_step=self.current_checkpoint_index)

        print('Write average losses to yaml.')
        self.write_losses_to_file(path=self.current_checkpoint_dir,
                                  losses=losses_train,
                                  filename=self.current_checkpoint_name+'_losses_training.yaml')

        print("Evaluate on 'val' split.")
        self.evaluate_on_checkpoint(test_run=test_run)

        self.current_checkpoint_index += 1



    def write_losses_to_file(self, path, losses, filename):
        losses_dir_path = path/'losses'
        if not losses_dir_path.exists():
            losses_dir_path.mkdir()
        losses_path = losses_dir_path/filename

        with losses_path.open('w+') as yaml_file:
            yaml.dump(losses, yaml_file)


    def write_metrics_to_file(self, path, metrics, filename, class_names=['background', 'weed', 'sugar_beet']):
        metrics['class_names'] = class_names

        metrics_dir_path = path/'metrics'
        if not metrics_dir_path.exists():
            metrics_dir_path.mkdir()
        metrics_path = metrics_dir_path/filename

        with metrics_path.open('w+') as yaml_file:
            yaml.dump(metrics, yaml_file)


    def evaluate_on_checkpoint(self, test_run):
        """
        Args:
            test_run (bool): Only do a few steps an show some output to see everything is working.
        """
        # folder to save some example images
        examples_dir = self.current_checkpoint_dir/'examples'
        if not examples_dir.exists():
            examples_dir.mkdir()

        accumulated_losses_val = {}
        accumulated_confusion_matrix_val = np.zeros((3, 3), np.long)
        accumulated_confusion_matrix_stem_val = np.zeros((2,2), np.long)
        accumulated_deviation_val = 0.0

        for batch_index, (input_batch, target_batch) in enumerate(self.data_loader_val):
            self.model.eval()

            # unpack batch
            semantic_target_batch = target_batch['semantic']
            stem_keypoint_target_batch = target_batch['stem_keypoint']
            stem_offset_target_batch = target_batch['stem_offset']
            stem_position_target_batch = target_batch['stem_position']
            stem_count_target_batch = target_batch['stem_count']

            input_batch = input_batch.to(self.device)
            semantic_target_batch = semantic_target_batch.to(self.device)
            stem_keypoint_target_batch = stem_keypoint_target_batch.to(self.device)
            stem_offset_target_batch = stem_offset_target_batch.to(self.device)

            stem_position_target_list = self.stem_positions_to_list(stem_position_target_batch,
                                                                    stem_count_target_batch)

            if test_run:
                image_false_color = vis.tensor_to_false_color(input_batch[0, :3], input_batch[0, 3],
                        **self.dataset_val.normalization_rgb_dict, **self.dataset_val.normalization_nir_dict)
                cv2.imshow('input', image_false_color)
                # cv2.waitKey()

            # foward pass
            semantic_output_batch, stem_keypoint_output_batch, stem_offset_output_batch = self.model(input_batch)

            # compute losses
            losses = self.compute_losses(semantic_output_batch=semantic_output_batch,
                                         stem_keypoint_output_batch=stem_keypoint_output_batch,
                                         stem_offset_output_batch=stem_offset_output_batch,
                                         semantic_target_batch=semantic_target_batch,
                                         stem_keypoint_target_batch=stem_keypoint_target_batch,
                                         stem_offset_target_batch=stem_offset_target_batch)

            self.accumulate_losses(losses, accumulated_losses_val)

            # postprocessing
            stem_position_output_batch = self.stem_inference_module(stem_keypoint_output_batch, stem_offset_output_batch)
            stem_position_target_batch = self.stem_inference_module(stem_keypoint_target_batch, stem_offset_target_batch)

            path_for_plots = examples_dir/'sample_{:02d}'.format(batch_index)
            self.make_plots(path_for_plots,
                            input_slice=input_batch[0],
                            semantic_output=semantic_output_batch[0],
                            semantic_target=semantic_target_batch[0],
                            semantic_predicted=make_classification_map(semantic_output_batch, self.sugar_beet_threshold, self.weed_threshold)[0],
                            stem_keypoint_output=stem_keypoint_output_batch[0],
                            stem_offset_output=stem_offset_output_batch[0],
                            stem_position_output=stem_position_output_batch[0],
                            stem_position_target=stem_position_target_list[0],
                            test_run=test_run)

            accumulated_confusion_matrix_val += compute_confusion_matrix(semantic_output_batch,
                    semantic_target_batch, self.sugar_beet_threshold, self.weed_threshold)

            # comute stem metrics
            confusion_matrix_stem_val, deviation_val = compute_stem_metrics(stem_position_output_batch,
                    stem_position_target_list, tolerance_radius=self.tolerance_radius)

            accumulated_confusion_matrix_stem_val += confusion_matrix_stem_val
            accumulated_deviation_val += deviation_val

            # debug
            if test_run and batch_index==3:
                break

        # average and save val losses

        losses_val_norm = len(self.data_loader_val)+1e-6
        losses_val = {key: value/losses_val_norm for key, value in accumulated_losses_val.items()}

        print("Write average losses of 'val' split to tensorboard log.")
        for key, value in losses_val.items():
            self.summary_writer.add_scalar('{}/val'.format(key),
                                           value,
                                           global_step=self.current_checkpoint_index)

        print("Write average losses of 'val' split to yaml.")
        self.write_losses_to_file(path=self.current_checkpoint_dir,
                                  losses=losses_val,
                                  filename=self.current_checkpoint_name+'_losses_val.yaml')


        print("Save confusion matrix of 'val' split.")
        plot_confusion_matrix(self.current_checkpoint_dir,
                              accumulated_confusion_matrix_val,
                              normalize=False,
                              filename=self.current_checkpoint_name+'_val.png')

        plot_confusion_matrix(self.current_checkpoint_dir,
                              accumulated_confusion_matrix_stem_val,
                              normalize=False,
                              class_names=['stem', 'no_stem'],
                              filename=self.current_checkpoint_name+'_stem_val.png')

        # calculate metrics on accumulated confusion matrix
        metrics_val = compute_metrics_from_confusion_matrix(accumulated_confusion_matrix_val)

        # NOTE we do not have a valid number of false negatives for stem detection, so some metrics computed here will not be valid
        metrics_stem_val = compute_metrics_from_confusion_matrix(accumulated_confusion_matrix_stem_val)

        mean_accuracy = np.mean(np.asarray(metrics_val['accuracy'])[1:]) # without background
        mean_iou = np.mean(np.asarray(metrics_val['iou'])[1:]) # without background

        print("Write metrics of split 'val' to tensorboard log.")
        self.summary_writer.add_scalar('iou_background/val', metrics_val['iou'][0], global_step=self.current_checkpoint_index)
        self.summary_writer.add_scalar('iou_weed/val', metrics_val['iou'][1], global_step=self.current_checkpoint_index)
        self.summary_writer.add_scalar('iou_sugar_beet/val', metrics_val['iou'][2], global_step=self.current_checkpoint_index)
        self.summary_writer.add_scalar('mean_iou/val', metrics_val['iou'][2], global_step=self.current_checkpoint_index)

        print("Write metrics of split 'val' to yaml.")
        self.write_metrics_to_file(self.current_checkpoint_dir,
                                   metrics_val, filename=self.current_checkpoint_name+'_val.yaml')
        self.write_metrics_to_file(self.current_checkpoint_dir,
                                   metrics_stem_val, class_names=['stem', 'no_stem'],
                                   filename=self.current_checkpoint_name+'_stem_val.yaml')

        print('  Mean IoU (without background): {:.04f}'.format(mean_iou))
        print("  IoU 'background': {:.04f}".format(metrics_val['iou'][0]))
        print("  IoU 'weed': {:.04f}".format(metrics_val['iou'][1]))
        print("  IoU 'sugar beet': {:.04f}".format(metrics_val['iou'][2]))
        print('  Mean accuracy (without background): {:.04f}'.format(mean_accuracy))
        print("  Accuracy 'background': {:.04f}".format(metrics_val['accuracy'][0]))
        print("  Accuracy 'weed': {:.04f}".format(metrics_val['accuracy'][1]))
        print("  Accuracy 'sugar beet': {:.04f}".format(metrics_val['accuracy'][2]))
        print("  Accuracy stem detection with {:.01f} px tolerance: {:.04f}".format(
            self.tolerance_radius, metrics_stem_val['accuracy'][0]))
        print("  Mean deviation stems within {:.01f} px tolerance: {:.04f} px".format(
            self.tolerance_radius, accumulated_deviation_val/(accumulated_confusion_matrix_stem_val[0, 0]+1e-6)))


    def stem_positions_to_list(self, stem_position_target_batch, stem_count_target_batch):
        """Convert stem positions tensor into a list of position per batch.
        """
        stem_position_target_list = []
        batch_size = stem_position_target_batch.shape[0]
        for index_in_batch in range(batch_size):
            stem_count_target = stem_count_target_batch[index_in_batch]
            stem_position_target_list.append(stem_position_target_batch[index_in_batch, :stem_count_target])
        return stem_position_target_list


    def make_plots(self, path, input_slice, semantic_output, semantic_target, semantic_predicted, stem_keypoint_output, stem_offset_output, stem_position_output, stem_position_target, test_run):
        """Make plots and write images.
        """
        image_bgr = vis.tensor_to_bgr(input_slice[:3], **self.dataset_val.normalization_rgb_dict)
        image_nir = vis.tensor_to_single_channel(input_slice[3], **self.dataset_val.normalization_nir_dict)
        image_false_color = vis.tensor_to_false_color(input_slice[:3], input_slice[3],
                **self.dataset_val.normalization_rgb_dict, **self.dataset_val.normalization_nir_dict)

        plot_semantics = vis.make_plot_from_semantic_output(input_rgb=input_slice[:3],
                                                            input_nir=input_slice[3],
                                                            semantic_output=semantic_output,
                                                            semantic_target=None,
                                                            apply_softmax=False,
                                                            **self.dataset_val.normalization_rgb_dict,
                                                            **self.dataset_val.normalization_nir_dict)

        plot_semantics_target_labels = vis.make_plot_from_semantic_labels(input_rgb=input_slice[:3],
                                                                          input_nir=input_slice[3],
                                                                          semantic_labels=semantic_target,
                                                                          **self.dataset_val.normalization_rgb_dict,
                                                                          **self.dataset_val.normalization_nir_dict)

        plot_semantics_predicted_labels = vis.make_plot_from_semantic_labels(input_rgb=input_slice[:3],
                                                                             input_nir=input_slice[3],
                                                                             semantic_labels=semantic_predicted,
                                                                             **self.dataset_val.normalization_rgb_dict,
                                                                             **self.dataset_val.normalization_nir_dict)

        plot_stems_keypoint_offset = vis.make_plot_from_stem_keypoint_offset_output(input_rgb=input_slice[:3],
                                                                                    input_nir=input_slice[3],
                                                                                    stem_keypoint_output=stem_keypoint_output,
                                                                                    stem_offset_output=stem_offset_output,
                                                                                    keypoint_radius=self.keypoint_radius,
                                                                                    apply_sigmoid=False,
                                                                                    apply_tanh=False,
                                                                                    **self.dataset_val.normalization_rgb_dict,
                                                                                    **self.dataset_val.normalization_nir_dict)

        plot_stems = vis.make_plot_from_stem_output(input_rgb=input_slice[:3],
                                                    input_nir=input_slice[3],
                                                    stem_position_output=stem_position_output,
                                                    stem_position_target=stem_position_target,
                                                    keypoint_radius=self.keypoint_radius,
                                                    target_width=stem_keypoint_output.shape[-1],
                                                    target_height=stem_keypoint_output.shape[-2],
                                                    **self.dataset_val.normalization_rgb_dict,
                                                    **self.dataset_val.normalization_nir_dict)

        path_rgb = path.parent/(path.name+'_rgb.jpg')
        path_nir = path.parent/(path.name+'_nir.jpg')
        path_false_color = path.parent/(path.name+'_false_color.jpg')
        path_semantics = path.parent/(path.name+'_semantics.jpg')
        path_semantics_target_labels = path.parent/(path.name+'_semantics_target_labels.jpg')
        path_semantics_predicted_labels = path.parent/(path.name+'_semantics_predicted_labels.jpg')
        path_stems_keypoint_offset = path.parent/(path.name+'_stems_keypoint_offset.jpg')
        path_stems = path.parent/(path.name+'_stems.jpg')

        if test_run:
            cv2.imshow('semantics', plot_semantics)
            cv2.imshow('semantics_target_labels', plot_semantics_target_labels)
            cv2.imshow('semantics_predicted_labels', plot_semantics_predicted_labels)
            cv2.imshow('stems_keypoint_offset', plot_stems_keypoint_offset)
            cv2.imshow('stems', plot_stems)
            # cv2.imshow('image_bgr', image_bgr)
            # cv2.imshow('image_nir', image_nir)
            # cv2.imshow('image_false_color', image_false_color)
            cv2.waitKey(0)

        cv2.imwrite(str(path_rgb), (255.0*image_bgr).astype(np.uint8))
        cv2.imwrite(str(path_nir), (255.0*image_nir).astype(np.uint8))
        cv2.imwrite(str(path_false_color), (255.0*image_false_color).astype(np.uint8))
        cv2.imwrite(str(path_semantics), (255.0*plot_semantics).astype(np.uint8))
        cv2.imwrite(str(path_semantics_target_labels), (255.0*plot_semantics_target_labels).astype(np.uint8))
        cv2.imwrite(str(path_semantics_predicted_labels), (255.0*plot_semantics_predicted_labels).astype(np.uint8))
        cv2.imwrite(str(path_stems_keypoint_offset), (255.0*plot_stems_keypoint_offset).astype(np.uint8))
        cv2.imwrite(str(path_stems), (255.0*plot_stems).astype(np.uint8))

