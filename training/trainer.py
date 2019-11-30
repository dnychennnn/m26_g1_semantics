"""
Authors: Yung-Yu Chen, Jan Quakernack

Note: Parts adapted from code originally written for MGE-MSR-P-S.
"""
import torch
from torch import nn
import datetime
import numpy as np
import cv2
from pathlib import Path

from training.models.model import Model
from training.dataloader import SugarBeetDataset
from training.losses import StemClassificationLoss, StemRegressionLoss
from training import vis
from training import LOGS_DIR, MODELS_DIR, CUDA_DEVICE_NAME, load_config
from training.evalmetrics import compute_confusion_matrix, compute_metrics_from_confusion_matrix, plot_confusion_matrix, write_metrics_to_file
from training.postprocessing.stem_inference import StemInference

class Trainer:

    @classmethod
    def from_config(cls):
        config = load_config('training.yaml')

        dataset = SugarBeetDataset.from_config()
        model = Model.by_name(architecture_name=config['architecture_name'],
                              phase='training',
                              path_to_weights_file=config['path_to_weights_file'],
                              verbose=True)

        trainer_parameters = {**config}
        trainer_parameters['dataset'] = dataset
        trainer_parameters['model'] = model
        del trainer_parameters['architecture_name']
        del trainer_parameters['path_to_weights_file']

        return Trainer(**trainer_parameters)


    def __init__(self,
                 learning_rate,
                 batch_size,
                 num_epochs,
                 model,
                 dataset,
                 input_channels,
                 input_height,
                 input_width,
                 target_height,
                 target_width,
                 size_test_set,
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
                 stem_inference_threshold_peaks):

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.input_channels = input_channels
        self.input_height = input_height
        self.input_width = input_width
        self.target_height = target_height
        self.target_width = target_width
        self.size_test_set = size_test_set
        self.semantic_loss_weight = semantic_loss_weight
        self.stem_loss_weight = stem_loss_weight
        self.stem_classification_loss_weight = stem_classification_loss_weight
        self.stem_regression_loss_weight = stem_regression_loss_weight
        self.weight_background = weight_background
        self.weight_weed = weight_weed
        self.weight_sugar_beet = weight_sugar_beet
        self.weight_stem_background = weight_stem_background
        self.weight_stem = weight_stem
        self.keypoint_radius = keypoint_radius

        self.device = torch.device(CUDA_DEVICE_NAME) if torch.cuda.is_available() else torch.device('cpu')

        self.dataset = dataset
        self.model = model.to(self.device)

        # split dataset into train and test set
        seed = 0 # reproducible order
        random = np.random.RandomState(seed)
        indices = random.permutation(len(dataset)).tolist()
        dataset_train = torch.utils.data.Subset(self.dataset, indices[:-size_test_set])
        dataset_test = torch.utils.data.Subset(self.dataset, indices[-size_test_set:])

        # init data loaders for splits
        self.data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=self.batch_size, shuffle=True)
        self.data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False)

        # init losses
        self.semantic_loss_function = nn.CrossEntropyLoss(ignore_index=3,
                weight=torch.Tensor([self.weight_background, self.weight_weed, self.weight_sugar_beet])).to(self.device)
        self.stem_classification_loss_function = StemClassificationLoss(weight_background=weight_stem_background, weight_stem=weight_stem).to(self.device)
        self.stem_regression_loss_function = StemRegressionLoss().to(self.device)

        # init opzimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # init postprocessing
        self.stem_inference_module = StemInference(self.target_width, self.target_height,
                keypoint_radius=self.keypoint_radius,
                threshold_votes=stem_inference_threshold_votes,
                threshold_peaks=stem_inference_threshold_peaks,
                kernel_size_votes=stem_inference_kernel_size_votes,
                kernel_size_peaks=stem_inference_kernel_size_peaks,
                device_option=stem_inference_device_option).to(self.device)

        # init logs directory
        now = datetime.datetime.now()
        run_time_string = now.strftime('%b_%d_%Y_%Hh%M')
        self.run_name = '{}_{}'.format('simple_unet', run_time_string)
        self.log_dir = LOGS_DIR/self.run_name
        if not self.log_dir.exists():
            self.log_dir.mkdir()

        # start at the first checkpoint
        self.current_checkpoint_index = 0
        self.current_checkpoint_name = None
        self.current_checkpoint_dir = None

        # TODO (optional) have a way to resume training from a given checkpoint


    def train(self, test_run=False):
        """
        Args:
            test_run (bool): Only do a few steps an show some output to see everything is working.
        """
        if test_run:
            print('Test run. Do not trust metrics!')

        for epoch_index in range(self.num_epochs):
            accumulated_confusion_matrix_train = np.zeros((3, 3,), dtype=np.long)

            for batch_index, batch in enumerate(self.data_loader_train):
                print('Train batch {}/{} in epoch {}/{}.'.format(batch_index, len(self.data_loader_train), epoch_index, self.num_epochs))

                self.model.train()
                self.optimizer.zero_grad()

                # get input an bring to device
                input_batch, semantic_target_batch, stem_keypoint_target_batch, stem_offset_target_batch = batch

                input_batch = input_batch.to(self.device)
                semantic_target_batch = semantic_target_batch.to(self.device)
                stem_keypoint_target_batch = stem_keypoint_target_batch.to(self.device)
                stem_offset_target_batch = stem_offset_target_batch.to(self.device)

                # foward pass
                semantic_output_batch, stem_keypoint_output_batch, stem_offset_output_batch = self.model(input_batch)

                # compute losses
                semantic_loss = self.semantic_loss_weight*\
                                self.semantic_loss_function(semantic_output_batch,
                                                            semantic_target_batch)

                stem_classification_loss = self.stem_loss_weight*self.stem_classification_loss_weight*\
                                           self.stem_classification_loss_function(stem_keypoint_output_batch=stem_keypoint_output_batch,
                                                                                  stem_keypoint_target_batch=stem_keypoint_target_batch)

                stem_regression_loss = self.stem_loss_weight*self.stem_regression_loss_weight*\
                                       self.stem_regression_loss_function(stem_offset_output_batch=stem_offset_output_batch,
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
                self.optimizer.step()

                # accumulate confusion matrix
                accumulated_confusion_matrix_train += compute_confusion_matrix(semantic_output_batch, semantic_target_batch)

                if test_run:
                    break

            # end of epoch
            print('End of epoch. Make checkpoint. Evaluate.')
            self.make_checkpoint(accumulated_confusion_matrix_train, test_run=test_run)

            if test_run:
                break


    def make_checkpoint(self, accumulated_confusion_matrix_train, test_run):
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

        # save accumulated confusion matrix
        print('Save confusion matrix of training.')
        plot_confusion_matrix(self.current_checkpoint_dir, accumulated_confusion_matrix_train, normalize=False, filename=self.current_checkpoint_name+'_training.png')

        # calculate metrics on accumulated confusion matrix
        metrics_train = compute_metrics_from_confusion_matrix(accumulated_confusion_matrix_train)
        print('Calculate metrics of training.')
        write_metrics_to_file(self.current_checkpoint_dir, metrics_train, filename=self.current_checkpoint_name+'_training.yaml')

        self.evaluate_on_checkpoint(test_run=test_run)

        self.current_checkpoint_index += 1


    def evaluate_on_checkpoint(self, test_run):
        """
        Args:
            test_run (bool): Only do a few steps an show some output to see everything is working.
        """
        # folder to save some example images
        examples_dir = self.current_checkpoint_dir/'examples'
        if not examples_dir.exists():
            examples_dir.mkdir()

        accumulated_confusion_matrix_test = np.zeros((3, 3), np.long)

        for batch_index, batch in enumerate(self.data_loader_test):
            self.model.eval()

            input_batch, semantic_target_batch, stem_keypoint_target_batch, stem_offset_target_batch = batch

            input_batch = input_batch.to(self.device)
            semantic_target_batch = semantic_target_batch.to(self.device)
            stem_keypoint_target_batch = stem_keypoint_target_batch.to(self.device)
            stem_offset_target_batch = stem_offset_target_batch.to(self.device)

            if test_run:
                image_false_color = vis.tensor_to_false_color(input_batch[0, :3], input_batch[0, 3],
                        **self.dataset.normalization_rgb_dict, **self.dataset.normalization_nir_dict)
                cv2.imshow('input', image_false_color)
                # cv2.waitKey()

            # foward pass
            semantic_output_batch, stem_keypoint_output_batch, stem_offset_output_batch = self.model(input_batch)

            # postprocessing
            stem_output = self.stem_inference_module(stem_keypoint_output_batch, stem_offset_output_batch)

            path_for_plots = examples_dir/'sample_{:02d}'.format(batch_index)
            self.make_plots(path_for_plots,
                            input_slice=input_batch[0],
                            semantic_output=semantic_output_batch[0],
                            stem_keypoint_output=stem_keypoint_output_batch[0],
                            stem_offset_output=stem_offset_output_batch[0],
                            stem_output=stem_output[0],
                            test_run=test_run)

            # compute IoU and Accuracy over every batch
            accumulated_confusion_matrix_test += compute_confusion_matrix(semantic_output_batch, semantic_target_batch)

            # debug
            if test_run and batch_index==2:
                break

        print('Save confusion matrix of test.')
        plot_confusion_matrix(self.current_checkpoint_dir, accumulated_confusion_matrix_test, normalize=False, filename=self.current_checkpoint_name+'_test.png')

        # calculate metrics on accumulated confusion matrix
        metrics_test = compute_metrics_from_confusion_matrix(accumulated_confusion_matrix_test)
        print('Calculate metrics of test.')
        write_metrics_to_file(self.current_checkpoint_dir, metrics_test, filename=self.current_checkpoint_name+'_test.yaml')

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


    def make_plots(self, path, input_slice, semantic_output, stem_keypoint_output, stem_offset_output, stem_output, test_run):
        """Make plots and write images.
        """
        image_bgr = vis.tensor_to_bgr(input_slice[:3], **self.dataset.normalization_rgb_dict)
        image_nir = vis.tensor_to_single_channel(input_slice[3], **self.dataset.normalization_nir_dict)
        image_false_color = vis.tensor_to_false_color(input_slice[:3], input_slice[3],
                **self.dataset.normalization_rgb_dict, **self.dataset.normalization_nir_dict)
        plot_semantics = vis.make_plot_from_semantic_output(input_rgb=input_slice[:3],
                                                            input_nir=input_slice[3],
                                                            semantic_output=semantic_output,
                                                            apply_softmax=False,
                                                            **self.dataset.normalization_rgb_dict,
                                                            **self.dataset.normalization_nir_dict)

        plot_stems_keypoint_offset = vis.make_plot_from_stem_keypoint_offset_output(input_rgb=input_slice[:3],
                                                                                    input_nir=input_slice[3],
                                                                                    stem_keypoint_output=stem_keypoint_output,
                                                                                    stem_offset_output=stem_offset_output,
                                                                                    keypoint_radius=self.keypoint_radius,
                                                                                    apply_sigmoid=False,
                                                                                    apply_tanh=False,
                                                                                    **self.dataset.normalization_rgb_dict,
                                                                                    **self.dataset.normalization_nir_dict)

        plot_stems = vis.make_plot_from_stem_output(input_rgb=input_slice[:3],
                                                    input_nir=input_slice[3],
                                                    stem_output=stem_output,
                                                    keypoint_radius=self.keypoint_radius,
                                                    **self.dataset.normalization_rgb_dict,
                                                    **self.dataset.normalization_nir_dict)

        path_rgb = path.parent/(path.name+'_rgb.jpg')
        path_nir = path.parent/(path.name+'_nir.jpg')
        path_false_color = path.parent/(path.name+'_false_color.jpg')
        path_semantics = path.parent/(path.name+'_semantics.jpg')
        path_stems_keypoint_offset = path.parent/(path.name+'_stems_keypoint_offset.jpg')
        path_stems = path.parent/(path.name+'_stems.jpg')

        if test_run:
            cv2.imshow('semantics', plot_semantics)
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
        cv2.imwrite(str(path_stems_keypoint_offset), (255.0*plot_stems_keypoint_offset).astype(np.uint8))
        cv2.imwrite(str(path_stems), (255.0*plot_stems).astype(np.uint8))

