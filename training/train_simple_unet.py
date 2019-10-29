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

from training.model.simple_unet import SimpleUnet
from training.dataloader import SugarBeetDataset
from training import LOGS_DIR


def main():
    learning_rate = 0.001
    batch_size = 6
    # semantic_loss_weight = 1.0
    # stem_loss_weight = 0.0

    num_epochs = 500

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    dataset = SugarBeetDataset.from_config()
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SimpleUnet.from_config().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    semantic_loss_function = nn.CrossEntropyLoss(ignore_index=1, weight=torch.Tensor([0.05, 0.3, 0.65])).to(device)

    now = datetime.datetime.now()
    run_time_string = now.strftime('%b_%d_%Y_%Hh%M')
    run_name = '{}_{}'.format('simple_unet', run_time_string)
    log_dir = LOGS_DIR/run_name
    if not log_dir.exists():
        log_dir.mkdir()

    for epoch_index in range(num_epochs):
        for batch_index, (input_batch, semantic_target_batch) in enumerate(data_loader):
            print('Train batch {}/{} in epoch {}/{}.'.format(batch_index, len(data_loader), epoch_index, num_epochs))

            model.train()
            optimizer.zero_grad()

            input_batch = input_batch.to(device)
            semantic_target_batch = semantic_target_batch.to(device)

            semantic_output_batch, stem_output_batch = model(input_batch)
            loss = semantic_loss_function(semantic_output_batch, semantic_target_batch)

            print('Loss: {:06f}'.format(loss.item()))

            loss.backward()
            optimizer.step()

            if batch_index==0: # len(data_loader)-1:
                # end of epoch, make checkpoint
                print('Checkpoint.')
                cp_dir = make_checkpoint(run_name, log_dir, epoch_index, model)

                # Save output of last batch
                semantic_output_batch = nn.functional.softmax(semantic_output_batch, dim=1)

                examples_dir = cp_dir/'examples'
                if not examples_dir.exists():
                    examples_dir.mkdir()

                num_slices = semantic_output_batch.shape[0]

                for slice_index in range(num_slices):
                    image = make_image(input_batch[slice_index], semantic_output_batch[slice_index])
                    cv2.imwrite(str(examples_dir/'example_{:02}.png'.format(slice_index)), image)


def make_checkpoint(run_name, log_dir, epoch, model):
    cp_dir = log_dir/'{}_cp_{:06d}'.format(run_name, epoch)
    if not cp_dir.exists():
        cp_dir.mkdir()

    weights_path = cp_dir/'{}_{:06d}.pth'.format(run_name, epoch)
    torch.save(model.state_dict(), str(weights_path))

    return cp_dir


def make_image(input_slice, semantic_slice):
    input_slice = input_slice.cpu().detach().numpy()
    semantic_slice = semantic_slice.cpu().detach().numpy()
    height, width = input_slice.shape[-2:]
    image = input_slice.transpose((1, 2, 0))[..., :3]

    sugar_beet_color = np.array([1.0, 0.0, 0.0]).reshape(1, 1, 3)
    weed_color = np.array([0, 0, 1.0]).reshape(1, 1, 3)

    weed_confidence = semantic_slice[1, ...][..., None]
    sugar_beet_confidence = semantic_slice[2, ...][..., None]

    image = np.where(weed_confidence>0.5, weed_color, image)
    image = np.where(sugar_beet_confidence>0.5, sugar_beet_color, image)

    # cv2.imshow('image', image)
    # cv2.waitKey()

    return (255.0*image).astype(np.uint8)


if __name__=='__main__':
    main()

