"""Script to get mean and standard deviation of all
images in sugar beet dataset.
"""

import click
from PIL import Image
import numpy as np

from training.dataloader import SugarBeetDataset


@click.command()
def compute_dataset_normalization_parameters():
    dataset = SugarBeetDataset.from_config()
    filenames = dataset.filenames
    num_images = len(filenames)

    mean = np.zeros((4,), dtype=np.longdouble)

    with click.progressbar(filenames, label='Reading images.') as bar:
        for filename in bar:
            rgb_image = np.array(Image.open(dataset.get_path_to_rgb_image(filename)).convert('RGB'))/255.0
            nir_image = np.array(Image.open(dataset.get_path_to_nir_image(filename)).convert('L'))/255.0

            mean[:3] += np.mean(rgb_image.reshape(-1, 3), axis=0)
            mean[3] += np.mean(nir_image)

    mean /= num_images

    variance = np.zeros((4,), dtype=np.longdouble)

    with click.progressbar(filenames, label='Reading images.') as bar:
        for filename in bar:
            rgb_image = np.array(Image.open(dataset.get_path_to_rgb_image(filename)).convert('RGB'))/255.0
            nir_image = np.array(Image.open(dataset.get_path_to_nir_image(filename)).convert('L'))/255.0

            variance[:3] += np.mean(np.square(rgb_image.reshape(-1, 3)-mean[:3].reshape(1, 3)), axis=0)
            variance[3] += np.mean(np.square(nir_image-mean[3]))

    variance /= num_images
    std = np.sqrt(variance)

    click.echo('mean_rgb: [{}, {}, {}]'.format(*list(mean[:3])))
    click.echo('mean_nir: {}'.format(mean[3]))
    click.echo('std_rgb: [{}, {}, {}]'.format(*list(std)))
    click.echo('std_nir: {}'.format(std[3]))

    click.echo('Done.')


if __name__=='__main__':
    compute_dataset_normalization_parameters()

