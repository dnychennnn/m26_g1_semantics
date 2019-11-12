"""Script to export model as .onnx file.

Author: Jan Quakernack
"""
import click
from pathlib import Path
import torch

from training.models.model import Model
from training import MODELS_DIR, load_config


@click.command()
@click.argument('architecture_name', type=str)
@click.argument('path_to_weights_file', type=click.Path(dir_okay=False, file_okay=True))
@click.option('-o', '--path-to-output-file', type=click.Path(dir_okay=False, file_okay=True), default=None)
@click.option('-d', '--device', type=str, default='cuda')
def export_model_as_onnx(architecture_name, path_to_weights_file, path_to_output_file, device):
    try:
        model = Model.by_name(architecture_name, phase='deployment', path_to_weights_file=path_to_weights_file, verbose=True)
    except ValueError:
        click.echo("Architechture '{}' is not supported.".format(architecture_name), err=True)

    if path_to_output_file is None:
        # name similar to .pth file
        path_to_output_file = Path(path_to_weights_file).stem+'.onnx'

    path_to_output_file = Path(path_to_output_file)
    if not path_to_output_file.is_absolute():
        path_to_output_file = MODELS_DIR/path_to_output_file

    click.echo('Wrinting result to {}'.format(path_to_output_file))

    # get input size from configuration file
    config = load_config('deployment.yaml')
    input_width = config['input_width']
    input_height = config['input_height']
    input_channels = config['input_channels']
    batch_size = config['batch_size']

    model = model.to(device)
    dummy_input = torch.randn(batch_size, input_channels, input_height, input_width, device=device)
    torch.onnx.export(model, dummy_input, str(path_to_output_file), verbose=True)


if __name__=='__main__':
    export_model_as_onnx()
