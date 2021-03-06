"""Script to export model as .onnx file.
"""
import click
from pathlib import Path
import torch

from training.models.model import Model
from training import MODELS_DIR, CUDA_DEVICE_NAME, load_config


@click.command()
@click.argument('architecture_name', type=str, default='hardnet56')
@click.option('-o', '--path-to-output-file', type=click.Path(dir_okay=False, file_okay=True), default=None)
@click.option('-d', '--device', type=str, default=CUDA_DEVICE_NAME)
@click.option('-t', '--file-type', type=str, default='onnx')
def export_model(architecture_name, path_to_output_file, device, file_type):
    assert file_type in ['onnx', 'pt']
    print('Exporting model as .{}.'.format(file_type))

    try:
        model = Model.by_name(architecture_name, phase='deployment', verbose=True)
    except ValueError:
        click.echo("Architechture '{}' is not supported.".format(architecture_name), err=True)

    if path_to_output_file is None:
        # name similar to architecture_name
        path_to_output_file = architecture_name+'.'+file_type

    path_to_output_file = Path(path_to_output_file)
    if not path_to_output_file.is_absolute():
        path_to_output_file = MODELS_DIR/path_to_output_file

    click.echo('Writing result to {}'.format(path_to_output_file))

    # get input size from configuration file
    config = load_config('deployment.yaml')
    input_width = config['input_width']
    input_height = config['input_height']
    input_channels = config['input_channels']
    batch_size = config['batch_size']

    model = model.to(device)
    model.eval()
    model.set_deploy(True) # set flag to add sigmoid, softmax after final layers

    dummy_input = torch.randn(batch_size, input_channels, input_height, input_width, device=device)
    model(dummy_input)

    if file_type=='onnx':
        torch.onnx.export(model=model,
                          args=dummy_input,
                          f=str(path_to_output_file),
                          export_params=True,
                          training=False,
                          input_names=['input'],
                          output_names=['semantic_output', 'stem_keypoint_output', 'stem_offset_output'],
                          verbose=False,
                          opset_version=7) # maximum opset version supported by tensorrt
    elif file_type=='pt':
        traced_script_module = torch.jit.trace(model, dummy_input)
        traced_script_module.save(str(path_to_output_file))


if __name__=='__main__':
    export_model()
