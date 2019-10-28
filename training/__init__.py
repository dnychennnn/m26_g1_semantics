"""Set some filepaths for be used in other parts of the package.

Author: Jan Quakernack

Note: This module contains parts, which were written for other student projects
conducted by the author.
"""

import os
import yaml
import warnings
from pathlib import Path

def _load_dir(name):
    try:
        return Path(os.environ[name])
    except KeyError as error:
        warnings.warn("Environment variable '{}' not set.".format(name))
        return None


DATA_DIR = _load_dir('M26_G1_SEMANTICS_DATA_DIR')
CONFIGS_DIR = _load_dir('M26_G1_SEMANTICS_CONFIGS_DIR')
LOGS_DIR = _load_dir('M26_G1_SEMANTICS_LOGS_DIR')
MODELS_DIR = _load_dir('M26_G1_SEMANTICS_MODELS_DIR')


def load_config(path):
    """Load the configuration file (.yaml format) from the given path as a
    dictionary.

    If the path is relative, the configuration file is expected to be found
    under CONFIGS_DIR/path. CONFIGS_DIR can be set through the environment
    variable M26_G1_SEMANTICS_CONFIGS_DIR.

    Usage:
        from training import load_config
        config = load_config('some_config_file.yaml')
        some_parameter_value = config['some_parameter_key']

    Args:
        path (str): Relative or absolute path to a .yaml file.

    Returns:
        A dictionary.
    """

    path = Path(path)
    path = path if path.is_absolute() else CONFIGS_DIR/path
    with path.open('r') as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config
