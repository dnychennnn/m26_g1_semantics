import click
from pathlib import Path
import yaml
import numpy as np

from training import DATA_DIR, CONFIGS_DIR


@click.command()
def split_dataset():
    image_folder = DATA_DIR/"sugar_beet_dataset"/"images"/"rgb"
    ext = ".png"
    size_val_split = 100
    size_test_split = 100
    gap = 10 # leave out some images between splits to make sure there is no overlap

    config_path_test = CONFIGS_DIR/"test_split.yaml"
    config_path_val = CONFIGS_DIR/"val_split.yaml"
    config_path_train = CONFIGS_DIR/"train_split.yaml"

    image_paths = list(image_folder.glob("*"+ext))

    keys = []

    for image_path in image_paths:
        splits = image_path.stem.split("_frame")
        start = splits[0].replace("flourish-rng_", "")
        end = "{:08d}".format(int(splits[1]))
        keys.append(start+"_"+end)

    indices = np.argsort(keys)

    sorted_paths = np.array(image_paths)[indices]

    val_split = sorted_paths[:size_val_split]
    test_split = sorted_paths[size_val_split+gap:size_val_split+gap+size_test_split]
    train_split = sorted_paths[size_val_split+gap+size_test_split+gap:]

    test_split = [path.stem for path in test_split]
    val_split = [path.stem for path in val_split]
    train_split = [path.stem for path in train_split]

    with config_path_test.open("w+") as yaml_file:
        yaml.dump(test_split, yaml_file)

    with config_path_val.open("w+") as yaml_file:
        yaml.dump(val_split, yaml_file)

    with config_path_train.open("w+") as yaml_file:
        yaml.dump(train_split, yaml_file)


if __name__=='__main__':
    split_dataset()
