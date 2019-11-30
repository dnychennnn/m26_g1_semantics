"""Script to train a network.

Author: Jan Quakernack
"""
import click
from pathlib import Path
import torch

from training.trainer import Trainer


@click.command()
@click.option('--test-run/--train-run', default=False)
def train(test_run):
    trainer = Trainer.from_config()
    trainer.train(test_run=test_run)


if __name__=='__main__':
    train()
