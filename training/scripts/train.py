"""Script to train a network.

Author: Jan Quakernack
"""
import click
from pathlib import Path
import torch

from training.trainer import Trainer


@click.command()
@click.option('--test-run/--no-test-run', default=False)
@click.option('--only-eval/--no-only-eval', default=False)
def train(test_run, only_eval):
    trainer = Trainer.from_config()
    trainer.train(test_run=test_run, only_eval=only_eval)


if __name__=='__main__':
    train()
