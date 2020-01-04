"""Script to train a network.
"""
import click
from pathlib import Path
import torch

from training.trainer import Trainer


@click.command()
@click.argument('architecture-name', type=str, default='densenet56')
@click.option('--test-run/--no-test-run', default=False)
@click.option('--only-eval/--no-only-eval', default=False)
@click.option('--overfit/--no-overfit', default=False)
def train(architecture_name, test_run, only_eval, overfit):
    trainer = Trainer.from_config(architecture_name=architecture_name)
    trainer.train(test_run=test_run, only_eval=only_eval, overfit=overfit)


if __name__=='__main__':
    train()
