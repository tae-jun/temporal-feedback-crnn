import os
import argparse
import wandb

from dataclasses import dataclass, asdict
from typing import Literal
from tfcrnn.utils import mkpath


@dataclass
class Config:
  # Path configurations.
  dataset_dir: str = mkpath(os.path.dirname(__file__), '../dataset')
  
  # Data configurations.
  input_seconds: float = 1.0
  sample_rate: int = 16000
  num_classes: int = 35
  crop_milliseconds: int = 10
  
  # Model configurations.
  hidden_size: int = 256
  window: int = 3200  # 200ms, timestep_size=100ms
  skeleton: Literal['tfcrnn', 'cnn', 'crnn'] = 'tfcrnn'
  block: Literal['basic', 'se', 'resse'] = 'basic'
  init_features: int = 128
  se_amp: float = 2 ** -3
  
  # Training configurations.
  batch_size: int = 64
  initial_lr: float = 0.025
  momentum: float = 0.9
  lr_decay: float = 0.2
  loss: Literal['many2many', 'many2one'] = 'many2many'
  num_max_epochs: int = 200  # TODO num_max_epochs
  num_decay: int = 3
  dropout: float = 0.5
  dropout_resse: float = 0.2
  weight_decay: float = 0.0
  patience: int = 2
  gpu: int = 0
  wandb_log_stepsize: int = 100
  
  @property
  def input_size(self):
    return int(self.sample_rate * self.input_seconds)
  
  @property
  def crop_size(self):
    return int(self.sample_rate * self.crop_milliseconds / 1000)
  
  def parse_cli(self):
    parser = argparse.ArgumentParser()
    for k, v in asdict(self).items():
      parser.add_argument(f'--{k}', type=type(v), default=v)
    
    args = parser.parse_args()
    for k, v in vars(args).items():
      setattr(self, k, v)
    
    if wandb.run is not None:
      # wandb is initialized.
      wandb.config.update(asdict(self), allow_val_change=True)
  
  def print(self):
    print('-' * 32 + ' Configurations ' + '-' * 32)
    for k, v in asdict(self).items():
      print(f'{k:20}: {v}')
    print('-' * 80)
  
  def init_wandb(self):
    wandb.init(anonymous='allow')
    if len(wandb.config.keys()) > 0:
      # It's a sweep of W & B.
      for k, v in wandb.config.items():
        setattr(self, k, v)
    wandb.config.update(asdict(self))
  
  def as_dict(self):
    return asdict(self)
