from __future__ import annotations

import torch
import torch.nn.functional as F

from glob import glob
from typing import Literal, List
from pedalboard.io import ReadableAudioFile
from torch.utils.data import Dataset
from tfcrnn.utils import mkpath
from tfcrnn.config import Config

CLASSES = ['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 'forward', 'four', 'go', 'happy',
           'house', 'learn', 'left', 'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six',
           'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero']

NAME2IDX = {name: i for i, name in enumerate(CLASSES)}


class SpeechCommandsDataset(Dataset):
  def __init__(
    self,
    split: Literal['train', 'valid', 'test'],
    config: Config,
  ):
    self.split = split
    self.config = config
    
    self.paths = load_audio_paths(config.dataset_dir, split)
  
  def __getitem__(self, i):
    path = self.paths[i]
    label_name = path.split('/')[-2]
    label = NAME2IDX.get(label_name, '_silence_')
    
    with ReadableAudioFile(path) as af:
      assert af.samplerate == self.config.sample_rate, (
        f'The configured sampling rate is {self.config.sample_rate}, '
        f'but got {af.samplerate} from: {path}'
      )
      x = af.read(self.config.input_size)
      x = x.squeeze()
      x = torch.from_numpy(x)
      x = x / 0.0860
    
    if len(x) < self.config.input_size:
      # Pad with zeros if audio is shorter than 16000 (1 sec).
      pad_size = self.config.input_size - len(x)
      x = F.pad(x, (pad_size // 2, pad_size // 2 + pad_size % 2), mode='constant', value=0)
    
    # Apply random cropping for the training set.
    if self.split == 'train' and self.config.crop_size > 0:
      crop_size = self.config.crop_size
      i = torch.randint(0, crop_size, (1,))
      x = x[i:i + x.shape[-1] - crop_size]
      x = F.pad(x, (crop_size // 2, crop_size // 2), mode='constant', value=0)
    
    # Add a dimension for channels.
    x = x.unsqueeze(0)
    
    assert x.shape == (1, self.config.input_size), (
      f'The processed waveform should have a shape of {(1, self.config.input_size)}, '
      f'but got {x.shape}'
    )
    
    return x, label
  
  def __len__(self):
    return len(self.paths)


def load_audio_paths(
  dataset_dir: str,
  split: Literal['train', 'valid', 'test'],
) -> List[str]:
  if split != 'test':
    with open(mkpath(dataset_dir, 'validation_list.txt')) as f:
      valid_paths = f.read().splitlines()
      valid_paths = [mkpath(dataset_dir, path) for path in valid_paths]
      valid_paths.sort()
  
  if split == 'valid':
    return valid_paths
  
  with open(mkpath(dataset_dir, 'testing_list.txt')) as f:
    test_paths = f.read().splitlines()
    test_paths = [mkpath(dataset_dir, path) for path in test_paths]
    test_paths.sort()
  
  if split == 'test':
    return test_paths
  
  audio_paths = glob(mkpath(dataset_dir, '*/*.wav'))
  noise_paths = glob(mkpath(dataset_dir, '_background_noise_/*.wav'))
  
  # Remove validation, test set, and noises from the training set.
  train_paths = list(set(audio_paths) - set(valid_paths) - set(test_paths) - set(noise_paths))
  train_paths.sort()
  
  return train_paths


if __name__ == '__main__':
  from tqdm import tqdm
  from torch.utils.data import DataLoader
  
  print('=> Start sanity check for the dataset')
  config = Config()
  config.parse_cli()
  config.init_wandb()
  config.print()
  
  splits = ['train', 'valid', 'test']
  datasets = [SpeechCommandsDataset(split, config) for split in splits]
  for dataset in datasets:
    loader = DataLoader(dataset, config.batch_size, shuffle=False, drop_last=False, num_workers=0)
    for x, y in tqdm(loader, desc=dataset.split):
      pass
