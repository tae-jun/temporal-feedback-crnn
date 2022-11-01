from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
from tfcrnn.config import Config
from ..blocks import BasicBlock, SEBlock, ResSEBlock


class CRNN(nn.Module):
  def __init__(self, config: Config):
    super(CRNN, self).__init__()
    self.config = config
    self.window = config.window
    self.hop = config.window // 2
    self.hidden_size = config.hidden_size
    
    num_features = config.init_features
    self.init_block = nn.Sequential(OrderedDict([
      ('conv0', nn.Conv1d(1, num_features, 3, 3, 1)),
      ('norm0', nn.BatchNorm1d(num_features)),
      ('relu0', nn.ReLU(inplace=True)),
    ]))
    
    num_blocks = int(math.log(config.window, 3)) - 1
    self.blocks = nn.Sequential()
    for i in range(num_blocks):
      out_channels = num_features * 2 if i in [2, num_blocks - 1] else num_features
      if num_blocks == 3:
        out_channels = [128, 256, 512][i]
      
      if config.block == 'basic':
        block = BasicBlock(num_features, out_channels)
      elif config.block == 'se':
        block = SEBlock(num_features, out_channels, config.se_amp)
      elif config.block == 'resse':
        block = ResSEBlock(num_features, out_channels, config.se_amp, dropout=config.dropout_resse)
      else:
        raise Exception(f'Unknown block for RNN: {config.block}')
      
      self.blocks.add_module(f'block{i}', block)
      num_features = out_channels
    
    self.cell = nn.GRUCell(num_features, self.hidden_size)
    self.classifier = nn.Linear(self.hidden_size, config.num_classes)
  
  def forward(self, x):
    logits = []
    hidden = torch.zeros(x.shape[0], self.hidden_size, dtype=x.dtype, device=x.device)
    num_segments = 2 * (x.shape[-1] // self.window) - 1
    for i in range(num_segments):
      s = x[..., i * self.hop:i * self.hop + self.window]
      s = self.init_block(s)
      s = self.blocks(s)
      s = F.adaptive_max_pool1d(s, 1).squeeze(-1)
      
      if self.config.dropout > 0:
        s = F.dropout(s, p=self.config.dropout, training=self.training)
      
      hidden = self.cell(s, hidden)
      logit = self.classifier(hidden)
      logits.append(logit)
    
    return torch.stack(logits), hidden
  
  def forward_variable_length(self, x_batch):
    """This method is faster and better performing than temporal avg."""
    sequence_lengths = [(x.shape[-1] // self.window) * 2 - 1 for x in x_batch]
    # Ensure x_batch sequence length is in descending order
    assert all([len1 >= len2 for len1, len2 in zip(sequence_lengths, sequence_lengths[1:])])
    
    max_sequence_lenth = sequence_lengths[0]
    hidden = torch.zeros(len(x_batch), self.hidden_size, dtype=x_batch[0].dtype, device=x_batch[0].device)
    for i in range(max_sequence_lenth):
      i_compute = [i < l for l in sequence_lengths]
      batch_size = sum(i_compute)
      x_this_batch = x_batch[:batch_size]  # get batch run this time
      s = [x[..., i * self.hop:i * self.hop + self.window] for x in x_this_batch]
      s = torch.stack(s)
      s = self.init_block(s)
      s = self.blocks(s)
      s = F.adaptive_max_pool1d(s, 1).squeeze(-1)
      
      if self.config.dropout > 0:
        s = F.dropout(s, p=self.config.dropout, training=self.training)
      
      hidden[:batch_size] = self.cell(s, hidden[:batch_size])
    
    logit = self.classifier(hidden)
    
    return logit, hidden
