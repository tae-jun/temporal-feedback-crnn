from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
from tfcrnn.config import Config
from ..blocks import BasicBlock, SEBlock, ResSEBlock


class CNN(nn.Module):
  def __init__(self, config: Config):
    super(CNN, self).__init__()
    self.config = config
    
    num_features = config.init_features
    self.init_block = nn.Sequential(OrderedDict([
      ('conv0', nn.Conv1d(1, num_features, 3, 3, 1)),
      ('norm0', nn.BatchNorm1d(num_features)),
      ('relu0', nn.ReLU(inplace=True)),
    ]))
    
    self.blocks = nn.Sequential()
    
    num_blocks = int(round(math.log(config.input_size, 3), 5)) - 1
    for i in range(num_blocks):
      out_channels = num_features * 2 if i in [2, num_blocks - 1] else num_features
      if config.block == 'basic':
        self.blocks.add_module(f'block{i}', BasicBlock(num_features, out_channels))
      elif config.block == 'se':
        self.blocks.add_module(f'block{i}', SEBlock(num_features, out_channels, config.se_amp))
      elif config.block == 'resse':
        self.blocks.add_module(f'block{i}',
                               ResSEBlock(num_features, out_channels, config.se_amp, dropout=config.dropout_resse))
      else:
        raise Exception(f'Unknown block: {config.block}')
      num_features = out_channels
    
    self.aggregation = nn.AdaptiveMaxPool1d(1)
    self.classifier = nn.Linear(num_features, config.num_classes)
  
  def forward(self, x):
    x = self.embed(x)
    if self.config.dropout > 0.:
      x = F.dropout(x, p=self.config.dropout, training=self.training)
    x = self.classifier(x)
    return x
  
  def embed(self, x):
    x = self.init_block(x)
    x = self.blocks(x)
    x = self.aggregation(x).squeeze(-1)
    return x
  
  def forward_variable_length(self, x_batch, F_prob=torch.sigmoid):
    """Aggregate features temporally"""
    segment_counts = [2 * (x.shape[-1] // self.config.input_size) - 1 for x in x_batch]
    max_num_segments = segment_counts[0]
    input_hop = self.config.input_size // 2
    embed_size = self.classifier.in_features
    feature_sum = torch.zeros(len(x_batch), embed_size, dtype=x_batch[0].dtype, device=x_batch[0].device)
    prob_sum = torch.zeros(len(x_batch), self.config.num_classes, dtype=x_batch[0].dtype, device=x_batch[0].device)
    
    for i in range(max_num_segments):
      i_compute = [i < n for n in segment_counts]
      batch_size = sum(i_compute)
      
      x_this_batch = x_batch[:batch_size]  # get batch run this time
      segment_batch = [x[:, i * input_hop:i * input_hop + self.config.input_size] for x in x_this_batch]
      segment_batch = torch.stack(segment_batch)
      
      embed = self.embed(segment_batch)
      logit = self.classifier(embed)
      
      feature_sum[:batch_size] += embed
      prob_sum[:batch_size] += F_prob(logit)
    
    feature_avg = feature_sum / torch.Tensor(segment_counts).unsqueeze(-1).to(feature_sum.device)
    prob_avg = prob_sum / torch.Tensor(segment_counts).unsqueeze(-1).to(prob_sum.device)
    logit = self.classifier(feature_avg)
    
    return logit, feature_avg, prob_avg
