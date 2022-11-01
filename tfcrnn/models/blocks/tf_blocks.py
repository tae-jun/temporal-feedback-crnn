import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from .plain_blocks import BasicBlock


class TFBasicBlock(nn.Module):
  def __init__(self, in_channels, out_channels, hidden_size, amp_rate):
    super().__init__()
    self.base_block = BasicBlock(in_channels, out_channels)
    self.excitation = nn.Sequential(OrderedDict([
      ('linear0', nn.Linear(hidden_size, int(out_channels * amp_rate))),
      ('relu', nn.ReLU(inplace=True)),
      ('linear1', nn.Linear(int(out_channels * amp_rate), out_channels)),
      ('scale', nn.Sigmoid()),
    ]))
    # Old implementation:
    # self.excitation = nn.Sequential(OrderedDict([
    #   ('linear', nn.Linear(hidden_size, out_channels)),
    #   ('scale', nn.Sigmoid()),
    # ]))
  
  def forward(self, x, hidden):
    x = self.base_block(x)
    scale = self.excitation(hidden)
    return x * scale.unsqueeze(-1)


class TFSEBlock(nn.Module):
  def __init__(self, in_channels, out_channels, hidden_size, amp_rate):
    super().__init__()
    self.base_block = BasicBlock(in_channels, out_channels)
    self.excitation = nn.Sequential(OrderedDict([
      ('linear0', nn.Linear(out_channels + hidden_size, int(out_channels * amp_rate))),
      ('relu', nn.ReLU(inplace=True)),
      ('linear1', nn.Linear(int(out_channels * amp_rate), out_channels)),
      ('scale', nn.Sigmoid()),
    ]))
  
  def forward(self, x, hidden):
    x = self.base_block(x)
    channel_stat = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
    scale = self.excitation(torch.cat([channel_stat, hidden], dim=1))
    return x * scale.unsqueeze(-1)


class TFResSEBlock(nn.Module):
  def __init__(self, in_channels, out_channels, hidden_size, amp_rate, dropout):
    super().__init__()
    
    self.identity = nn.Sequential()
    if in_channels != out_channels:
      self.identity.add_module('conv', nn.Conv1d(in_channels, out_channels, 1))
      self.identity.add_module('norm', nn.BatchNorm1d(out_channels))
    
    self.residual = nn.Sequential()
    self.residual.add_module('conv0', nn.Conv1d(in_channels, out_channels, 3, 1, 1))
    self.residual.add_module('norm0', nn.BatchNorm1d(out_channels))
    self.residual.add_module('relu0', nn.ReLU(inplace=True))
    if dropout > 0.0:
      self.residual.add_module('drop', nn.Dropout(p=dropout))
    self.residual.add_module('conv1', nn.Conv1d(out_channels, out_channels, 3, 1, 1))
    self.residual.add_module('norm1', nn.BatchNorm1d(out_channels))
    
    self.pool = nn.Sequential()
    self.pool.add_module('relu', nn.ReLU(inplace=True))
    self.pool.add_module('pool', nn.MaxPool1d(3))
    
    self.excitation = nn.Sequential(OrderedDict([
      ('linear0', nn.Linear(out_channels + hidden_size, int(out_channels * amp_rate))),
      ('relu', nn.ReLU(inplace=True)),
      ('linear1', nn.Linear(int(out_channels * amp_rate), out_channels)),
      ('scale', nn.Sigmoid()),
    ]))
  
  def forward(self, x, hidden):
    shortcut = self.identity(x)
    x = self.residual(x)
    x = x + shortcut
    x = self.pool(x)
    
    channel_stat = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
    scale = self.excitation(torch.cat([channel_stat, hidden], dim=1))
    return x * scale.unsqueeze(-1)
