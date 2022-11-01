from __future__ import annotations

import numpy as np
import os
import abc
import wandb
import torch
import torch.optim as optim
import torch.nn.functional as F

from torch.optim.lr_scheduler import ReduceLROnPlateau
from tfcrnn.models import CNN, CRNN, TFCRNN
from tfcrnn.config import Config
from tfcrnn.utils import mkpath


class BaseRunner(abc.ABC):
  def __init__(
    self,
    config: Config,
    checkpoint_path: str = None,
  ):
    self.config = config
    self.checkpoint_path = checkpoint_path or mkpath(wandb.run.dir, 'checkpoint.pth')
    self.device = torch.device(f'cuda:{config.gpu}') if torch.cuda.is_available() else torch.device('cpu')
    
    if config.skeleton == 'cnn':
      self.model = CNN(config).to(self.device)
    elif config.skeleton == 'crnn':
      self.model = CRNN(config).to(self.device)
    elif config.skeleton == 'tfcrnn':
      self.model = TFCRNN(config).to(self.device)
    else:
      raise ValueError(f'Unknown skeleton: {config.skeleton}')
    
    self.lr = config.initial_lr
    self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, nesterov=False if config.momentum == 0. else True,
                               momentum=config.momentum, weight_decay=config.weight_decay)
    self.scheduler = ReduceLROnPlateau(self.optimizer, factor=config.lr_decay, patience=config.patience,
                                       threshold=1e-5, verbose=True)
    self.loss_func = F.cross_entropy
    self.best_loss = np.inf
    self.total_trained_samples = 0
    self.total_trained_steps = 0
    
    # Child runners should implement.
    self.dataset_train = None
    self.dataset_valid = None
    self.dataset_test = None
    
    self.loader_train = None
    self.loader_valid = None
    self.loader_test = None
  
  @abc.abstractmethod
  def train(self):
    ...
  
  @abc.abstractmethod
  def eval(self, loader):
    ...
  
  def validate(self):
    return self.eval(self.loader_valid)
  
  def test(self):
    return self.eval(self.loader_test)
  
  def compute_loss_cnn(self, x, y, mask=None):
    logit = self.model(x)
    
    if mask is None:
      loss = self.loss_func(logit, y)
    else:
      loss = self.loss_func(logit, y, reduction='none')
      # Give feedback to the model only using annotated label.
      # The masked loss is averaged only for annotated label.
      loss = (mask * loss).sum(dim=1) / mask.sum(dim=1)  # average for each sample
      loss = loss.mean()
    
    return loss, logit
  
  def compute_loss_crnn(self, x, y, mask=None):
    logits, _ = self.model(x)
    
    if self.config.loss == 'many2one':
      # Use logit from the last time step.
      if mask is None:
        loss = self.loss_func(logits[-1], y)
      else:
        loss = self.loss_func(logits[-1], y, reduction='none')
        # Give feedback to the model only using annotated label.
        # The masked loss is averaged only for annotated label.
        loss = (mask * loss).sum(dim=1) / mask.sum(dim=1)  # average for each sample
        loss = loss.mean()
    elif self.config.loss == 'many2many':
      weights = torch.ones(len(logits), device=self.device)
      weights /= weights.sum()
      
      loss = 0.
      for logit, weight in zip(logits, weights):
        if mask is None:
          loss += torch.mean(self.loss_func(logit, y, reduction='none') * weight)
        else:
          loss_step = self.loss_func(logit, y, reduction='none')
          # Give feedback to the model only using annotated label.
          # The masked loss is averaged only for annotated label.
          loss_step = (mask * loss_step).sum(dim=1) / mask.sum(dim=1)  # average for each sample
          loss_step = loss_step.mean()
          loss += loss_step * weight
    else:
      raise ValueError(f'Unknown loss: {self.config.loss}')
    
    return loss, logits
  
  def accuracy(self, input, target):
    input = input.max(1)[1].long().cpu()
    target = target.cpu()
    correct = (input == target).sum().item()
    return correct / float(input.shape[0])
  
  # Early stopping function for given validation loss
  def early_stop(self, validation_loss):
    self.scheduler.step(validation_loss)
    if self.lr > self.optimizer.param_groups[0]['lr']:
      self.lr = self.optimizer.param_groups[0]['lr']
      return True
    else:
      return False
  
  def is_best(self, validation_loss):
    if validation_loss < (self.best_loss - self.scheduler.threshold):
      self.best_loss = validation_loss
      return True
    else:
      return False
  
  def save_checkpoint(self, **kwargs):
    torch.save({
      'config': self.config.as_dict(),
      'model_state_dict': self.model.state_dict(),
      'optimizer_state_dict': self.optimizer.state_dict(),
      **kwargs
    }, self.checkpoint_path)
  
  def load_checkpoint(self):
    checkpoint = torch.load(self.checkpoint_path)
    self.model.load_state_dict(checkpoint['model_state_dict'])
    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    self.optimizer.param_groups[0]['lr'] = self.lr
    return checkpoint
  
  def rename_checkpoint(self, score_test):
    new_path = self.checkpoint_path.replace('.pth', f'_{score_test:.4f}.pth')
    os.rename(self.checkpoint_path, new_path)
    return new_path
