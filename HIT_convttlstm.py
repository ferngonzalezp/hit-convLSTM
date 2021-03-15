from pytorch_lightning import LightningModule
from utils.convlstmnet import ConvLSTMNet
from utils.spec_loss import spec_loss
from utils.corr_loss import corr_loss
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.normalize import normalize
import HIT_dataset
import os, argparse
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from utils.animation import fluid_anim
import matplotlib.pyplot as plt

def magnitude(x):
  return (torch.sum(x**2,dim=0))**0.5

class convttlstm(LightningModule):
  def __init__(self,params):
    super().__init__()
    self.hparams = params
    self.save_hyperparameters(params)
    if self.hparams.model == 'convttlstm':
      self.model = ConvLSTMNet(
        input_channels = 3, 
        output_sigmoid = False,
        # model architecture
        layers_per_block = (3, 3, 3, 3), 
        hidden_channels  = (32, 48, 48, 32), 
        skip_stride = 2,
        # convolutional tensor-train layers
        cell = 'convttlstm',
        cell_params = {
            "order": 3, 
            "steps": 3, 
            "ranks": 8},
        # convolutional parameters
        kernel_size = 5)
    else:
        self.model = ConvLSTMNet(
        input_channels = 3, 
        output_sigmoid = False,
        # model architecture
        layers_per_block = (3, 3, 3, 3), 
        hidden_channels  = (32, 48, 48, 32), 
        skip_stride = 2,
        # convolutional tensor-train layers
        cell = 'convlstm',
        cell_params = {},
        # convolutional parameters
        kernel_size = 5)
  
  @staticmethod
  def add_model_specific_args(parent_parser):
      parser = ArgumentParser(parents=[parent_parser], add_help=False)
      parser.add_argument('--input_frames', type=int, default=5)
      parser.add_argument('--future_frames', type=int, default=5)
      parser.add_argument('--output_frames', type=int, default=10)
      parser.add_argument('--batch_size', type=int, default=8)
      parser.add_argument('--lr', type=float, default=1e-4)
      parser.add_argument('--use-checkpointing', dest = 'use_checkpointing', 
        action = 'store_true',  help = 'Use checkpointing to reduce memory utilization.')
      parser.add_argument( '--no-checkpointing', dest = 'use_checkpointing', 
          action = 'store_false', help = 'No checkpointing (faster training).')
      parser.set_defaults(use_checkpointing = False)
      parser.add_argument('--model', default = 'convttlstm', type = str,
        help = 'The model is either \"convlstm\", \"convttlstm\".')
      return parser

  def forward(self,x,input_frames,future_frames,output_frames,teacher_forcing=False):
      pred = self.model(x, 
                input_frames  =  input_frames, 
                future_frames = future_frames, 
                output_frames = output_frames, 
                teacher_forcing = teacher_forcing)
      return pred

  def loss(self,output,target):
    return F.l1_loss(output,target) + F.mse_loss(output,target) + 1e3*spec_loss(output,target)
  
  def training_step(self,batch,batch_idx):

    frames = batch
    frames = normalize(frames)
    inputs = frames[:, :-1]
    origin = frames[:, -self.hparams.output_frames:]

    pred = self(inputs,self.hparams.input_frames,self.hparams.future_frames,self.hparams.output_frames,teacher_forcing=True)

    loss = self.loss(pred, origin)
    #tensorboard_logs = {'loss': loss.detach()}
    self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    #return {'loss': loss, 'log': tensorboard_logs}
    return loss

  def validation_step(self, batch, batch_idx):

    frames = batch
    frames = normalize(frames)
    inputs = frames[:,  :self.hparams.input_frames]
    origin = frames[:, -self.hparams.output_frames:]

    pred = self(inputs,self.hparams.input_frames,self.hparams.future_frames,self.hparams.output_frames)

    loss = self.loss(pred, origin)
    self.log('val_loss', loss, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
    return {'val_loss': loss}
  
  def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('val_loss', avg_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        #avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        #tensorboard_logs = {'val_loss': avg_loss}
        #return {'val_loss': avg_loss, 'log': tensorboard_logs}
  
  def test_step(self, batch, batch_idx):

    frames = batch
    frames = normalize(frames)
    inputs = frames[:,  :self.hparams.input_frames]
    origin = frames[:, -self.hparams.output_frames:]

    pred = self(inputs,self.hparams.input_frames,self.hparams.future_frames,self.hparams.output_frames)

    loss = self.loss(pred, origin)
    spec_loss = spec_loss(pred,origin)
    self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
    self.log('spec_loss', spec_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    

  def configure_optimizers(self):
      opt = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
      return opt