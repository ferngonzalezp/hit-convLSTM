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
from utils.spec_loss import spec

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
      parser.add_argument('--ckpt_path', default = './checkpt.ckpt', type = str)
      return parser

  def forward(self,x,input_frames,future_frames,output_frames,teacher_forcing=False):
      pred = self.model(x, 
                input_frames  =  input_frames, 
                future_frames = future_frames, 
                output_frames = output_frames, 
                teacher_forcing = teacher_forcing)
      return pred

  def loss(self,output,target):
    return F.l1_loss(output,target) + F.mse_loss(output,target) + 1*spec_loss(output,target)
  
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
    s_loss = spec_loss(pred,origin)
    self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
    self.log('spec_loss', s_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
    return {'origin':frames, 'pred':pred}

  def test_epoch_end(self, outputs):
    origin = outputs[0]['origin']
    pred = outputs[0]['pred']
    fluid_anim(origin[0],'source')
    fluid_anim(pred[0],'prediction')
    k, E = spec(pred,smooth=True)
    k, E_o = spec(origin,smooth=True)
    fig = plt.figure()
    plt.xscale('log')
    plt.yscale('log')
    plt.plot(k,torch.mean(E[:,0,0,:],dim=0).cpu(),label='predicted')
    plt.plot(k,torch.mean(E_o[:,0,0,:],dim=0).cpu(),label='original')
    plt.legend()
    plt.savefig('avg_spectrum.png')
    fig = plt.figure()
    plt.subplot(2,3,1)
    plt.imshow(magnitude(origin[0,0]).float().cpu())
    plt.title('t = 0')
    plt.subplot(2,3,2)
    plt.imshow(magnitude(origin[0,origin.shape[1]//2]).float().cpu())
    plt.title('t = '+str(origin.shape[1]//2))
    plt.subplot(2,3,3)
    plt.imshow(magnitude(origin[0,-1]).float().cpu())
    plt.title('t = '+str(origin.shape[1]-1))
    plt.subplot(2,3,4)
    plt.imshow(magnitude(pred[0,0]).float().cpu())
    plt.title('t = 0')
    plt.subplot(2,3,5)
    plt.imshow(magnitude(pred[0,pred.shape[1]//2]).float().cpu())
    plt.title('t = '+str(origin.shape[1]//2))
    plt.subplot(2,3,6)
    plt.imshow(magnitude(pred[0,-1]).float().cpu())
    plt.title('t = '+str(origin.shape[1]-1))
    plt.savefig('velocity.png')


  def configure_optimizers(self):
      opt = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
      return opt