import numpy as np
import torch
import torch.nn.functional as F
from torch.fft import fftn
from torch import conj

def spec(field,lx=2*np.pi/4,smooth=False):
  n = field.shape[-1]
  nt = field.shape[1]
  result = []
  for i in range(nt):
    '''
    uh = torch.rfft(field[:,i,0],1,onesided=False)/n
    vh = torch.rfft(field[:,i,1],1,onesided=False)/n
    wh = torch.rfft(field[:,i,2],1,onesided=False)/n
    '''
    uh = fftn(field[:,i,0])/n
    vh = fftn(field[:,i,1])/n
    wh = fftn(field[:,i,2])/n
    uspec = 0.5 * (uh*conj(uh)).real
    vspec = 0.5 * (vh*conj(vh)).real
    wspec = 0.5 * (wh*conj(wh)).real
    uspec = uspec.reshape(uspec.shape[0],1,n,n)
    vspec = vspec.reshape(vspec.shape[0],1,n,n)
    wspec = wspec.reshape(wspec.shape[0],1,n,n)
    k = 2.0 * np.pi / lx
    wave_numbers = k*np.arange(0,n)
    spec = torch.cat((uspec,vspec,wspec),dim=1)
    spec[:,:,:,int(n/2+1):,] = 0

    if smooth == True:
      window = torch.ones(3,3,5,5).type_as(spec)/ 5
      specsmooth = F.conv2d(spec,window,padding=2)
      #specsmooth[:,:,:,0:4] = spec[:,:,:,0:4]
      spec = specsmooth
    result.append(spec.unsqueeze(0))

  result = torch.cat(result,dim=0)
  result = torch.mean(result,dim=0)
  return wave_numbers, spec

def spec_loss(x,y):
    _, ex = spec(x.float())
    _, ey = spec(y.float())
    return F.l1_loss(ex,ey) + F.mse_loss(ex,ey)