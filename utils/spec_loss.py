import numpy as np
import torch
import torch.nn.functional as F

def spec(field,lx=2*np.pi/4,smooth=False):
  n = field.shape[-1]
  nt = field.shape[1]
  result = []
  for i in range(nt):
    uh = torch.rfft(field[:,i,0],1,onesided=False)/n
    vh = torch.rfft(field[:,i,1],1,onesided=False)/n
    wh = torch.rfft(field[:,i,2],1,onesided=False)/n
    uspec = 0.5 * (uh[:,:,:,0]**2+uh[:,:,:,1]**2)
    vspec = 0.5 * (vh[:,:,:,0]**2+vh[:,:,:,1]**2)
    wspec = 0.5 * (wh[:,:,:,0]**2+wh[:,:,:,1]**2)
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
    _, ex = spec(x)
    _, ey = spec(y)
    return F.l1_loss(ex,ey) + F.mse_loss(ex,ey)