import torch
import torch.nn.functional as F

def autocorr(frames):
  corr = []
  n = frames.shape[0]
  nt = frames.shape[1]
  for i in range(n):
    weight = torch.cat((frames[i].permute(1,0,2,3),frames[i].permute(1,0,2,3),frames[i].permute(1,0,2,3)),dim=0).reshape(3,3,nt,128,128)
    corr.append(F.conv3d(frames[i].permute(1,0,2,3).unsqueeze(0),weight))
  corr = torch.cat(corr,dim=0)
  return corr

def corr_loss(x,y):
  Rx = autocorr(x)
  Ry = autocorr(y)
  return F.l1_loss(Rx,Ry) + F.mse_loss(Rx,Ry)