from . import layers
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
NIN = layers.NIN

class CrossAttnBlockpp(nn.Module):
  """Channel-wise self-attention block. Modified from DDPM."""

  def __init__(self, channels, cond_chans, n_heads=4, skip_rescale=False, init_scale=0.):
    super().__init__()
    self.n_heads = n_heads
    self.GroupNorm_0 = nn.GroupNorm(num_groups=min(channels // 4, 32), num_channels=channels,
                                  eps=1e-6)
    self.NIN_0 = NIN(channels+cond_chans, channels*n_heads)
    self.NIN_1 = NIN(channels+cond_chans, channels*n_heads)
    self.NIN_2 = NIN(channels+cond_chans, channels*n_heads)
    self.NIN_3 = NIN(channels*n_heads, channels, init_scale=init_scale)
    self.skip_rescale = skip_rescale

  def split(self,x):
    # splits views along batch dim to two tensors
    B, E, C, H, W = x.shape
    x_split = x.reshape(B//2,2,E,C,H,W)
    x_a = x_split[:,0,:,:,:,:]
    x_b = x_split[:,1,:,:,:,:]

    return x_a,x_b

  def stack_rays(self,q_cond,k_a_cond,k_b_cond):
    # stacks views along batch dim
    b,c,h,w = q_cond.shape
    q_stacked = torch.cat([q_cond[:,None,:,:,:],q_cond[:,None,:,:,:]],1).reshape(b*2,c,h,w)
    b,c,h,w = k_a_cond.shape
    k_stacked = torch.cat([k_a_cond[:,None,:,:,:],k_b_cond[:,None,:,:,:]],1).reshape(b*2,c,h,w)
    return q_stacked,k_stacked

  def forward(self, x, q_cond, k_a_cond, k_b_cond):
    B, C, H, W = x.shape
    q_cond_stacked, k_cond_stacked = self.stack_rays(q_cond,k_a_cond,k_b_cond)
    h = self.GroupNorm_0(x)
    q = self.NIN_0(torch.cat([h,q_cond_stacked],1)).reshape(B,self.n_heads,C,H,W)
    k = self.NIN_1(torch.cat([h,k_cond_stacked],1)).reshape(B,self.n_heads,C,H,W)
    v = self.NIN_2(torch.cat([h,k_cond_stacked],1)).reshape(B,self.n_heads,C,H,W)

    # split into two halves
    q_a,q_b = self.split(q)
    k_a,k_b = self.split(k)
    v_a,v_b = self.split(v)

    # cross for part a
    w_a = torch.einsum('bechw,becij->behwij', q_a, k_b) * (int(C) ** (-0.5))
    w_a = torch.reshape(w_a, (B//2, self.n_heads, H, W, H * W))
    w_a = F.softmax(w_a, dim=-1)
    w_a = torch.reshape(w_a, (B//2, self.n_heads, H, W, H, W))
    h_a = torch.einsum('behwij,becij->bechw', w_a, v_b)

    # cross for part a
    w_b = torch.einsum('bechw,becij->behwij', q_b, k_a) * (int(C) ** (-0.5))
    w_b = torch.reshape(w_b, (B//2, self.n_heads, H, W, H * W))
    w_b = F.softmax(w_b, dim=-1)
    w_b = torch.reshape(w_b, (B//2, self.n_heads, H, W, H, W))
    h_b = torch.einsum('behwij,becij->bechw', w_b, v_a)

    # recombine
    h = torch.cat([h_a[:,None,:,:,:,:],h_b[:,None,:,:,:,:]],1)
    h = h.reshape(B,self.n_heads*C,H,W)
    h = self.NIN_3(h)
    if not self.skip_rescale:
      return x + h
    else:
      return (x + h) / np.sqrt(2.)

class CrossAttnBlockppTwoCams(nn.Module):
  """Channel-wise self-attention block. Modified from DDPM."""

  def __init__(self, channels, cond_chans, n_heads=4, skip_rescale=False, init_scale=0., left_right_pose_cond=False):
    super().__init__()
    self.n_heads = n_heads
    self.GroupNorm_0 = nn.GroupNorm(num_groups=min(channels // 4, 32), num_channels=channels,
                                  eps=1e-6)
    self.NIN_0 = NIN(channels+cond_chans, channels*n_heads)
    self.NIN_1 = NIN(channels+cond_chans, channels*n_heads)
    self.NIN_2 = NIN(channels+cond_chans, channels*n_heads)
    self.left_right_pose_cond = left_right_pose_cond
    if self.left_right_pose_cond:
      # n_heads+2 to account for the extra cross-attention for q_a and k_c,v_c as well as q_c and k_a,v_a
      self.NIN_3 = NIN(channels*(n_heads+2), channels, init_scale=init_scale)
    else:
      self.NIN_3 = NIN(channels*(n_heads), channels, init_scale=init_scale)
    self.skip_rescale = skip_rescale

  def split(self,x):
    # splits views along batch dim to four tensors
    B, E, C, H, W = x.shape
    x_split = x.reshape(B//4,4,E,C,H,W)
    x_a = x_split[:,0,:,:,:,:]
    x_b = x_split[:,1,:,:,:,:]
    x_c = x_split[:,2,:,:,:,:]
    x_d = x_split[:,3,:,:,:,:]

    return x_a, x_b, x_c, x_d

  def stack_rays(self,q_cond,k_a_cond,k_b_cond):
    # stacks views along batch dim
    b,c,h,w = q_cond.shape
    q_stacked = torch.cat([q_cond[:,None,:,:,:],q_cond[:,None,:,:,:]],1).reshape(b*2,c,h,w)
    b,c,h,w = k_a_cond.shape
    k_stacked = torch.cat([k_a_cond[:,None,:,:,:],k_b_cond[:,None,:,:,:]],1).reshape(b*2,c,h,w)
    return q_stacked,k_stacked

  def forward(self, x, q_cond, k_a_cond, k_b_cond):
    B, C, H, W = x.shape
    q_cond_stacked, k_cond_stacked = self.stack_rays(q_cond,k_a_cond,k_b_cond)
    h = self.GroupNorm_0(x)
    q = self.NIN_0(torch.cat([h,q_cond_stacked],1)).reshape(B,self.n_heads,C,H,W)
    k = self.NIN_1(torch.cat([h,k_cond_stacked],1)).reshape(B,self.n_heads,C,H,W)
    v = self.NIN_2(torch.cat([h,k_cond_stacked],1)).reshape(B,self.n_heads,C,H,W)

    # split into four parts
    q_a, q_b, q_c, q_d = self.split(q)
    k_a, k_b, k_c, k_d = self.split(k)
    v_a, v_b, v_c, v_d = self.split(v)

    # cross for part a
    w_a = torch.einsum('bechw,becij->behwij', q_a, k_b) * (int(C) ** (-0.5))
    w_a = torch.reshape(w_a, (B//4, self.n_heads, H, W, H * W))
    w_a = F.softmax(w_a, dim=-1)
    w_a = torch.reshape(w_a, (B//4, self.n_heads, H, W, H, W))
    h_a = torch.einsum('behwij,becij->bechw', w_a, v_b)

    # cross for part b
    w_b = torch.einsum('bechw,becij->behwij', q_b, k_a) * (int(C) ** (-0.5))
    w_b = torch.reshape(w_b, (B//4, self.n_heads, H, W, H * W))
    w_b = F.softmax(w_b, dim=-1)
    w_b = torch.reshape(w_b, (B//4, self.n_heads, H, W, H, W))
    h_b = torch.einsum('behwij,becij->bechw', w_b, v_a)

    # cross for part c
    w_c = torch.einsum('bechw,becij->behwij', q_c, k_d) * (int(C) ** (-0.5))
    w_c = torch.reshape(w_c, (B//4, self.n_heads, H, W, H * W))
    w_c = F.softmax(w_c, dim=-1)
    w_c = torch.reshape(w_c, (B//4, self.n_heads, H, W, H, W))
    h_c = torch.einsum('behwij,becij->bechw', w_c, v_d)

    # cross for part d
    w_d = torch.einsum('bechw,becij->behwij', q_d, k_c) * (int(C) ** (-0.5))
    w_d = torch.reshape(w_d, (B//4, self.n_heads, H, W, H * W))
    w_d = F.softmax(w_d, dim=-1)
    w_d = torch.reshape(w_d, (B//4, self.n_heads, H, W, H, W))
    h_d = torch.einsum('behwij,becij->bechw', w_d, v_c)

    if self.left_right_pose_cond:
      # cross for part ac
      w_ac = torch.einsum('bechw,becij->behwij', q_a, k_c) * (int(C) ** (-0.5))
      w_ac = torch.reshape(w_ac, (B//4, self.n_heads, H, W, H * W))
      w_ac = F.softmax(w_ac, dim=-1)
      w_ac = torch.reshape(w_ac, (B//4, self.n_heads, H, W, H, W))
      h_ac = torch.einsum('behwij,becij->bechw', w_ac, v_c)

      # cross for part ca
      w_ca = torch.einsum('bechw,becij->behwij', q_c, k_a) * (int(C) ** (-0.5))
      w_ca = torch.reshape(w_ca, (B//4, self.n_heads, H, W, H * W))
      w_ca = F.softmax(w_ca, dim=-1)
      w_ca = torch.reshape(w_ca, (B//4, self.n_heads, H, W, H, W))
      h_ca = torch.einsum('behwij,becij->bechw', w_ca, v_a)

      # recombine
      h = torch.cat([h_a[:,None,:,:,:,:], h_b[:,None,:,:,:,:], h_c[:,None,:,:,:,:], h_d[:,None,:,:,:,:], h_ac[:,None,:,:,:,:], h_ca[:,None,:,:,:,:]],1)
      h = h.reshape(B,(self.n_heads+2)*C,H,W)
    else:
      h = torch.cat([h_a[:,None,:,:,:,:], h_b[:,None,:,:,:,:], h_c[:,None,:,:,:,:], h_d[:,None,:,:,:,:]],1)
      h = h.reshape(B,(self.n_heads)*C,H,W)

    h = self.NIN_3(h)
    if not self.skip_rescale:
      return x + h
    else:
      return (x + h) / np.sqrt(2.)
