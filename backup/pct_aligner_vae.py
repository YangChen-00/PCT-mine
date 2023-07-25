import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
import cv2

from mmpose.models.builder import build_loss
from mmpose.models.builder import HEADS
from timm.models.layers import trunc_normal_

from .modules import MixerLayer

@HEADS.register_module()
class PCT_Aligner_VAE(nn.Module):
    def __init__(self,):
        super().__init__()
        
        