# --------------------------------------------------------
# Pose Compositional Tokens
# Written by Zigang Geng (zigang@mail.ustc.edu.cn)
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import (constant_init, normal_init)
from mmpose.models.builder import build_loss
from mmpose.models.heads.topdown_heatmap_base_head import TopdownHeatmapBaseHead
from mmpose.models.builder import HEADS

from .pct_tokenizer_vae import PCT_Tokenizer_VAE
from .modules import MixerLayer, FCBlock, BasicBlock


@HEADS.register_module()
class PCT_Head_VAE(TopdownHeatmapBaseHead):
    def __init__(self,
                 stage_pct,
                 in_channels,
                 image_size,
                 num_joints,
                 aligner_head=None,
                 tokenizer=None,
                 loss_keypoint=None,):
        super().__init__()
        
        self.image_size = image_size
        self.stage_pct = stage_pct
        
        self.guide_ratio = tokenizer['guide_ratio']
        self.img_guide = self.guide_ratio > 0.0
        
        self.conv_channels = aligner_head['conv_channels']
        self.hidden_dim = aligner_head['hidden_dim']

        self.num_blocks = aligner_head['num_blocks']
        self.hidden_inter_dim = aligner_head['hidden_inter_dim']
        self.token_inter_dim = aligner_head['token_inter_dim']
        self.dropout = aligner_head['dropout']
        
        self.latent_dim = tokenizer['latent']['latent_dim']
        
        if stage_pct == "aligner":
            # conv_head + conv_trans - 改通道数
            self.conv_trans = self._make_transition_for_head(
                in_channels, self.conv_channels)
            self.conv_head = self._make_aligner_head(aligner_head)

            input_size = (image_size[0]//32)*(image_size[1]//32)
            self.mixer_trans = FCBlock( # 线性层
                self.conv_channels * input_size, 
                self.latent_dim * self.hidden_dim)

            self.mixer_head = nn.ModuleList(
                [MixerLayer(self.hidden_dim, self.hidden_inter_dim,
                    self.latent_dim, self.token_inter_dim,  
                    self.dropout) for _ in range(self.num_blocks)])
            self.mixer_norm_layer = FCBlock(
                self.hidden_dim, self.hidden_dim)

            self.pred_mu_layer = nn.Linear(
                self.hidden_dim, self.latent_dim)
            self.pred_logvar_layer = nn.Linear(
                self.hidden_dim, self.latent_dim)
        
        self.tokenizer = PCT_Tokenizer_VAE(
            stage_pct=stage_pct, tokenizer=tokenizer, num_joints=num_joints,
            guide_ratio=self.guide_ratio, guide_channels=in_channels)
         
        self.loss = build_loss(loss_keypoint)
         
    def get_loss(self, gt_mu, gt_logvar, pred_mu, pred_logvar):
        losses = dict()
        
        losses['aligner_loss']= self.loss(gt_mu, gt_logvar, pred_mu, pred_logvar)

        losses.append(losses['aligner_loss'])
                
        return losses
    
    def forward(self, x, extra_x, joints=None):
        if not self.img_guide:
            joints_feat = None
        else:
            joints_feat = self.extract_joints_feat(extra_x[-1], joints)
                
        if self.stage_pct == "tokenizer":
            recoverd_joints, mu, log_var, z = self.tokenizer(joints, joints_feat)
            
            return recoverd_joints, mu, log_var, z
        elif self.stage_pct == "aligner":
            batch_size = x[-1].shape[0]
            image_feat = self.conv_head[0](self.conv_trans(x[-1]))

            image_feat = image_feat.flatten(2).transpose(2,1).flatten(1)
            image_feat = self.mixer_trans(image_feat)
            image_feat = image_feat.reshape(batch_size, self.latent_dim, -1)

            for mixer_layer in self.mixer_head:
                image_feat = mixer_layer(image_feat)
            image_feat = self.mixer_norm_layer(image_feat)

            pred_mu = self.pred_mu_layer(image_feat)
            pred_logvar = self.pred_logvar_layer(image_feat)
            
            # recoverd_joints在此阶段为None
            recoverd_joints, gt_mu, gt_logvar, z = self.tokenizer(joints, joints_feat)
            
            return gt_mu, gt_logvar, pred_mu, pred_logvar
        else:
            pass
    
    def _make_transition_for_head(self, inplanes, outplanes):
        transition_layer = [
            nn.Conv2d(inplanes, outplanes, 1, 1, 0, bias=False),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(True)
        ]
        return nn.Sequential(*transition_layer)

    def _make_aligner_head(self, layer_config):
        feature_convs = []
        feature_conv = self._make_layer(
            BasicBlock,
            layer_config['conv_channels'],
            layer_config['conv_channels'],
            layer_config['conv_num_blocks'],
            dilation=layer_config['dilation']
        )
        feature_convs.append(feature_conv)
        
        return nn.ModuleList(feature_convs)

    def _make_layer(
            self, block, inplanes, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=0.1),
            )

        layers = []
        layers.append(block(inplanes, planes, 
                stride, downsample, dilation=dilation))
        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)
    
    def extract_joints_feat(self, feature_map, joint_coords):
        assert self.image_size[1] == self.image_size[0], \
            'If you want to use a rectangle input, ' \
            'please carefully check the length and width below.'
        batch_size, _, _, height = feature_map.shape
        stride = self.image_size[0] / feature_map.shape[-1]
        joint_x = (joint_coords[:,:,0] / stride + 0.5).int()
        joint_y = (joint_coords[:,:,1] / stride + 0.5).int()
        joint_x = joint_x.clamp(0, feature_map.shape[-1] - 1)
        joint_y = joint_y.clamp(0, feature_map.shape[-2] - 1)
        joint_indices = (joint_y * height + joint_x).long()

        flattened_feature_map = feature_map.clone().flatten(2)
        joint_features = flattened_feature_map[
            torch.arange(batch_size).unsqueeze(1), :, joint_indices]

        return joint_features
    
    def init_weights(self):
        if self.stage_pct == "aligner":
            self.tokenizer.eval()
            for name, params in self.tokenizer.named_parameters():
                params.requires_grad = False

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)