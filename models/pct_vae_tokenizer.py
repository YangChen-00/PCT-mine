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
class PCT_VAE_Tokenizer(nn.Module):
    def __init__(self,
                 stage_pct,
                 tokenizer=None,
                 num_joints=17,
                 guide_ratio=0,
                 guide_channels=0):
        super().__init__()
        
        self.stage_pct = stage_pct
        self.guide_ratio = guide_ratio
        self.num_joints = num_joints
        
        self.drop_rate = tokenizer['encoder']['drop_rate']     
        self.enc_num_blocks = tokenizer['encoder']['num_blocks']
        self.enc_hidden_dim = tokenizer['encoder']['hidden_dim']
        self.enc_hidden_inter_dim = tokenizer['encoder']['hidden_inter_dim']
        self.enc_token_inter_dim = tokenizer['encoder']['token_inter_dim']
        self.enc_dropout = tokenizer['encoder']['dropout']
        
        self.dec_num_blocks = tokenizer['decoder']['num_blocks']
        self.dec_hidden_dim = tokenizer['decoder']['hidden_dim']
        self.dec_token_inter_dim = tokenizer['decoder']['token_inter_dim']
        self.dec_hidden_inter_dim = tokenizer['decoder']['hidden_inter_dim']
        self.dec_dropout = tokenizer['decoder']['dropout']
        
        self.latent_num = tokenizer['latent_num']
        
        self.invisible_token = nn.Parameter(
            torch.zeros(1, 1, self.enc_hidden_dim))
        # 用正态分布替换变量中的值（[-0.02,0.02]的值）
        trunc_normal_(self.invisible_token, mean=0., std=0.02, a=-0.02, b=0.02)
        
        # start_img_embed和start_embed得到的最终维度加和为定值，为self.enc_hidden_dim
        if self.guide_ratio > 0:
            self.start_img_embed = nn.Linear(
                guide_channels, int(self.enc_hidden_dim*self.guide_ratio))
        self.start_embed = nn.Linear(
            2, int(self.enc_hidden_dim*(1-self.guide_ratio)))
        
        self.encoder = nn.ModuleList(
            [MixerLayer(self.enc_hidden_dim, self.enc_hidden_inter_dim, 
                self.num_joints, self.enc_token_inter_dim,
                self.enc_dropout) for _ in range(self.enc_num_blocks)])
        self.encoder_layer_norm = nn.LayerNorm(self.enc_hidden_dim)
        
        self.latent_mlp = nn.Linear(
            self.num_joints, self.latent_num)
        self.feature_embed = nn.Linear(
            self.enc_hidden_dim, self.latent_num)
        
        self.fc_mu = nn.Linear(self.enc_hidden_dim, 1)
        self.fc_var = nn.Linear(self.enc_hidden_dim, 1)
        
        self.decoder_start = nn.Linear(
            self.latent_num, self.num_joints * self.dec_hidden_dim)
        
        self.decoder = nn.ModuleList(
            [MixerLayer(self.dec_hidden_dim, self.dec_hidden_inter_dim,
                self.num_joints, self.dec_token_inter_dim, 
                self.dec_dropout) for _ in range(self.dec_num_blocks)])
        self.decoder_layer_norm = nn.LayerNorm(self.dec_hidden_dim)

        self.recover_embed = nn.Linear(self.dec_hidden_dim, 2)
        
        self.loss = build_loss(tokenizer['loss_keypoint'])
        
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
        
    def forward(self, joints, joints_feature, pred_mu, pred_var, train=True):
        """
        Args:
            joints (Tensor): Input joints.
            joints_feature (Tensor): Input joints feature.
            train (bool): Training stage or not.
        """
        
        # Encoder of Tokenizer, Get the PCT groundtruth class labels.
        # [bs, k, 2], [bs, k, 1], bs
        if train:
            joints_coord, joints_visible, bs \
                = joints[:,:,:-1], joints[:,:,-1].bool(), joints.shape[0]
        else:
            bs = pred_mu.shape[0]
            
        if train or self.stage_pct == "tokenizer":    
            encode_feat = self.start_embed(joints_coord)
            if self.guide_ratio > 0: # 实际上可以设置不参考图片特征，仅用关节点坐标提取的特征
                encode_img_feat = self.start_img_embed(joints_feature)
                encode_feat = torch.cat((encode_feat, encode_img_feat), dim=2)
                # [bs, k, enc_hidden_dim]
                
            # 随机丢弃关节点
            rand_mask_ind = torch.rand(
                joints_visible.shape, device=joints.device) > self.drop_rate
            joints_visible = torch.logical_and(rand_mask_ind, joints_visible)
            
            mask_tokens = self.invisible_token.expand(bs, joints.shape[1], -1) # 从[1, 1, enc_hidden_dim]扩张到[bs, k, enc_hidden_dim]
            w = joints_visible.unsqueeze(-1).type_as(mask_tokens) # joints_visible从[bs, k]扩张到[bs, k, enc_hidden_dim]
            encode_feat = encode_feat * w + mask_tokens * (1 - w) # 将那些丢弃的与不可见的关节点的特征用 从正态分布中采样的值来表示（MIM做法）
            
            for num_layer in self.encoder:
                encode_feat = num_layer(encode_feat)
            encode_feat = self.encoder_layer_norm(encode_feat)
            
            encode_feat = encode_feat.transpose(2, 1) # 从[bs, k, enc_hidden_dim]改为[bs, enc_hidden_dim, k]
            encode_feat = self.latent_mlp(encode_feat).transpose(2, 1) # 从[bs, enc_hidden_dim, latent_num]变为[bs, latent_num, enc_hidden_dim]
            mu = self.fc_mu(encode_feat).squeeze(-1) # [bs, latent_num]
            log_var = self.fc_var(encode_feat).squeeze(-1) # [bs, latent_num]
            dist_z = self.reparameterize(mu, log_var) # [bs, latent_num]
        
        if self.stage_pct == "predictor":
            mu = pred_mu
            log_var = pred_var
            dist_z = self.reparameterize(mu, log_var) # [bs, latent_num]
        
        # Decoder of Tokenizer, Recover the joints.
        decode_feat = self.decoder_start(dist_z).reshape(bs, -1, self.dec_hidden_dim) # [bs, latent_num] -> [bs, k*dec_hidden_dim] -> [bs, k, dec_hidden_dim]

        for num_layer in self.decoder:
            decode_feat = num_layer(decode_feat)
        decode_feat = self.decoder_layer_norm(decode_feat) # [bs, k, dec_hidden_dim]

        recoverd_joints = self.recover_embed(decode_feat) # [bs, k, 2]

        return recoverd_joints, mu, log_var, dist_z
    
    def get_loss(self, output_joints, joints, mu, log_var):
        """Calculate loss for training tokenizer. Stage1

        Note:
            batch_size: N
            num_keypoints: K

        Args:
            output_joints (torch.Tensor[NxKx3]): Recovered joints.
            joints(torch.Tensor[NxKx3]): Target joints.
            e_latent_loss(torch.Tensor[1]): Loss for training codebook.
        """

        losses = dict()

        recons_loss, kld_loss = self.loss(output_joints, joints, mu, log_var)
        
        losses['recons_loss'] = recons_loss
        losses['kld_loss'] = kld_loss

        return losses
    
    def init_weights(self, pretrained=""):
        """Initialize model weights."""

        parameters_names = set()
        for name, _ in self.named_parameters():
            parameters_names.add(name)

        buffers_names = set()
        for name, _ in self.named_buffers():
            buffers_names.add(name)

        if os.path.isfile(pretrained):
            assert (self.stage_pct == "predictor"), \
                "Training tokenizer does not need to load model"
            pretrained_state_dict = torch.load(pretrained, 
                            map_location=lambda storage, loc: storage)

            need_init_state_dict = {}

            for name, m in pretrained_state_dict['state_dict'].items():
                if 'keypoint_head.tokenizer.' in name:
                    name = name.replace('keypoint_head.tokenizer.', '')
                if name in parameters_names or name in buffers_names:
                    need_init_state_dict[name] = m
            self.load_state_dict(need_init_state_dict, strict=True)
        else:
            if self.stage_pct == "predictor":
                print('If you are training a predictor, '\
                    'must check that the well-trained tokenizer '\
                    'is located in the correct path.')
                
    