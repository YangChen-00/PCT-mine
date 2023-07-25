# --------------------------------------------------------
# Pose Compositional Tokens
# Written by Zigang Geng (zigang@mail.ustc.edu.cn)
# --------------------------------------------------------

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
class PCT_Tokenizer(nn.Module):
    """ Tokenizer of Pose Compositional Tokens.
        paper ref: Zigang Geng et al. "Human Pose as
            Compositional Tokens"

    Args:
        stage_pct (str): Training stage (Tokenizer or Classifier).
        tokenizer (list): Config about the tokenizer.
        num_joints (int): Number of annotated joints in the dataset.
        guide_ratio (float): The ratio of image guidance.
        guide_channels (int): Feature Dim of the image guidance.
    """

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
        self.enc_token_inter_dim = tokenizer['encoder']['token_inter_dim']
        self.enc_hidden_inter_dim = tokenizer['encoder']['hidden_inter_dim']
        self.enc_dropout = tokenizer['encoder']['dropout']

        self.dec_num_blocks = tokenizer['decoder']['num_blocks']
        self.dec_hidden_dim = tokenizer['decoder']['hidden_dim']
        self.dec_token_inter_dim = tokenizer['decoder']['token_inter_dim']
        self.dec_hidden_inter_dim = tokenizer['decoder']['hidden_inter_dim']
        self.dec_dropout = tokenizer['decoder']['dropout']

        self.token_num = tokenizer['codebook']['token_num']
        self.token_class_num = tokenizer['codebook']['token_class_num']
        self.token_dim = tokenizer['codebook']['token_dim']
        self.decay = tokenizer['codebook']['ema_decay']

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
        
        self.token_mlp = nn.Linear(
            self.num_joints, self.token_num)
        self.feature_embed = nn.Linear(
            self.enc_hidden_dim, self.token_dim)

        self.register_buffer('codebook', 
            torch.empty(self.token_class_num, self.token_dim))
        self.codebook.data.normal_() # 标准正态分布初始化codebook, shape=[token_class_num, token_dim]
        self.register_buffer('ema_cluster_size', 
            torch.zeros(self.token_class_num)) # shape=[token_class_num]
        self.register_buffer('ema_w', 
            torch.empty(self.token_class_num, self.token_dim))
        self.ema_w.data.normal_() # shape=[token_class_num, token_dim]
        
        self.decoder_token_mlp = nn.Linear(
            self.token_num, self.num_joints)
        self.decoder_start = nn.Linear(
            self.token_dim, self.dec_hidden_dim)

        self.decoder = nn.ModuleList(
            [MixerLayer(self.dec_hidden_dim, self.dec_hidden_inter_dim,
                self.num_joints, self.dec_token_inter_dim, 
                self.dec_dropout) for _ in range(self.dec_num_blocks)])
        self.decoder_layer_norm = nn.LayerNorm(self.dec_hidden_dim)

        self.recover_embed = nn.Linear(self.dec_hidden_dim, 2)

        self.loss = build_loss(tokenizer['loss_keypoint'])

    def forward(self, joints, joints_feature, cls_logits, train=True):
        """Forward function. """

        if train or self.stage_pct == "tokenizer":
            # Encoder of Tokenizer, Get the PCT groundtruth class labels.
            # [bs, k, 2], [bs, k, 1], bs
            joints_coord, joints_visible, bs \
                = joints[:,:,:-1], joints[:,:,-1].bool(), joints.shape[0]

            encode_feat = self.start_embed(joints_coord)
            if self.guide_ratio > 0: # 实际上可以设置不参考图片特征，仅用关节点坐标提取的特征
                encode_img_feat = self.start_img_embed(joints_feature)
                encode_feat = torch.cat((encode_feat, encode_img_feat), dim=2)

            if train and self.stage_pct == "tokenizer":
                # 随机丢弃关节点
                rand_mask_ind = torch.rand(
                    joints_visible.shape, device=joints.device) > self.drop_rate
                joints_visible = torch.logical_and(rand_mask_ind, joints_visible) 

            mask_tokens = self.invisible_token.expand(bs, joints.shape[1], -1) # 从[1, 1, enc_hidden_dim]扩张到[bs, k, enc_hidden_dim]
            w = joints_visible.unsqueeze(-1).type_as(mask_tokens) # joints_visible从[bs, k]扩张到[bs, k, enc_hidden_dim]
            encode_feat = encode_feat * w + mask_tokens * (1 - w) # 将那些丢弃的与不可见的关节点的特征用 从正态分布中采样的值来表示
                    
            for num_layer in self.encoder:
                encode_feat = num_layer(encode_feat)
            encode_feat = self.encoder_layer_norm(encode_feat)
            
            encode_feat = encode_feat.transpose(2, 1) # 从[bs, k, enc_hidden_dim]改为[bs, enc_hidden_dim, k]
            encode_feat = self.token_mlp(encode_feat).transpose(2, 1) # 从[bs, enc_hidden_dim, token_num]变为[bs, token_num, enc_hidden_dim]
            # flatten(x, start_dim, end_dimension)函数执行的功能是将从start_dim到end_dim之间的所有维度值乘起来，其他的维度保持不变。
            encode_feat = self.feature_embed(encode_feat).flatten(0,1) # 从[bs, token_num, token_dim]变为[bs*token_num, token_dim]
            
            # 欧式距离的方，distances - [bs*token_num, token_class_num]
            distances = torch.sum(encode_feat**2, dim=1, keepdim=True) \
                + torch.sum(self.codebook**2, dim=1) \
                - 2 * torch.matmul(encode_feat, self.codebook.t())
            
            # torch.save(self.codebook,'codebook.pt')
            # self.get_codebook_feature()
                
            # 每个样本的token_nums对应的token_dim都在码本中找到最相似的token_class_num对应的index
            encoding_indices = torch.argmin(distances, dim=1) # [bs*token_num,]
            encodings = torch.zeros( # [bs*token_num, token_class_num]
                encoding_indices.shape[0], self.token_class_num, device=joints.device)
            # scatter(dim, index, src)将src中数据根据index中的索引按照dim的方向进行填充。
            encodings.scatter_(1, encoding_indices.unsqueeze(1), 1) # 转为one-hot
        else: #todo 待看
            bs = cls_logits.shape[0] // self.token_num
            encoding_indices = None
        
        if self.stage_pct == "classifier": #todo
            # cls_logits - [bs*token_num, token_class_dim]
            # self.codebook - [token_class_dim, token_dim]
            # part_token_feat - [bs*token_num, token_dim]
            part_token_feat = torch.matmul(cls_logits, self.codebook) # 取出各token向量在分类结果加权后的结果
        else: # tokenizer
            # part_token_feat - [bs*token_num, token_dim]
            part_token_feat = torch.matmul(encodings, self.codebook) # 取出最相似的码本中的特征向量

        if train and self.stage_pct == "tokenizer":
            # Updating Codebook using EMA
            # dw - [token_class_num, token_dim] 得出最相似的结果index上encoder出来的token特征
            dw = torch.matmul(encodings.t(), encode_feat.detach())
            # sync 下面三行完全为了做GPU的数据同步（传输数据到其他GPU）
            n_encodings, n_dw = encodings.numel(), dw.numel()
            encodings_shape, dw_shape = encodings.shape, dw.shape
            combined = torch.cat((encodings.flatten(), dw.flatten()))
            # dist.all_reduce(combined) # math sum
            # 下面三行就是重新组织成tensor
            sync_encodings, sync_dw = torch.split(combined, [n_encodings, n_dw])
            sync_encodings, sync_dw = \
                sync_encodings.view(encodings_shape), sync_dw.view(dw_shape)

            # torch.sum(sync_encodings, 0) - 统计每个token出现的次数
            self.ema_cluster_size = self.ema_cluster_size * self.decay + \
                                    (1 - self.decay) * torch.sum(sync_encodings, 0)
            
            n = torch.sum(self.ema_cluster_size.data)
            #? 这一步又是做什么？
            self.ema_cluster_size = ( 
                (self.ema_cluster_size + 1e-5)
                / (n + self.token_class_num * 1e-5) * n)
            
            # ema_w - [token_class_num, token_dim]
            self.ema_w = self.ema_w * self.decay + (1 - self.decay) * sync_dw
            #? 为什么要用ema_w去除以ema_cluster_size
            self.codebook = self.ema_w / self.ema_cluster_size.unsqueeze(1)
            e_latent_loss = F.mse_loss(part_token_feat.detach(), encode_feat) # 计算离最相似的token的相似距离作为e_latent_loss
            
            part_token_feat = encode_feat + (part_token_feat - encode_feat).detach() # [bs*token_num, token_dim]
        else: # Classifier
            e_latent_loss = None
        
        # Decoder of Tokenizer, Recover the joints.
        part_token_feat = part_token_feat.view(bs, -1, self.token_dim) # [bs, token_num, token_dim]
        
        part_token_feat = part_token_feat.transpose(2,1) # [bs, token_dim, token_num]
        #? decoder_token_mlp 将 token_num -> k 在提取什么东西？
        part_token_feat = self.decoder_token_mlp(part_token_feat).transpose(2,1) # [bs, token_dim, token_num]-> [bs, token_dim, k] -> [bs, k, token_dim]
        decode_feat = self.decoder_start(part_token_feat) # [bs, k, dec_hidden_dim]

        for num_layer in self.decoder:
            decode_feat = num_layer(decode_feat)
        decode_feat = self.decoder_layer_norm(decode_feat) # [bs, k, dec_hidden_dim]

        #? recover_embed 将 dec_hidden_dim -> 2 又在提取什么东西？
        recoverd_joints = self.recover_embed(decode_feat) # [bs, k, 2]

        # classifier中encoding_indices用来干什么？——用来对分类结果做CELoss
        return recoverd_joints, encoding_indices, e_latent_loss

    def get_loss(self, output_joints, joints, e_latent_loss):
        """Calculate loss for training tokenizer.

        Note:
            batch_size: N
            num_keypoints: K

        Args:
            output_joints (torch.Tensor[NxKx3]): Recovered joints.
            joints(torch.Tensor[NxKx3]): Target joints.
            e_latent_loss(torch.Tensor[1]): Loss for training codebook.
        """
        
        losses = dict()

        kpt_loss, e_latent_loss = self.loss(output_joints, joints, e_latent_loss)
        
        losses['joint_loss'] = kpt_loss
        losses['e_latent_loss'] = e_latent_loss

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
            assert (self.stage_pct == "classifier"), \
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
            if self.stage_pct == "classifier":
                print('If you are training a classifier, '\
                    'must check that the well-trained tokenizer '\
                    'is located in the correct path.')

    # def get_codebook_feature(self):
    #     codebook = self.codebook.cpu()
    #     for i in range(codebook.shape[0]):
    #         feature = codebook[i, :] # 在channel维度上，每个channel代表了一个卷积核的输出特征图，所以对每个channel的图像分别进行处理和保存
    #         feature = feature.view(32, 16) # batch为1，所以可以直接view成二维张量
    #         feature = feature.data.numpy() # 转为numpy

    #         # 根据图像的像素值中最大最小值，将特征图的像素值归一化到了[0,1];
    #         feature = (feature - np.amin(feature))/(np.amax(feature) - np.amin(feature) + 1e-5) # 注意要防止分母为0！ 
    #         feature = np.round(feature * 255) # [0, 1]——[0, 255],为cv2.imwrite()函数而进行

    #         cv2.imwrite('./feat/codebook/' + str(i) + '.jpg', feature)  # 保存当前层输出的每个channel上的特征图为一张图像