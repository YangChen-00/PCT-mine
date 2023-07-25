import torch

from torch import nn
from timm.models.layers import trunc_normal_

class PCT_Tokenizer(nn.Module):
    def __init__(self,
                 tokenizer=None,
                 num_joints=17,
                 guide_ratio=0,
                 guide_channels=0):
        super().__init__()

        self.guide_ratio = guide_ratio
        self.num_joints = num_joints
        
        
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
        
        self.latent_dim = tokenizer['latent']['latent_dim']
        
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

        self.fc_mu = nn.Linear(self.enc_hidden_dim, self.latent_dim)
        self.fc_var = nn.Linear(self.enc_hidden_dim, self.latent_dim)
        
        self.decoder_start = nn.Linear(
            self.latent_dim, self.dec_hidden_dim)

        self.decoder = nn.ModuleList(
            [MixerLayer(self.dec_hidden_dim, self.dec_hidden_inter_dim,
                self.num_joints, self.dec_token_inter_dim, 
                self.dec_dropout) for _ in range(self.dec_num_blocks)])
        self.decoder_layer_norm = nn.LayerNorm(self.dec_hidden_dim)

        self.recover_embed = nn.Linear(self.dec_hidden_dim, 2)

        
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
    
    def forward(self, joints, joints_feature):
        """
        Args:
            input: tensor[BS, K, D]
        """
        # Encoder of Tokenizer, Get the PCT groundtruth class labels.
        # [bs, k, 2], [bs, k, 1], bs
        joints_coord, joints_visible, bs \
            = joints[:,:,:-1], joints[:,:,-1].bool(), joints.shape[0]
            
        encode_feat = self.start_embed(joints_coord) 
        # encode_feat - [bs, k, int(self.enc_hidden_dim*(1-self.guide_ratio))]
        
        if self.guide_ratio > 0: # 实际上可以设置不参考图片特征，仅用关节点坐标提取的特征
            encode_img_feat = self.start_img_embed(joints_feature) 
            # encode_img_feat - [bs, k, int(self.enc_hidden_dim*self.guide_ratio)]
            
            encode_feat = torch.cat((encode_feat, encode_img_feat), dim=2) 
            # encode_feat - [bs, k, self.enc_hidden_dim]
            
        # 随机丢弃关节点
        rand_mask_ind = torch.rand(
            joints_visible.shape, device=joints.device) > self.drop_rate
        joints_visible = torch.logical_and(rand_mask_ind, joints_visible) 
        
        mask_tokens = self.invisible_token.expand(bs, joints.shape[1], -1) 
        # 从[1, 1, enc_hidden_dim]扩张到[bs, k, enc_hidden_dim]
        w = joints_visible.unsqueeze(-1).type_as(mask_tokens) 
        # joints_visible从[bs, k]扩张到[bs, k, enc_hidden_dim]
        encode_feat = encode_feat * w + mask_tokens * (1 - w) 
        # 将那些丢弃的与不可见的关节点的特征用 从正态分布中采样的值来表示
        
        for num_layer in self.encoder:
            encode_feat = num_layer(encode_feat)
        encode_feat = self.encoder_layer_norm(encode_feat)
        # encode_feat - [bs, k, enc_hidden_dim]
        
        mu = self.fc_mu(encode_feat) # mu - [bs, k, latent_dim]
        log_var = self.fc_var(encode_feat) # log_var - [bs, k, latent_dim]
        z = self.reparameterize(mu, log_var) # z - [bs, k, latent_dim]
        
        
        decode_feat = self.decoder_start(z)
        # decode_feat - [bs, k, dec_hidden_dim]
        
        for num_layer in self.decoder:
            decode_feat = num_layer(decode_feat)
        decode_feat = self.decoder_layer_norm(decode_feat) # [bs, k, dec_hidden_dim]
        
        recoverd_joints = self.recover_embed(decode_feat) # [bs, k, 2]
    
        return recoverd_joints, mu, log_var, z
        
class MLPBlock(nn.Module):
    def __init__(self, dim, inter_dim, dropout_ratio):
        super().__init__()

        self.ff = nn.Sequential(
            nn.Linear(dim, inter_dim),
            nn.GELU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(inter_dim, dim),
            nn.Dropout(dropout_ratio)
        )

    def forward(self, x):
        return self.ff(x)
    
class MixerLayer(nn.Module):
    def __init__(self, 
                 hidden_dim, 
                 hidden_inter_dim, 
                 token_dim, 
                 token_inter_dim, 
                 dropout_ratio):
        super().__init__()
        
        self.layernorm1 = nn.LayerNorm(hidden_dim)
        self.MLP_token = MLPBlock(token_dim, token_inter_dim, dropout_ratio)
        self.layernorm2 = nn.LayerNorm(hidden_dim)
        self.MLP_channel = MLPBlock(hidden_dim, hidden_inter_dim, dropout_ratio)

    def forward(self, x):
        y = self.layernorm1(x)
        y = y.transpose(2, 1)
        y = self.MLP_token(y)
        y = y.transpose(2, 1)
        z = self.layernorm2(x + y)
        z = self.MLP_channel(z)
        out = x + y + z
        return out