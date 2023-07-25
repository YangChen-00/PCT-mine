# --------------------------------------------------------
# Pose Compositional Tokens
# Written by Zigang Geng (zigang@mail.ustc.edu.cn)
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmpose.models.builder import LOSSES


@LOSSES.register_module()
class JointS1Loss(nn.Module):
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def smooth_l1_loss(self, pred, gt):
        l1_loss = torch.abs(pred - gt)
        cond = l1_loss < self.beta
        # torch.where(condition, a, b) - 输入参数condition：条件限制，如果满足条件，则选择a，否则选择b作为输出。
        loss = torch.where(cond, 0.5*l1_loss**2/self.beta, l1_loss-0.5*self.beta)
        return loss

    def forward(self, pred, gt):
        
        joint_dim = gt.shape[2] - 1
        visible = gt[..., joint_dim:]
        pred, gt = pred[..., :joint_dim], gt[..., :joint_dim]
 
        loss = self.smooth_l1_loss(pred, gt) * visible
        loss = loss.mean(dim=2).mean(dim=1).mean(dim=0) #? 为什么这么写？

        return loss


@LOSSES.register_module()
class Tokenizer_loss(nn.Module):
    def __init__(self, joint_loss_w, e_loss_w, beta=0.05):
        """
            joint_loss_w - 关节点重建Loss的加权
            e_loss_w - 最相似token的加权
        """
        super().__init__()

        self.joint_loss = JointS1Loss(beta)
        self.joint_loss_w = joint_loss_w

        self.e_loss_w = e_loss_w

    def forward(self, output_joints, joints, e_latent_loss):

        losses = []
        joint_loss = self.joint_loss(output_joints, joints)
        joint_loss *= self.joint_loss_w
        losses.append(joint_loss)

        e_latent_loss *= self.e_loss_w
        losses.append(e_latent_loss)

        return losses
    
@LOSSES.register_module()
class HeatmapWeightedL1Loss(nn.Module):
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def smooth_l1_loss(self, pred, gt, visible):
        l1_loss = torch.abs(pred - gt) # [batch_size, num_joints, ht_h, ht_w]
        cond = l1_loss < self.beta
        # torch.where(condition, a, b) - 输入参数condition：条件限制，如果满足条件，则选择a，否则选择b作为输出。
        loss = torch.where(cond, 0.5*l1_loss**2/self.beta, l1_loss-0.5*self.beta) * visible
        
        pos_weight = torch.sum(gt == 0) / (gt.shape[0] * gt.shape[1] * gt.shape[2] * gt.shape[3])
        neg_loss = (loss[gt == 0] * (1 - pos_weight)).mean()
        pos_loss = (loss[gt != 0] * pos_weight).mean()
        
        return pos_loss, neg_loss

    def forward(self, pred, gt, visible): 
        # pred - [batch_size, num_joints, ht_h, ht_w]
        # gt - [batch_size, num_joints, ht_h, ht_w]
        # visible - [batch_size, num_joints, 1]
        visible = visible.unsqueeze(-1)
        pos_loss, neg_loss = self.smooth_l1_loss(pred, gt, visible)

        return pos_loss + neg_loss

@LOSSES.register_module()
class Tokenizer_Heatmap_loss(nn.Module):
    def __init__(self, heatmap_loss_w, e_loss_w, beta=0.05):
        """
            joint_loss_w - 关节点重建Loss的加权
            e_loss_w - 最相似token的加权
        """
        super().__init__()

        self.heatmap_loss = HeatmapWeightedL1Loss(beta)
        self.heatmap_loss_w = heatmap_loss_w

        self.e_loss_w = e_loss_w

    def forward(self, output_heatmap, target, target_weight, heatmap_e_latent_loss):

        losses = []
        heatmap_loss = self.heatmap_loss(output_heatmap, target, target_weight)
        heatmap_loss *= self.heatmap_loss_w
        losses.append(heatmap_loss)

        heatmap_e_latent_loss *= self.e_loss_w
        losses.append(heatmap_e_latent_loss)

        return losses
    

@LOSSES.register_module()
class Joint_VAE_loss(nn.Module):
    def __init__(self, recons_loss_w, kld_loss_w):
        super().__init__()
        self.recons_loss_w = recons_loss_w
        self.kld_loss_w = kld_loss_w

    def forward(self, recoverd_joints, joints, mu, log_var):
        joint_dim = joints.shape[2] - 1
        gt_joints = joints[..., :joint_dim]
        visible = joints[..., joint_dim:]
        
        losses = []
        
        recons_loss = (recoverd_joints - gt_joints) ** 2 * visible
        recons_loss = recons_loss.mean(dim=2).mean(dim=1).mean(dim=0)
        recons_loss *= self.recons_loss_w
        losses.append(recons_loss)

        kld_loss = 0.0
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
                        
        kld_loss *= self.kld_loss_w
        
        losses.append(kld_loss)

        return losses

@LOSSES.register_module()
class Classifer_loss(nn.Module):
    def __init__(self, token_loss=1.0, joint_loss=1.0, beta=0.05):
        super().__init__()

        self.token_loss = nn.CrossEntropyLoss()
        self.token_loss_w = token_loss

        self.joint_loss = JointS1Loss(beta=beta)
        self.joint_loss_w = joint_loss

    def forward(self, p_logits, p_joints, g_logits, joints):

        losses = []
        if self.token_loss_w > 0:
            token_loss = self.token_loss(p_logits, g_logits)
            token_loss *= self.token_loss_w
            losses.append(token_loss)
        else:
            losses.append(None)
        
        if self.joint_loss_w > 0:
            joint_loss = self.joint_loss(p_joints, joints)
            joint_loss *= self.joint_loss_w
            losses.append(joint_loss)
        else:
            losses.append(None)
            
        return losses

@LOSSES.register_module()
class Aligner_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, gt_mu, gt_logvar, pred_mu, pred_logvar):
        """
            gt_mu - Keypoints Encoder得出的mu
            gt_lovar - Kerypoints Encoder得出的logvar
            pred_mu - Image Encoder得出的mu
            pred_lovar - Image Encoder得出的logvar
        """
        losses = []
        
        gt_std = torch.exp(0.5 * gt_logvar)
        pred_std = torch.exp(0.5 * pred_logvar)
        
        # +1e-5避免分母为0
        aligner_loss = torch.mean(torch.log(pred_std / (gt_logvar + 1e-5)) 
                          - 0.5 
                          + (pred_std ** 2 - (pred_mu - gt_mu) ** 2) / (2 * (gt_std ** 2) + 1e-5)
                          )
        
        losses.append(aligner_loss)
        
        return losses