import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import (constant_init, normal_init)
from mmpose.models.builder import build_loss
from mmpose.models.heads.topdown_heatmap_base_head import TopdownHeatmapBaseHead
from mmpose.models.builder import HEADS

from .pct_vae_tokenizer import PCT_VAE_Tokenizer
from .modules import MixerLayer, FCBlock, BasicBlock
from evaluate.eval import pose_pck_accuracy

@HEADS.register_module()
class PCT_VAE_Head(TopdownHeatmapBaseHead):
    def __init__(self,
                 stage_pct,
                 in_channels,
                 image_size,
                 num_joints,
                 pred_head=None,
                 tokenizer=None,
                 loss_keypoint=None,):
        super().__init__()

        self.image_size = image_size
        self.stage_pct = stage_pct
        
        self.guide_ratio = tokenizer['guide_ratio']
        self.img_guide = self.guide_ratio > 0.0
        
        self.conv_channels = pred_head['conv_channels']
        self.hidden_dim = pred_head['hidden_dim']

        self.num_blocks = pred_head['num_blocks']
        self.hidden_inter_dim = pred_head['hidden_inter_dim']
        self.latent_inter_dim = pred_head['latent_inter_dim']
        self.dropout = pred_head['dropout']
        
        self.latent_num = tokenizer['latent_num']
        
        if stage_pct == "predictor":
            self.conv_trans = self._make_transition_for_head(
                in_channels, self.conv_channels)
            self.conv_head = self._make_pred_head(pred_head)

            input_size = (image_size[0]//32)*(image_size[1]//32)
            self.mixer_trans = FCBlock(
                self.conv_channels * input_size, 
                self.latent_num * self.hidden_dim)

            self.mixer_head = nn.ModuleList(
                [MixerLayer(self.hidden_dim, self.hidden_inter_dim,
                    self.latent_num, self.latent_inter_dim,  
                    self.dropout) for _ in range(self.num_blocks)])
            self.mixer_norm_layer = FCBlock(
                self.hidden_dim, self.hidden_dim)

            self.pred_mu_layer = nn.Linear(
                self.hidden_dim, 1)
            self.pred_var_layer = nn.Linear(
                self.hidden_dim, 1)
        
        self.tokenizer = PCT_VAE_Tokenizer(
            stage_pct=stage_pct, tokenizer=tokenizer, num_joints=num_joints,
            guide_ratio=self.guide_ratio, guide_channels=in_channels,)
        
        self.loss = build_loss(loss_keypoint)
        
    def get_loss(self, output_joints, joints, mu, log_var):
        """Calculate loss for training predictor. Stage2

        Note:
            batch_size: N
            num_keypoints: K
            num_token: M
            num_token_class: V

        Args:
            p_joints(torch.Tensor[NxKx3]): Predicted joints 
                recovered from the predicted class.
            gt_joints(torch.Tensor[NxKx3]): Groundtruth joints.
        """
        
        losses = dict()

        recons_loss, kld_loss = self.loss(output_joints, joints, mu, log_var)
        
        losses['recons_loss'] = recons_loss
        losses['kld_loss'] = kld_loss

        unused_losses = []
        for name, loss in losses.items():
            if loss == None:
                unused_losses.append(name)
        for unused_loss in unused_losses:
            losses.pop(unused_loss)
                
        return losses
    
    def get_accuracy(self, output, target):
        """Calculate accuracy for keypoint loss.
        """

        target_weight = target[:, :, -1]
        target = target[:, :, :-1]
        
        # print(target_weight)
        
        accuracy = dict()
        
        _, avg_acc, _ = pose_pck_accuracy(
            output.detach().cpu().numpy(),
            target.detach().cpu().numpy(),
            # [N,K]维度的true or false，当关节点存在权值时意味着该关节点在GT中visible
            target_weight.detach().cpu().numpy() > 0) 
        accuracy['acc_pose'] = float(avg_acc)
        
        return accuracy
    
    def forward(self, x, extra_x, joints=None, train=True):
        """Forward function."""
        
        if self.stage_pct == "predictor":
            batch_size = x[-1].shape[0]
            pred_feat = self.conv_head[0](self.conv_trans(x[-1]))
            # [bs, c, h, w]
            
            pred_feat = pred_feat.flatten(2).transpose(2,1).flatten(1)
            # [bs, c, h, w] -> [bs, h*w, c] -> [bs, c*h*w] 
            pred_feat = self.mixer_trans(pred_feat)
            # [bs, c*h*w] -> [bs, after_dim]
            pred_feat = pred_feat.reshape(batch_size, self.latent_num, -1)
            # [bs, latent_num, after_dim]

            for mixer_layer in self.mixer_head:
                pred_feat = mixer_layer(pred_feat)
            pred_feat = self.mixer_norm_layer(pred_feat) # # [bs, latent_num, hidden_dim]

            pred_mu = self.pred_mu_layer(pred_feat).squeeze(-1) # [bs, latent_num]
            pred_var = self.pred_var_layer(pred_feat).squeeze(-1) # [bs, latent_num]
        else:
            pred_mu = None
            pred_var = None   

        if not self.img_guide or \
            (self.stage_pct == "predictor" and not train):
            joints_feat = None
        else:
            joints_feat = self.extract_joints_feat(extra_x[-1], joints)

        output_joints, mu, log_var, dist_z = \
            self.tokenizer(joints, joints_feat, pred_mu, pred_var, train=train)

        if train:
            return output_joints, mu, log_var, dist_z
        else:
            return output_joints, mu, log_var, dist_z
    
    def _make_transition_for_head(self, inplanes, outplanes):
        transition_layer = [
            nn.Conv2d(inplanes, outplanes, 1, 1, 0, bias=False),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(True)
        ]
        return nn.Sequential(*transition_layer)

    def _make_pred_head(self, layer_config):
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
        if self.stage_pct == "predictor":
            self.tokenizer.eval()
            for name, params in self.tokenizer.named_parameters():
                params.requires_grad = False

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
                
    