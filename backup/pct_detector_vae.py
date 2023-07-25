# --------------------------------------------------------
# Pose Compositional Tokens
# Based on MMPose (https://github.com/open-mmlab/mmpose)
# Written by Zigang Geng (zigang@mail.ustc.edu.cn)
# --------------------------------------------------------

import time
import torch
import numpy as np

import mmcv
from mmcv.runner import auto_fp16
from mmpose.models import builder
from mmpose.models.builder import POSENETS
from mmpose.models.detectors.base import BasePose
from mmpose.core.post_processing import transform_preds


@POSENETS.register_module()
class PCT_VAE(BasePose):
    def __init__(self,
                 backbone,
                 keypoint_head=None,
                 test_cfg=None,
                 pretrained=None):
        super().__init__()
        self.stage_pct = keypoint_head['stage_pct']
        assert self.stage_pct in ["tokenizer", "aligner", "predictor"]
        self.image_guide = keypoint_head['tokenizer']['guide_ratio'] > 0
        
        if self.stage_pct == "tokenizer":
            # For training tokenizer
            keypoint_head['loss_keypoint'] \
                = keypoint_head['tokenizer']['loss_keypoint']
                
        if self.stage_pct == "aligner":
            # For training classifier
            # backbone is only needed for training aligner
            self.backbone = builder.build_backbone(backbone)
        
        if self.image_guide:
            # extra_backbone is optional feature to guide the training tokenizer
            # It brings a slight impact on performance
            self.extra_backbone = builder.build_backbone(backbone)
        
        self.keypoint_head = builder.build_head(keypoint_head)
        
        self.init_weights(pretrained, keypoint_head['tokenizer']['ckpt'])
        
        self.flip_test = test_cfg.get('flip_test', True)
        self.dataset_name = test_cfg.get('dataset_name', 'COCO')
    
    def init_weights(self, pretrained, tokenizer):
        """Weight initialization for model."""
        if self.stage_pct == "aligner":
            self.backbone.init_weights(pretrained)
        if self.image_guide:
            self.extra_backbone.init_weights(pretrained)
        self.keypoint_head.init_weights()
        self.keypoint_head.tokenizer.init_weights(tokenizer)
        
    @auto_fp16(apply_to=('img', ))
    def forward(self,
                img,
                joints_3d=None,
                joints_3d_visible=None,
                img_metas=None,
                return_loss=True,
                **kwargs):
        if self.stage_pct == "tokenizer" or self.stage_pct == "aligner":
            joints = joints_3d
            joints[...,-1] = joints_3d_visible[...,0]
        else:
            # Just a placeholder during inference of PCT
            joints = None
            
        if return_loss:
            return self.forward_train(img, joints, img_metas, **kwargs)
        return self.forward_test(
            img, joints, img_metas, **kwargs)
        
    def forward_train(self, img, joints, img_metas, **kwargs):
        output = self.backbone(img) if self.stage_pct == "predictor" or self.stage_pct == "aligner" else None
        extra_output = self.extra_backbone(img) if self.image_guide else None
        
        losses = dict()
        if self.stage_pct == "tokenizer":
            recoverd_joints, mu, log_var, z = self.keypoint_head(output, extra_output, joints)
            tokenizer_losses = \
                self.keypoint_head.tokenizer.get_loss(recoverd_joints, joints, mu, log_var)
            losses.update(tokenizer_losses)
        elif self.stage_pct == "aligner":
            gt_mu, gt_logvar, pred_mu, pred_logvar = self.keypoint_head(output, extra_output, joints)
            aligner_losses = self.keypoint_head.get_loss(gt_mu, gt_logvar, pred_mu, pred_logvar)
            losses.update(aligner_losses)
        else:
            pass
        
        return losses
    
    def forward_test(self):
        return None
    
    def show_result(self):
        # Not implemented
        return None