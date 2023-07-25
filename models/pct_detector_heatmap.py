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
from evaluate.eval import keypoints_from_heatmaps
from evaluate.vis_heatmap import visualize_heatmap_single_sample

@POSENETS.register_module()
class PCT_Heatmap(BasePose):
    """ Detector of Pose Compositional Tokens.
        paper ref: Zigang Geng et al. "Human Pose as
            Compositional Tokens"

    Args:
        backbone (dict): Backbone modules to extract feature.
        keypoint_head (dict): Keypoint head to process feature.
        test_cfg (dict): Config for testing. Default: None.
        pretrained (str): Path to the pretrained models.
    """

    def __init__(self,
                 backbone,
                 keypoint_head=None,
                 test_cfg=None,
                 pretrained=None):
        super().__init__()
        self.stage_pct = keypoint_head['stage_pct']
        assert self.stage_pct in ["tokenizer", "classifier"]

        if self.stage_pct == "tokenizer":
            # For training tokenizer
            keypoint_head['loss_keypoint'] \
                = keypoint_head['tokenizer']['loss_keypoint']

        if self.stage_pct == "classifier":
            # For training classifier
            # backbone is only needed for training classifier
            self.backbone = builder.build_backbone(backbone)

        self.keypoint_head = builder.build_head(keypoint_head)

        self.init_weights(pretrained, keypoint_head['tokenizer']['ckpt'])

        self.test_cfg = test_cfg
        self.flip_test = test_cfg.get('flip_test', True)
        self.dataset_name = test_cfg.get('dataset_name', 'COCO')

    def init_weights(self, pretrained, tokenizer):
        """Weight initialization for model."""
        if self.stage_pct == "classifier":
            self.backbone.init_weights(pretrained)
        self.keypoint_head.init_weights()
        self.keypoint_head.tokenizer.init_weights(tokenizer)

    @auto_fp16(apply_to=('img', ))
    def forward(self,
                img,
                target=None,
                target_weight=None,
                img_metas=None,
                return_loss=True,
                **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True. Note this setting will change the expected inputs.
        When `return_loss=True`, img and img_meta are single-nested (i.e.
        Tensor and List[dict]), and when `resturn_loss=False`, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.

        Note:
            batch_size: N
            num_keypoints: K
            num_img_channel: C (Default: 3)
            img height: imgH
            img width: imgW

        Args:
            img (torch.Tensor[NxCximgHximgW]): Input images.
            joints_3d (torch.Tensor[NxKx3]): Target joints.
            joints_3d_visible (torch.Tensor[NxKx3]): Visibility of each target joint.
                Only first NxKx1 is valid.
            img_metas (list(dict)): Information about data augmentation
                By default this includes:
                - "image_file: path to the image file
                - "center": center of the bbox
                - "scale": scale of the bbox
                - "rotation": rotation of the bbox
                - "bbox_score": score of bbox
            return_loss (bool): Option to `return loss`. `return loss=True`
                for training, `return loss=False` for validation & test.

        Returns:
            dict|tuple: if `return loss` is true, then return losses.
              Otherwise, return predicted poses, boxes, image paths.
        """
        if not (return_loss or self.stage_pct == "tokenizer"):
            # Just a placeholder during inference of PCT
            target = None
            target_weight = None

        if return_loss:
            return self.forward_train(img, target, target_weight, img_metas, **kwargs)
        return self.forward_test(
            img, target, target_weight, img_metas, **kwargs)

    def forward_train(self, img, target, target_weight, img_metas, **kwargs):
        """Defines the computation performed at every call when training."""

        output = None if self.stage_pct == "tokenizer" else self.backbone(img)

        """
            p_logits - 分类网络输出的logits（only Stage2）
            p_joints - Decoder得出的热力图（Stage1 and Stage2）
            g_logits - Encoder得出的最相似token的index（Stage1 and Stage2）
            e_latent_loss - 计算离最相似的token的相似距离作为e_latent_loss（only Stage1）
        """
        p_logits, p_joints, g_logits, e_latent_loss = \
            self.keypoint_head(output, target, target_weight)

        # if return loss
        losses = dict()
        if self.stage_pct == "classifier":
            keypoint_losses = self.keypoint_head.get_loss(
                p_logits, p_joints, g_logits, target, target_weight)
            losses.update(keypoint_losses)

            topk = (1,2,5)
            keypoint_accuracy = \
                self.get_class_accuracy(p_logits, g_logits, topk)
            kpt_accs = {}
            for i in range(len(topk)):
                kpt_accs['top%s-acc' % str(topk[i])] \
                    = keypoint_accuracy[i]
            losses.update(kpt_accs)
            
            keypoint_accuracy = self.keypoint_head.get_accuracy(
                p_joints, target, target_weight)
            losses.update(keypoint_accuracy)
        elif self.stage_pct == "tokenizer":
            visualize_heatmap_single_sample(img, p_joints, target)
            
            keypoint_losses = \
                self.keypoint_head.tokenizer.get_loss(
                    p_joints, target, target_weight, e_latent_loss)
            losses.update(keypoint_losses)
        
        return losses

    def get_class_accuracy(self, output, target, topk):
        
        maxk = max(topk)
        batch_size = target.size(0)
        # torch.topk(input, k, dim=None, largest=True, sorted=True, out=None) 
        # - 其中k是保留的k个值，dim在指定的维度进行取最大最小
        # - largest=True意味着选取最大的，sorted=True是指将返回结果排序
        # - topk返回的是一个tuple，第一个元素指返回的具体值，第二个元素指返回值的index
        _, pred = output.topk(maxk, 1, True, True) # 取出最大的前5值
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred)) # tensor.eq逐元素判断是否相同，返回boolean
        return [
            correct[:k].reshape(-1).float().sum(0) \
                * 100. / batch_size for k in topk]
        
    def forward_test(self, img, target, target_weight, img_metas, **kwargs):
        """Defines the computation performed at every call when testing."""
        assert img.size(0) == len(img_metas)

        results = {}
    
        batch_size, _, img_height, img_width = img.shape
        if batch_size > 1:
            assert 'bbox_id' in img_metas[0]
            
        output = None if self.stage_pct == "tokenizer" \
            else self.backbone(img) 
        
        p_joints, encoding_scores = \
            self.keypoint_head(output, target, target_weight, train=False)
            
        test_c = np.zeros((batch_size, 2), dtype=np.float32)
        test_s = np.zeros((batch_size, 2), dtype=np.float32)
        test_preds, test_maxvals = keypoints_from_heatmaps(
            p_joints.cpu().numpy(),
            test_c,
            test_s,
            unbiased=self.test_cfg.get('unbiased_decoding', False),
            post_process=self.test_cfg.get('post_process', 'default'),
            kernel=self.test_cfg.get('modulate_kernel', 11),
            valid_radius_factor=self.test_cfg.get('valid_radius_factor',
                                                  0.0546875),
            use_udp=self.test_cfg.get('use_udp', False),
            target_type=self.test_cfg.get('target_type', 'GaussianHeatmap'))

        if self.flip_test:
            FLIP_INDEX = {'COCO': [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15], \
                    'CROWDPOSE': [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 12, 13], \
                    'OCCLUSIONPERSON':[0, 4, 5, 6, 1, 2, 3, 7, 8, 12, 13, 14, 9, 10, 11],\
                    'MPII': [5, 4, 3, 2, 1, 0, 6, 7, 8, 9, 15, 14, 13, 12, 11, 10]}

            img_flipped = img.flip(3)
    
            features_flipped = None if self.stage_pct == "tokenizer" \
                else self.backbone(img_flipped) 

            if target is not None:
                joints_flipped = target.clone()
                joints_flipped = joints_flipped[:,FLIP_INDEX[self.dataset_name],:]
                joints_flipped[:,:,0] = img.shape[-1] - 1 - joints_flipped[:,:,0]
            else:
                joints_flipped = None
                
            p_joints_f, encoding_scores_f = \
                self.keypoint_head(features_flipped, joints_flipped, target_weight, train=False)

            p_joints_f = p_joints_f[:,FLIP_INDEX[self.dataset_name],:]
            p_joints_f[:,:,0] = img.shape[-1] - 1 - p_joints_f[:,:,0]

            p_joints = (p_joints + p_joints_f)/2.0

        batch_size = len(img_metas)

        if 'bbox_id' in img_metas[0]:
            bbox_ids = []
        else:
            bbox_ids = None

        c = np.zeros((batch_size, 2), dtype=np.float32)
        s = np.zeros((batch_size, 2), dtype=np.float32)
        image_paths = []
        score = np.ones(batch_size)
        for i in range(batch_size):
            c[i, :] = img_metas[i]['center']
            s[i, :] = img_metas[i]['scale']
            image_paths.append(img_metas[i]['image_file'])

            if 'bbox_score' in img_metas[i]:
                score[i] = np.array(img_metas[i]['bbox_score']).reshape(-1)
            if bbox_ids is not None:
                bbox_ids.append(img_metas[i]['bbox_id'])

        p_joints = p_joints.cpu().numpy()
        
        preds, maxvals = keypoints_from_heatmaps(
            p_joints,
            c,
            s,
            unbiased=self.test_cfg.get('unbiased_decoding', False),
            post_process=self.test_cfg.get('post_process', 'default'),
            kernel=self.test_cfg.get('modulate_kernel', 11),
            valid_radius_factor=self.test_cfg.get('valid_radius_factor',
                                                  0.0546875),
            use_udp=self.test_cfg.get('use_udp', False),
            target_type=self.test_cfg.get('target_type', 'GaussianHeatmap'))
        
        all_preds = np.zeros((batch_size, p_joints.shape[1], 3), dtype=np.float32)
        all_boxes = np.zeros((batch_size, 6), dtype=np.float32)
        all_preds[:, :, 0:2] = preds[:, :, 0:2]
        all_preds[:, :, 2:3] = maxvals
        all_boxes[:, 0:2] = c[:, 0:2]
        all_boxes[:, 2:4] = s[:, 0:2]
        all_boxes[:, 4] = np.prod(s * 200.0, axis=1)
        all_boxes[:, 5] = score

        final_preds = {}
        final_preds['preds'] = all_preds
        final_preds['boxes'] = all_boxes
        final_preds['image_paths'] = image_paths
        final_preds['bbox_ids'] = bbox_ids
        results.update(final_preds)
        results['output_heatmap'] = None

        return results

    def show_result(self):
        # Not implemented
        return None