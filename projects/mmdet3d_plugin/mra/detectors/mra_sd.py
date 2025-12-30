import os
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from mmcv.runner import auto_fp16
from mmdet.models import DETECTORS

from .mra import MRA
from mmcv.runner import load_checkpoint


class SelfDistillTeacherOccWrapper(nn.Module):
    def __init__(self, teacher):
        super().__init__()
        self.teacher = teacher
        for name, para in self.teacher.named_parameters():
            para.requires_grad_(False)

    @torch.no_grad()
    def load_pre_checkpoint(self, checkpoint_path):
        load_checkpoint(self.teacher, checkpoint_path, map_location='cpu', strict=False)

class SelfDistillTeacherOcc(MRA):
    @torch.no_grad()
    def momentum_update(self, cur_model, teacher_momentum=None, step=None):
        """
        Momentum update of the key encoder
        """
        if teacher_momentum is None:
            teacher_momentum = min(1 - 1 / (step + 1), self.teacher_momentum)
        for param_q, param_k in zip(cur_model.parameters(), self.parameters()):
            param_k.data = param_k.data * teacher_momentum + param_q.data * (1. - teacher_momentum)

@DETECTORS.register_module()
class MRA_SD(MRA):
    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 occupancy=False,
                 ):

        super(MRA_SD,
              self).__init__(use_grid_mask, pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)
        
        self.tsw = SelfDistillTeacherOccWrapper(
            SelfDistillTeacherOcc(use_grid_mask, pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained))
        self.tsw.teacher.eval()
        self.tsw.teacher.teacher_momentum = 0.99
        self.turn_on_teacher = True
        self.local_step = 0
    
    def forward_pts_train(self,
                          img_feats,
                          teacher_img_feats,
                          img_metas,
                          target):
        """Forward function'
        """
        csa = self.get_csa(target)
        outs = self.pts_bbox_head(img_feats, img_metas, target)
        with torch.no_grad():
            teacher_outs = self.tsw.teacher.pts_bbox_head(teacher_img_feats, img_metas, target)
        losses = self.pts_bbox_head.training_step(outs, target, img_metas, csa, teacher_outs)
        return losses
    
    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)
    
    @auto_fp16(apply_to=('img', 'points'))
    def forward_train(self,
                      img_metas=None,
                      img=None,
                      target=None):
        """Forward training function.
        Args:
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            img (torch.Tensor): Images of each sample with shape
                (batch, C, H, W). Defaults to None.
            target (torch.Tensor): ground-truth of semantic scene completion
                (batch, X_grids, Y_grids, Z_grids)
        Returns:
            dict: Losses of different branches.
        """

        if self.turn_on_teacher:
            if self.local_step == 0:
                conv_dict = {k: v for k, v in self.state_dict().items() if 'tsw' not in k}
                self.tsw.teacher.load_state_dict(conv_dict)
            self.tsw.teacher.momentum_update(self, step=self.local_step)
            self.local_step += 1

        len_queue = img.size(1)
        
        img_metas = [each[len_queue-1] for each in img_metas]
        img = img[:, -1, ...]
        if self.only_occ:
            img_feats = None
        else:
            img_feats = self.extract_feat(img=img) 
        with torch.no_grad():
            if self.turn_on_teacher:
                teacher_img_feats = self.tsw.teacher.extract_feat(img=img)
        losses = dict()
        losses_pts = self.forward_pts_train(img_feats, teacher_img_feats, img_metas, target)
        losses.update(losses_pts)
        return losses
    
    def forward_test(self,
                     img_metas=None,
                     img=None,
                     target=None,
                      **kwargs):
        """Forward testing function.
        Args:
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            img (torch.Tensor): Images of each sample with shape
                (batch, C, H, W). Defaults to None.
            target (torch.Tensor): ground-truth of semantic scene completion
                (batch, X_grids, Y_grids, Z_grids)
        Returns:
            dict: Completion result.
        """

        len_queue = img.size(1)
        
        img_metas = [each[len_queue-1] for each in img_metas]
        img = img[:, -1, ...]
        if self.only_occ:
            img_feats = None
        else:
            img_feats = self.extract_feat(img=img)  
        outs = self.pts_bbox_head(img_feats, img_metas, target)
        completion_results = self.pts_bbox_head.validation_step(outs, target, img_metas)

        return completion_results