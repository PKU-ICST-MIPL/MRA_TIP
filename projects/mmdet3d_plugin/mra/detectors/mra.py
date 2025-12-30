import os
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from mmcv.runner import auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector


@DETECTORS.register_module()
class MRA(MVXTwoStageDetector):
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

        super(MRA,
              self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)
        self.only_occ = occupancy

    def extract_img_feat(self, img, img_metas, len_queue=None):
        """Extract features of images."""

        B = img.size(0)
        if img is not None:
            if img.dim() == 5 and img.size(0) == 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)

            img_feats = self.img_backbone(img)

            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(img_feat.view(int(B/len_queue), len_queue, int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    @auto_fp16(apply_to=('img'))
    def extract_feat(self, img, img_metas=None, len_queue=None):
        """Extract features from images and points."""

        img_feats = self.extract_img_feat(img, img_metas, len_queue=len_queue)
        
        return img_feats

    def forward_pts_train(self,
                          img_feats, 
                          img_metas,
                          target):
        """Forward function'
        """
        csa = self.get_csa(target)
        outs = self.pts_bbox_head(img_feats, img_metas, target)
        losses = self.pts_bbox_head.training_step(outs, target, img_metas, csa)
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

        len_queue = img.size(1)
        
        img_metas = [each[len_queue-1] for each in img_metas]
        img = img[:, -1, ...]
        if self.only_occ:
            img_feats = None
        else:
            img_feats = self.extract_feat(img=img) 
        losses = dict()
        losses_pts = self.forward_pts_train(img_feats, img_metas, target)
        losses.update(losses_pts)
        return losses
    
    def forward_dummy(self,
                      img_metas=None,
                      img=None,
                      target=None):
        len_queue = img.size(1)
        img_metas = [each[len_queue-1] for each in img_metas]
        img = img[:, -1, ...]
        if self.only_occ:
            img_feats = None
        else:
            img_feats = self.extract_feat(img=img)
        outs = self.pts_bbox_head(img_feats, img_metas, target)
        return outs

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
    
    def get_csa(self, target):
        device = target.device
        target_copy = torch.zeros_like(target)
        c0 = [
            [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
        ]
        c1 = [
            [1,2,3,4,5,6,7,8,14,18,19],
            [9,10,11,12,13,15,16,17]
        ]
        c2 = [
            [1,2,3,4,5],
            [6,7,8],
            [9,10,11,12,17],
            [13],
            [14,18,19],
            [15,16]
        ]
        for group in range(len(c2)):
            for label in c2[group]:
                target_copy[target==label] = group+1
        
        target_copy[target==255] = 255
        target_copy = target_copy.unsqueeze(0)

        kernel_up = torch.zeros((1, 1, 3, 3, 3)).to(device)
        kernel_down = torch.zeros((1, 1, 3, 3, 3)).to(device)
        kernel_left = torch.zeros((1, 1, 3, 3, 3)).to(device)
        kernel_right = torch.zeros((1, 1, 3, 3, 3)).to(device)
        kernel_front = torch.zeros((1, 1, 3, 3, 3)).to(device)
        kernel_back = torch.zeros((1, 1, 3, 3, 3)).to(device)
              
        kernel_corners = [torch.zeros((1, 1, 3, 3, 3)).to(device) for _ in range(8)]
        kernel_edges = [torch.zeros((1, 1, 3, 3, 3)).to(device) for _ in range(12)]

        kernel_up[0, 0, 0, 1, 1] = 1
        kernel_down[0, 0, 2, 1, 1] = 1
        kernel_left[0, 0, 1, 0, 1] = 1
        kernel_right[0, 0, 1, 2, 1] = 1
        kernel_front[0, 0, 1, 1, 0] = 1
        kernel_back[0, 0, 1, 1, 2] = 1
        
        corner_idx = 0
        for i in [0, 2]:
            for j in [0, 2]:
                for k in [0, 2]:
                    kernel_corners[corner_idx][0, 0, i, j, k] = 1
                    corner_idx += 1
        edge_coords = [(0,0,1), (0,2,1), (2,0,1), (2,2,1), 
                       (0,1,0), (0,1,2), (2,1,0), (2,1,2),
                       (1,0,0), (1,0,2), (1,2,0), (1,2,2)]
        edge_num = len(edge_coords)
        for i in range(edge_num):
            kernel_edges[i][0, 0, edge_coords[i][0], edge_coords[i][1], edge_coords[i][2]] = 1

        neighbor_up = F.conv3d(target_copy.float(), kernel_up, padding=1)
        neighbor_down = F.conv3d(target_copy.float(), kernel_down, padding=1)
        neighbor_left = F.conv3d(target_copy.float(), kernel_left, padding=1)
        neighbor_right = F.conv3d(target_copy.float(), kernel_right, padding=1)
        neighbor_front = F.conv3d(target_copy.float(), kernel_front, padding=1)
        neighbor_back = F.conv3d(target_copy.float(), kernel_back, padding=1)
        
        neighbor_corners = [F.conv3d(target_copy.float(), kernel_corners[idx], padding=1) for idx in range(corner_idx)]
        neighbor_edges = [F.conv3d(target_copy.float(), kernel_edges[edge_idx], padding=1) for edge_idx in range(edge_num)]

        xor_up = (target_copy != neighbor_up).float()
        xor_down = (target_copy != neighbor_down).float()
        xor_left = (target_copy != neighbor_left).float()
        xor_right = (target_copy != neighbor_right).float()
        xor_front = (target_copy != neighbor_front).float()
        xor_back = (target_copy != neighbor_back).float()
        
        xor_corners = [(target_copy != neighbor_corner).float() for neighbor_corner in neighbor_corners]
        xor_corners_stack = torch.stack(xor_corners, dim=0)
        xor_corner = torch.sum(xor_corners_stack, dim=0)
        
        xor_edges = [(target_copy != neighbor_edge).float() for neighbor_edge in neighbor_edges]
        xor_edges_stack = torch.stack(xor_edges, dim=0)
        xor_edge = torch.sum(xor_edges_stack, dim=0)

        weight_corner = 0.3
        weight_edge = 0.1

        csa = xor_up + xor_down + xor_left + xor_right + xor_front + xor_back + xor_corner * weight_corner + xor_edge * weight_edge

        return csa.squeeze(0)
