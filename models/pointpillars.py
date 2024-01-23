import torch
from torch import nn
import numpy as np
import cv2 as cv
from typing import List
from utils.kitti import AnnotationKitti, iou2d_nearest, limit_period
from utils.geometry import nms_rotate_cpu
from torch.nn import functional as f


class Loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, beta=1 / 9, cls_w=1.0, reg_w=2.0, dir_w=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.cls_w = cls_w
        self.reg_w = reg_w
        self.dir_w = dir_w
        self.smooth_l1_loss = nn.SmoothL1Loss(reduction='none',
                                              beta=beta)
        self.dir_cls = nn.CrossEntropyLoss()

    def forward(self,
                bbox_cls_pred,
                bbox_pred,
                bbox_dir_cls_pred,
                batched_labels,
                num_cls_pos,
                batched_bbox_reg,
                batched_dir_labels
                ):
        nclasses = bbox_cls_pred.size(1)
        batched_labels = f.one_hot(batched_labels, nclasses + 1)[:, :nclasses].float()
        bbox_cls_pred_sigmoid = bbox_cls_pred.sigmoid()
        weights = self.alpha * (1 - bbox_cls_pred_sigmoid).pow(self.gamma) * batched_labels + \
                  (1 - self.alpha) * bbox_cls_pred_sigmoid.pow(self.gamma) * (1 - batched_labels)
        cls_loss = f.binary_cross_entropy(bbox_cls_pred_sigmoid, batched_labels, reduction='none')
        cls_loss = cls_loss * weights
        cls_loss = cls_loss.sum() / num_cls_pos

        reg_loss = self.smooth_l1_loss(bbox_pred, batched_bbox_reg)
        reg_loss = reg_loss.sum() / reg_loss.size(0)

        dir_cls_loss = self.dir_cls(bbox_dir_cls_pred, batched_dir_labels)
        total_loss = self.cls_w * cls_loss + self.reg_w * reg_loss + self.dir_w * dir_cls_loss
        loss_dict = {'cls_loss': cls_loss,
                     'reg_loss': reg_loss,
                     'dir_cls_loss': dir_cls_loss,
                     'total_loss': total_loss}
        return loss_dict


class PillarEncoder(nn.Module):
    def __init__(self, voxel_size, voxel_range, in_channel, out_channel):
        super().__init__()
        self.out_channel = out_channel
        self.vx, self.vy = voxel_size[0], voxel_size[1]
        self.x_offset = voxel_size[0] / 2 + voxel_range[0]
        self.y_offset = voxel_size[1] / 2 + voxel_range[1]
        self.x_l = int((voxel_range[3] - voxel_range[0]) / voxel_size[0])
        self.y_l = int((voxel_range[4] - voxel_range[1]) / voxel_size[1])
        self.conv = nn.Conv1d(in_channel, out_channel, 1, bias=False)
        self.bn = nn.BatchNorm1d(out_channel, eps=1e-3, momentum=0.01)
        self.relu = nn.ReLU()

    def forward(self, pillars, coors_batch, np_per_pillar):
        device = pillars.device
        offset_pt_center = pillars[:, :, :3] - torch.sum(pillars[:, :, :3], dim=1, keepdim=True) / np_per_pillar[:, None, None]
        x_offset_pi_center = pillars[:, :, :1] - (coors_batch[:, None, 1:2] * self.vx + self.x_offset)
        y_offset_pi_center = pillars[:, :, 1:2] - (coors_batch[:, None, 2:3] * self.vy + self.y_offset)
        features = torch.cat([pillars, offset_pt_center, x_offset_pi_center, y_offset_pi_center], dim=-1)
        # features[:, :, 0:1] = x_offset_pi_center  # tmp
        # features[:, :, 1:2] = y_offset_pi_center  # tmp
        voxel_ids = torch.arange(0, pillars.size(1)).to(device)
        mask = voxel_ids[:, None] < np_per_pillar[None, :]
        mask = mask.permute(1, 0).contiguous()
        features *= mask[:, :, None]
        features = features.permute(0, 2, 1).contiguous()
        pooling_features = self.relu(self.bn(self.conv(features))).amax(dim=-1)

        batched_canvas = []
        bs = coors_batch[-1, 0] + 1
        for i in range(bs):
            cur_coors_idx = coors_batch[:, 0] == i
            cur_coors = coors_batch[cur_coors_idx, :]
            cur_features = pooling_features[cur_coors_idx]
            canvas = torch.zeros((self.x_l, self.y_l, self.out_channel), dtype=torch.float32, device=device)
            canvas[cur_coors[:, 1], cur_coors[:, 2]] = cur_features
            canvas = canvas.permute(2, 1, 0).contiguous()
            batched_canvas.append(canvas)
        batched_canvas = torch.stack(batched_canvas, dim=0)
        return batched_canvas


class Backbone(nn.Module):
    def __init__(self, in_channel, out_channels, layer_nums, layer_strides=None):
        super().__init__()
        if layer_strides is None:
            layer_strides = [2, 2, 2]
        self.multi_blocks = nn.ModuleList()
        for i in range(len(layer_strides)):
            blocks = list()
            blocks.append(nn.Conv2d(in_channel, out_channels[i], 3, stride=layer_strides[i], bias=False, padding=1))
            blocks.append(nn.BatchNorm2d(out_channels[i], eps=1e-3, momentum=0.01))
            blocks.append(nn.ReLU(inplace=True))
            for _ in range(layer_nums[i]):
                blocks.append(nn.Conv2d(out_channels[i], out_channels[i], 3, bias=False, padding=1))
                blocks.append(nn.BatchNorm2d(out_channels[i], eps=1e-3, momentum=0.01))
                blocks.append(nn.ReLU(inplace=True))
            in_channel = out_channels[i]
            self.multi_blocks.append(nn.Sequential(*blocks))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        outs = []
        for i in range(len(self.multi_blocks)):
            x = self.multi_blocks[i](x)
            outs.append(x)
        return outs


class Neck(nn.Module):
    def __init__(self, in_channels, upsample_strides, out_channels):
        super().__init__()
        self.decoder_blocks = nn.ModuleList()
        for i in range(len(in_channels)):
            decoder_block = []
            decoder_block.append(nn.ConvTranspose2d(in_channels[i],
                                                    out_channels[i],
                                                    upsample_strides[i],
                                                    stride=upsample_strides[i],
                                                    bias=False))
            decoder_block.append(nn.BatchNorm2d(out_channels[i], eps=1e-3, momentum=0.01))
            decoder_block.append(nn.ReLU(inplace=True))

            self.decoder_blocks.append(nn.Sequential(*decoder_block))
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        '''
        x: [(bs, 64, 248, 216), (bs, 128, 124, 108), (bs, 256, 62, 54)]
        return: (bs, 384, 248, 216)
        '''
        outs = []
        for i in range(len(self.decoder_blocks)):
            xi = self.decoder_blocks[i](x[i])  # (bs, 128, 248, 216)
            outs.append(xi)
        out = torch.cat(outs, dim=1)
        return out


class Head(nn.Module):
    def __init__(self, in_channel, n_anchors, n_classes):
        super().__init__()
        self.conv_cls = nn.Conv2d(in_channel, n_anchors * n_classes, 1)
        self.conv_reg = nn.Conv2d(in_channel, n_anchors * 7, 1)
        self.conv_dir_cls = nn.Conv2d(in_channel, n_anchors * 2, 1)

        conv_layer_id = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if conv_layer_id == 0:
                    prior_prob = 0.01
                    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
                    nn.init.constant_(m.bias, bias_init)
                else:
                    nn.init.constant_(m.bias, 0)
                conv_layer_id += 1

    def forward(self, x):
        '''
        x: (bs, 384, 248, 216)
        return:
              bbox_cls_pred: (bs, n_anchors*3, 248, 216)
              bbox_pred: (bs, n_anchors*7, 248, 216)
              bbox_dir_cls_pred: (bs, n_anchors*2, 248, 216)
        '''
        bbox_cls_pred = self.conv_cls(x)
        bbox_pred = self.conv_reg(x)
        bbox_dir_cls_pred = self.conv_dir_cls(x)
        return bbox_cls_pred, bbox_pred, bbox_dir_cls_pred


class Anchor(object):
    def __init__(self, ranges, sizes, rotations):
        assert len(ranges) == len(sizes)
        self.ranges = ranges
        self.sizes = sizes
        self.rotations = rotations

    def get_anchors(self, feature_map_size, anchor_range, anchor_size, rotations):
        device = feature_map_size.device
        # print(feature_map_size, anchor_range, anchor_size, rotations)
        x_centers = torch.linspace(anchor_range[0], anchor_range[3], feature_map_size[1] + 1, device=device)
        y_centers = torch.linspace(anchor_range[1], anchor_range[4], feature_map_size[0] + 1, device=device)
        z_centers = torch.linspace(anchor_range[2], anchor_range[5], 1 + 1, device=device)

        x_shift = (x_centers[1] - x_centers[0]) / 2
        y_shift = (y_centers[1] - y_centers[0]) / 2
        z_shift = (z_centers[1] - z_centers[0]) / 2

        x_centers = x_centers[:feature_map_size[1]] + x_shift  # (feature_map_size[1], )
        y_centers = y_centers[:feature_map_size[0]] + y_shift  # (feature_map_size[0], )
        z_centers = z_centers[:1] + z_shift  # (1, )
        meshgrids = torch.meshgrid(x_centers, y_centers, z_centers, rotations, indexing="ij")
        meshgrids = list(meshgrids)
        for i in range(len(meshgrids)):
            meshgrids[i] = meshgrids[i][..., None]
        anchor_size = anchor_size[None, None, None, None, :]
        repeat_shape = [feature_map_size[1], feature_map_size[0], 1, len(rotations), 1]
        anchor_size = anchor_size.repeat(repeat_shape)
        meshgrids.insert(3, anchor_size)
        anchors = torch.cat(meshgrids, dim=-1).permute(2, 1, 0, 3, 4).contiguous()
        return anchors.squeeze(0)

    def get_multi_anchors(self, feature_map_size):
        device = feature_map_size.device
        ranges = torch.tensor(self.ranges, device=device)
        sizes = torch.tensor(self.sizes, device=device)
        rotations = torch.tensor(self.rotations, device=device)
        multi_anchors = []
        for anchor_range, anchor_size in zip(ranges, sizes):
            anchors = self.get_anchors(feature_map_size, anchor_range, anchor_size, rotations)
            multi_anchors.append(anchors[:, :, None, :, :])
        multi_anchors = torch.cat(multi_anchors, dim=2)
        return multi_anchors

    @staticmethod
    def anchors2bboxes(anchors, deltas):
        '''
        anchors: (M, 7),  (x, y, z, w, l, h, theta)
        deltas: (M, 7)
        return: (M, 7)
        '''
        da = torch.sqrt(anchors[:, 3] ** 2 + anchors[:, 4] ** 2)
        x = deltas[:, 0] * da + anchors[:, 0]
        y = deltas[:, 1] * da + anchors[:, 1]
        z = deltas[:, 2] * anchors[:, 5] + anchors[:, 2] + anchors[:, 5] / 2

        w = anchors[:, 3] * torch.exp(deltas[:, 3])
        l = anchors[:, 4] * torch.exp(deltas[:, 4])
        h = anchors[:, 5] * torch.exp(deltas[:, 5])

        z = z - h / 2

        theta = anchors[:, 6] + deltas[:, 6]

        bboxes = torch.stack([x, y, z, w, l, h, theta], dim=1)
        return bboxes

    @staticmethod
    def bboxes2deltas(bboxes, anchors):
        '''
        bboxes: (M, 7), (x, y, z, w, l, h, theta)
        anchors: (M, 7)
        return: (M, 7)
        '''
        da = torch.sqrt(anchors[:, 3] ** 2 + anchors[:, 4] ** 2)

        dx = (bboxes[:, 0] - anchors[:, 0]) / da
        dy = (bboxes[:, 1] - anchors[:, 1]) / da

        zb = bboxes[:, 2] + bboxes[:, 5] / 2  # bottom center
        za = anchors[:, 2] + anchors[:, 5] / 2  # bottom center
        dz = (zb - za) / anchors[:, 5]  # bottom center

        dw = torch.log(bboxes[:, 3] / anchors[:, 3])
        dl = torch.log(bboxes[:, 4] / anchors[:, 4])
        dh = torch.log(bboxes[:, 5] / anchors[:, 5])
        dtheta = bboxes[:, 6] - anchors[:, 6]

        deltas = torch.stack([dx, dy, dz, dw, dl, dh, dtheta], dim=1)
        return deltas


class Assigner(object):
    def __init__(self, anchor, thresh, n_classes=3):
        self.anchor = anchor
        self.thresh = thresh
        self.n_classes = n_classes

    def __call__(self, feature_map_size, targets: List[AnnotationKitti]):
        anchors = self.anchor.get_multi_anchors(feature_map_size)
        batch_size = len(targets)
        batched_labels, batched_label_weights = [], []
        batched_bbox_reg, batched_bbox_reg_weights = [], []
        batched_dir_labels, batched_dir_labels_weights = [], []

        for i in range(batch_size):
            gt_boxes = targets[i].valid_bboxes_in_lidar
            gt_labels = targets[i].labels
            gt_boxes = torch.from_numpy(gt_boxes).float().to(feature_map_size.device)
            gt_labels = torch.tensor(gt_labels).long().to(feature_map_size.device)
            ly, lx, na, nr, c = anchors.shape

            multi_labels, multi_label_weights = [], []
            multi_bbox_reg, multi_bbox_reg_weights = [], []
            multi_dir_labels, multi_dir_labels_weights = [], []

            for j in range(len(self.thresh)):
                assigner = self.thresh[j]
                pos_iou_thr, neg_iou_thr, min_iou_thr = \
                    assigner['pos_iou_thr'], assigner['neg_iou_thr'], assigner['min_iou_thr']
                cur_anchors = anchors[:, :, j, :, :].reshape(-1, 7)
                overlaps = iou2d_nearest(gt_boxes, cur_anchors)

                max_overlaps, max_overlaps_idx = torch.max(overlaps, dim=0)

                gt_max_overlaps, _ = torch.max(overlaps, dim=1)

                assigned_gt_inds = -torch.ones_like(cur_anchors[:, 0], dtype=torch.long)

                assigned_gt_inds[max_overlaps < neg_iou_thr] = 0

                assigned_gt_inds[max_overlaps >= pos_iou_thr] = max_overlaps_idx[max_overlaps >= pos_iou_thr] + 1
                for k in range(len(gt_boxes)):
                    if gt_max_overlaps[k] >= min_iou_thr:
                        assigned_gt_inds[overlaps[k] == gt_max_overlaps[k]] = k + 1
                pos_flag = assigned_gt_inds > 0
                neg_flag = assigned_gt_inds == 0

                assigned_gt_labels = torch.zeros_like(cur_anchors[:, 0], dtype=torch.long) + self.n_classes
                assigned_gt_labels[pos_flag] = gt_labels[assigned_gt_inds[pos_flag] - 1].long()
                assigned_gt_labels_weights = torch.zeros_like(cur_anchors[:, 0])
                assigned_gt_labels_weights[pos_flag] = 1
                assigned_gt_labels_weights[neg_flag] = 1

                assigned_gt_reg_weights = torch.zeros_like(cur_anchors[:, 0])
                assigned_gt_reg_weights[pos_flag] = 1

                assigned_gt_reg = torch.zeros_like(cur_anchors)

                positive_anchors = cur_anchors[pos_flag]
                corr_gt_bboxes = gt_boxes[assigned_gt_inds[pos_flag] - 1]
                assigned_gt_reg[pos_flag] = Anchor.bboxes2deltas(corr_gt_bboxes, positive_anchors)

                assigned_gt_dir_weights = torch.zeros_like(cur_anchors[:, 0])
                assigned_gt_dir_weights[pos_flag] = 1

                assigned_gt_dir = torch.zeros_like(cur_anchors[:, 0], dtype=torch.long)

                dir_cls_targets = limit_period(corr_gt_bboxes[:, 6].cpu(), 0, 2 * np.pi).to(corr_gt_bboxes)

                dir_cls_targets = torch.floor(dir_cls_targets / np.pi).long()
                assigned_gt_dir[pos_flag] = torch.clamp(dir_cls_targets, min=0, max=1)

                multi_labels.append(assigned_gt_labels.reshape(ly, lx, 1, nr))
                multi_label_weights.append(assigned_gt_labels_weights.reshape(ly, lx, 1, nr))
                multi_bbox_reg.append(assigned_gt_reg.reshape(ly, lx, 1, nr, -1))
                multi_bbox_reg_weights.append(assigned_gt_reg_weights.reshape(ly, lx, 1, nr))
                multi_dir_labels.append(assigned_gt_dir.reshape(ly, lx, 1, nr))
                multi_dir_labels_weights.append(assigned_gt_dir_weights.reshape(ly, lx, 1, nr))
            multi_labels = torch.cat(multi_labels, dim=-2).reshape(-1)
            multi_label_weights = torch.cat(multi_label_weights, dim=-2).reshape(-1)
            multi_bbox_reg = torch.cat(multi_bbox_reg, dim=-3).reshape(-1, c)
            multi_bbox_reg_weights = torch.cat(multi_bbox_reg_weights, dim=-2).reshape(-1)
            multi_dir_labels = torch.cat(multi_dir_labels, dim=-2).reshape(-1)
            multi_dir_labels_weights = torch.cat(multi_dir_labels_weights, dim=-2).reshape(-1)

            batched_labels.append(multi_labels)
            batched_label_weights.append(multi_label_weights)
            batched_bbox_reg.append(multi_bbox_reg)
            batched_bbox_reg_weights.append(multi_bbox_reg_weights)
            batched_dir_labels.append(multi_dir_labels)
            batched_dir_labels_weights.append(multi_dir_labels_weights)
        rt_dict = dict(
            batched_labels=torch.stack(batched_labels, 0),  # (bs, y_l * x_l * 3 * 2)
            batched_label_weights=torch.stack(batched_label_weights, 0),  # (bs, y_l * x_l * 3 * 2)
            batched_bbox_reg=torch.stack(batched_bbox_reg, 0),  # (bs, y_l * x_l * 3 * 2, 7)
            batched_bbox_reg_weights=torch.stack(batched_bbox_reg_weights, 0),  # (bs, y_l * x_l * 3 * 2)
            batched_dir_labels=torch.stack(batched_dir_labels, 0),  # (bs, y_l * x_l * 3 * 2)
            batched_dir_labels_weights=torch.stack(batched_dir_labels_weights, 0)  # (bs, y_l * x_l * 3 * 2)
        )
        return rt_dict


class PointPillar(nn.Module):
    def __init__(self, n_classes=3, voxel_size=(0.16, 0.16, 4), point_cloud_range=(0, -39.68, -3, 69.12, 39.68, 1)):
        super().__init__()
        self.n_classes = n_classes
        self.nms_pre = 100
        self.nms_thr = 0.01
        self.score_thr = 0.1
        self.max_num = 50
        self.pillar_encoder = PillarEncoder(voxel_size=voxel_size, voxel_range=point_cloud_range, in_channel=9, out_channel=64)
        self.backbone = Backbone(in_channel=64, out_channels=[64, 128, 256], layer_nums=[3, 5, 5])
        self.neck = Neck(in_channels=[64, 128, 256], upsample_strides=[1, 2, 4], out_channels=[128, 128, 128])
        self.head = Head(in_channel=384, n_anchors=2 * n_classes, n_classes=n_classes)
        self.anchor = Anchor(
            ranges=[[0, -39.68, -0.6, 69.12, 39.68, -0.6],
                    [0, -39.68, -0.6, 69.12, 39.68, -0.6],
                    [0, -39.68, -1.78, 69.12, 39.68, -1.78]],
            sizes=[[0.6, 0.8, 1.73], [0.6, 1.76, 1.73], [1.6, 3.9, 1.56]],
            rotations=[0, 1.57]
        )
        self.assigner = Assigner(anchor=self.anchor, thresh=[
            {'pos_iou_thr': 0.5, 'neg_iou_thr': 0.35, 'min_iou_thr': 0.35},
            {'pos_iou_thr': 0.5, 'neg_iou_thr': 0.35, 'min_iou_thr': 0.35},
            {'pos_iou_thr': 0.6, 'neg_iou_thr': 0.45, 'min_iou_thr': 0.45},
        ], n_classes=n_classes)

        self.loss = Loss()

    def forward(self, v, c, n, targets: List[AnnotationKitti] = None):
        x = self.pillar_encoder(v, c, n)
        xs = self.backbone(x)
        bbox_cls_pred, bbox_pred, bbox_dir_cls_pred = self.head(self.neck(xs))
        feature_size = torch.tensor(list(bbox_cls_pred.size()[-2:]), device=bbox_cls_pred.device)
        if self.training:
            ret = self.compute_loss(bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, targets, feature_size)
        else:
            anchors = self.anchor.get_multi_anchors(feature_size)
            ret = self.predicts(bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, anchors)
        return ret

    @torch.no_grad()
    def predicts(self, bboxes_label_preds, bboxes_preds, bboxes_dir_preds, anchors):
        anchors = anchors.reshape(-1, 7)
        ret = list()
        for bbox_cls_pred, bbox_pred, bbox_dir_cls_pred in zip(bboxes_label_preds, bboxes_preds, bboxes_dir_preds):

            bbox_cls_pred = bbox_cls_pred.permute(1, 2, 0).reshape(-1, self.n_classes)
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 7)
            bbox_dir_cls_pred = bbox_dir_cls_pred.permute(1, 2, 0).reshape(-1, 2)
            bbox_cls_pred = torch.sigmoid(bbox_cls_pred)
            bbox_dir_cls_pred = torch.max(bbox_dir_cls_pred, dim=1)[1]
            max_score, max_idx = bbox_cls_pred.max(dim=-1)
            idx = torch.argsort(max_score, descending=True)[:self.nms_pre]
            bbox_pred = bbox_pred[idx, :]
            bbox_dir_cls_pred = bbox_dir_cls_pred[idx]
            filtered_anchor = anchors[idx, :]

            bbox_pred = Anchor.anchors2bboxes(filtered_anchor, bbox_pred)
            score, cls_id = max_score[idx], max_idx[idx]

            bbox_pred2d_xy = bbox_pred[:, [0, 1]]
            bbox_pred2d_wl = bbox_pred[:, [3, 4]]

            bbox_pred2d = torch.cat([bbox_pred2d_xy, bbox_pred2d_wl, torch.rad2deg(bbox_pred[:, 6:])], dim=-1)
            det = list()
            score_mask = score >= self.score_thr
            for i in range(self.n_classes):
                cls_mask = ((cls_id == i) & score_mask)
                if cls_mask.sum() == 0:
                    continue
                c_score = score[cls_mask]
                c_boxes = bbox_pred[cls_mask]
                c_dir = bbox_dir_cls_pred[cls_mask]
                c_bbox2d = bbox_pred2d[cls_mask]
                keep = nms_rotate_cpu(c_bbox2d.cpu().numpy(), c_score.cpu().numpy(), self.nms_thr, max_output_size=self.nms_pre)
                c_score = c_score[keep]
                c_boxes = c_boxes[keep]
                c_dir = c_dir[keep]
                c_boxes[:, -1] = limit_period(c_boxes[:, -1].cpu(), 1, np.pi).to(c_boxes)
                c_boxes[:, -1] += (1 - c_dir) * np.pi
                det.append(torch.cat([c_boxes, c_score[:, None], cls_id[cls_mask][keep][:, None].to(c_boxes)], dim=-1))
            det = torch.cat(det, dim=0) if len(det) else None
            ret.append(det)
        return {"boxes": ret}

    def compute_loss(self, bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, targets, feature_size):
        bbox_cls_pred = bbox_cls_pred.permute(0, 2, 3, 1).reshape(-1, self.n_classes)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 7)
        bbox_dir_cls_pred = bbox_dir_cls_pred.permute(0, 2, 3, 1).reshape(-1, 2)
        assert targets is not None
        anchor_targets = self.assigner(feature_size, targets)

        batched_bbox_labels = anchor_targets['batched_labels'].reshape(-1)
        batched_label_weights = anchor_targets['batched_label_weights'].reshape(-1)
        batched_bbox_reg = anchor_targets['batched_bbox_reg'].reshape(-1, 7)

        batched_dir_labels = anchor_targets['batched_dir_labels'].reshape(-1)

        pos_idx = (batched_bbox_labels >= 0) & (batched_bbox_labels < self.n_classes)

        bbox_pred = bbox_pred[pos_idx]
        batched_bbox_reg = batched_bbox_reg[pos_idx]

        bbox_pred[:, -1] = torch.sin(bbox_pred[:, -1].clone()) * torch.cos(batched_bbox_reg[:, -1].clone())
        batched_bbox_reg[:, -1] = torch.cos(bbox_pred[:, -1].clone()) * torch.sin(batched_bbox_reg[:, -1].clone())

        bbox_dir_cls_pred = bbox_dir_cls_pred[pos_idx]
        batched_dir_labels = batched_dir_labels[pos_idx]

        num_cls_pos = (batched_bbox_labels < self.n_classes).sum()

        bbox_cls_pred = bbox_cls_pred[batched_label_weights > 0]

        batched_bbox_labels[batched_bbox_labels < 0] = self.n_classes
        batched_bbox_labels = batched_bbox_labels[batched_label_weights > 0]

        ret = self.loss(bbox_cls_pred=bbox_cls_pred,
                        bbox_pred=bbox_pred,
                        bbox_dir_cls_pred=bbox_dir_cls_pred,
                        batched_labels=batched_bbox_labels,
                        num_cls_pos=num_cls_pos,
                        batched_bbox_reg=batched_bbox_reg,
                        batched_dir_labels=batched_dir_labels)
        return ret
