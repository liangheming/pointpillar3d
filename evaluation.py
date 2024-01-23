import torch
import cv2 as cv
import numpy as np
from tqdm import tqdm
from datasets.kitti import Kitti
from models.pointpillars import PointPillar
from utils.kitti import read_pickle, CalibKitti, AnnotationKitti, read_points, write_pickle
from datasets.voxelize import Voxelize
from datasets.transform import PointRangeFilter
from visualize import keep_bbox_from_image_range, keep_bbox_from_lidar_range

CLASSES = Kitti.CLASSES
LABEL2CLASSES = {v: k for k, v in CLASSES.items()}


@torch.no_grad()
def main():
    format_results = {}
    gt_results = {}
    pcd_limit_range = np.array([0, -40, -3, 70.4, 40, 0.0], dtype=np.float32)
    ckpt = torch.load("workspace/pillar/version_1/ckpt/last.ckpt", map_location="cpu")['state_dict']
    state_dict = dict()
    for k, v in ckpt.items():
        nk = k.replace("pillar.", "")
        state_dict[nk] = v
    pillar = PointPillar(3)
    pillar.load_state_dict(state_dict)
    pillar.cuda()
    pillar.eval()
    pkl = read_pickle("/home/lion/large_data/data/cache/val/training_infos.pkl")
    voxelizer = Voxelize(
        voxel_size=[0.16, 0.16, 4],
        point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1],
        max_num_points=32,
        max_voxels=40000)
    transform = PointRangeFilter([0, -39.68, -3, 69.12, 39.68, 1])
    for k, v in tqdm(pkl.items()):
        image_info, lidar_info = v['image'], v['lidar']
        calib_info: CalibKitti = v['calib']
        annotation: AnnotationKitti = lidar_info['annotation']
        lidar_points = read_points(lidar_info['path'])
        lidar_points, _ = transform(lidar_points, annotation)
        v, c, npo = voxelizer(points=lidar_points)
        v = torch.from_numpy(v).float()
        c = torch.from_numpy(c).long()
        npo = torch.from_numpy(npo).long()
        bs = torch.zeros(size=(len(c), 1), dtype=c.dtype)
        c = torch.cat([bs, c], dim=-1)
        predict = pillar(v.cuda(), c.cuda(), npo.cuda())['boxes'][0]
        if predict is not None:
            predict = predict.cpu()
        else:
            predict = torch.zeros(size=(0, 9))
        # predict = predict[:, :7].numpy()
        predict = predict.numpy()
        shape_wh = image_info['shape_wh']
        img_path = image_info['path']
        result = {
            "lidar_bboxes": predict[:, :7],
            "labels": predict[:, -1],
            "scores": predict[:, -2]
        }
        idx = image_info['id']
        result_filter = keep_bbox_from_image_range(result, calib_info.velo_to_cam, calib_info.r0_rect, calib_info.p2, shape_wh)
        result_filter = keep_bbox_from_lidar_range(result_filter, pcd_limit_range)
        lidar_bboxes = result_filter['lidar_bboxes']
        labels, scores = result_filter['labels'], result_filter['scores']
        bboxes2d, camera_bboxes = result_filter['bboxes2d'], result_filter['camera_bboxes']
        format_result = {
            'name': [],
            'truncated': [],
            'occluded': [],
            'alpha': [],
            'bbox': [],
            'dimensions': [],
            'location': [],
            'rotation_y': [],
            'score': []
        }
        for lidar_bbox, label, score, bbox2d, camera_bbox in \
                zip(lidar_bboxes, labels, scores, bboxes2d, camera_bboxes):
            format_result['name'].append(LABEL2CLASSES[int(label)])
            format_result['truncated'].append(0.0)
            format_result['occluded'].append(0)
            alpha = camera_bbox[6] - np.arctan2(camera_bbox[0], camera_bbox[2])
            format_result['alpha'].append(alpha)
            format_result['bbox'].append(bbox2d)
            format_result['dimensions'].append(camera_bbox[3:6])
            format_result['location'].append(camera_bbox[:3])
            format_result['rotation_y'].append(camera_bbox[6])
            format_result['score'].append(score)
        format_results[idx] = {k: np.array(v) for k, v in format_result.items()}
        gt_results[idx] = annotation.to_dict()
    write_pickle(format_results, "predict.pkl")
    write_pickle(gt_results, "gt.pkl")


if __name__ == '__main__':
    main()
