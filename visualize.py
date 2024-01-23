import torch
import cv2 as cv
import numpy as np
from datasets.kitti import Kitti
from models.pointpillars import PointPillar
from utils.kitti import read_pickle, CalibKitti, AnnotationKitti, read_points
from datasets.voxelize import Voxelize
from datasets.transform import PointRangeFilter

import open3d as o3d
from utils.kitti import bbox3d2corner

pcd_limit_range = np.array([0, -40, -3, 70.4, 40, 0.0], dtype=np.float32)
COLORS = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]]
COLORS_IMG = [[0, 0, 255], [0, 255, 0], [255, 0, 0], [0, 255, 255]]

LINES = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 0],
    [4, 5],
    [5, 6],
    [6, 7],
    [7, 4],
    [2, 6],
    [7, 3],
    [1, 5],
    [4, 0]
]


def vis_img_3d(img, image_points, labels):
    '''
    img: (h, w, 3)
    image_points: (n, 8, 2)
    labels: (n, )
    '''

    for i in range(len(image_points)):
        label = labels[i]
        bbox_points = image_points[i]  # (8, 2)
        if label >= 0 and label < 3:
            color = COLORS_IMG[int(label)]
        else:
            color = COLORS_IMG[-1]
        for line_id in LINES:
            x1, y1 = bbox_points[line_id[0]]
            x2, y2 = bbox_points[line_id[1]]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv.line(img, (x1, y1), (x2, y2), color, 1)

    return img


def points_camera2image(points, P2):
    '''
    points: shape=(N, 8, 3)
    P2: shape=(4, 4)
    return: shape=(N, 8, 2)
    '''
    extended_points = np.pad(points, ((0, 0), (0, 0), (0, 1)), 'constant', constant_values=1.0)  # (n, 8, 4)
    image_points = extended_points @ P2.T  # (N, 8, 4)
    image_points = image_points[:, :, :2] / image_points[:, :, 2:3]
    return image_points


def bbox3d2corners_camera(bboxes):
    '''
    bboxes: shape=(n, 7)
    return: shape=(n, 8, 3)
        z (front)            6 ------ 5
        /                  / |     / |
       /                  2 -|---- 1 |
      /                   |  |     | |
    |o ------> x(right)   | 7 -----| 4
    |                     |/   o   |/
    |                     3 ------ 0
    |
    v y(down)
    '''
    centers, dims, angles = bboxes[:, :3], bboxes[:, 3:6], bboxes[:, 6]

    # 1.generate bbox corner coordinates, clockwise from minimal point
    bboxes_corners = np.array([[0.5, 0.0, -0.5], [0.5, -1.0, -0.5], [-0.5, -1.0, -0.5], [-0.5, 0.0, -0.5],
                               [0.5, 0.0, 0.5], [0.5, -1.0, 0.5], [-0.5, -1.0, 0.5], [-0.5, 0.0, 0.5]],
                              dtype=np.float32)
    bboxes_corners = bboxes_corners[None, :, :] * dims[:, None, :]  # (1, 8, 3) * (n, 1, 3) -> (n, 8, 3)

    # 2. rotate around y axis
    rot_sin, rot_cos = np.sin(angles), np.cos(angles)
    # in fact, angle
    rot_mat = np.array([[rot_cos, np.zeros_like(rot_cos), rot_sin],
                        [np.zeros_like(rot_cos), np.ones_like(rot_cos), np.zeros_like(rot_cos)],
                        [-rot_sin, np.zeros_like(rot_cos), rot_cos]],
                       dtype=np.float32)  # (3, 3, n)
    rot_mat = np.transpose(rot_mat, (2, 1, 0))  # (n, 3, 3)
    bboxes_corners = bboxes_corners @ rot_mat  # (n, 8, 3)

    # 3. translate to centers
    bboxes_corners += centers[:, None, :]
    return bboxes_corners


def bbox_lidar2camera(bboxes, tr_velo_to_cam, r0_rect):
    '''
    bboxes: shape=(N, 7)
    tr_velo_to_cam: shape=(4, 4)
    r0_rect: shape=(4, 4)
    return: shape=(N, 7)
    '''
    x_size, y_size, z_size = bboxes[:, 3:4], bboxes[:, 4:5], bboxes[:, 5:6]
    xyz_size = np.concatenate([y_size, z_size, x_size], axis=1)
    extended_xyz = np.pad(bboxes[:, :3], ((0, 0), (0, 1)), 'constant', constant_values=1.0)
    rt_mat = r0_rect @ tr_velo_to_cam
    xyz = extended_xyz @ rt_mat.T
    bboxes_camera = np.concatenate([xyz[:, :3], xyz_size, bboxes[:, 6:]], axis=1)
    return bboxes_camera


def keep_bbox_from_image_range(result, tr_velo_to_cam, r0_rect, P2, image_shape):
    '''
    result: dict(lidar_bboxes, labels, scores)
    tr_velo_to_cam: shape=(4, 4)
    r0_rect: shape=(4, 4)
    P2: shape=(4, 4)
    image_shape: (h, w)
    return: dict(lidar_bboxes, labels, scores, bboxes2d, camera_bboxes)
    '''
    w, h = image_shape

    lidar_bboxes = result['lidar_bboxes']
    labels = result['labels']
    scores = result['scores']
    camera_bboxes = bbox_lidar2camera(lidar_bboxes, tr_velo_to_cam, r0_rect)  # (n, 7)
    bboxes_points = bbox3d2corners_camera(camera_bboxes)  # (n, 8, 3)
    image_points = points_camera2image(bboxes_points, P2)  # (n, 8, 2)
    image_x1y1 = np.min(image_points, axis=1)  # (n, 2)
    image_x1y1 = np.maximum(image_x1y1, 0)
    image_x2y2 = np.max(image_points, axis=1)  # (n, 2)
    image_x2y2 = np.minimum(image_x2y2, [w, h])
    bboxes2d = np.concatenate([image_x1y1, image_x2y2], axis=-1)

    keep_flag = (image_x1y1[:, 0] < w) & (image_x1y1[:, 1] < h) & (image_x2y2[:, 0] > 0) & (image_x2y2[:, 1] > 0)

    result = {
        'lidar_bboxes': lidar_bboxes[keep_flag],
        'labels': labels[keep_flag],
        'scores': scores[keep_flag],
        'bboxes2d': bboxes2d[keep_flag],
        'camera_bboxes': camera_bboxes[keep_flag]
    }
    return result


def keep_bbox_from_lidar_range(result, pcd_limit_range):
    '''
    result: dict(lidar_bboxes, labels, scores, bboxes2d, camera_bboxes)
    pcd_limit_range: []
    return: dict(lidar_bboxes, labels, scores, bboxes2d, camera_bboxes)
    '''
    lidar_bboxes, labels, scores = result['lidar_bboxes'], result['labels'], result['scores']
    if 'bboxes2d' not in result:
        result['bboxes2d'] = np.zeros_like(lidar_bboxes[:, :4])
    if 'camera_bboxes' not in result:
        result['camera_bboxes'] = np.zeros_like(lidar_bboxes)
    bboxes2d, camera_bboxes = result['bboxes2d'], result['camera_bboxes']
    flag1 = lidar_bboxes[:, :3] > pcd_limit_range[:3][None, :]  # (n, 3)
    flag2 = lidar_bboxes[:, :3] < pcd_limit_range[3:][None, :]  # (n, 3)
    keep_flag = np.all(flag1, axis=-1) & np.all(flag2, axis=-1)

    result = {
        'lidar_bboxes': lidar_bboxes[keep_flag],
        'labels': labels[keep_flag],
        'scores': scores[keep_flag],
        'bboxes2d': bboxes2d[keep_flag],
        'camera_bboxes': camera_bboxes[keep_flag]
    }
    return result


def visualize_box3d(points, boxes_in_3d):
    edge = [
        [0, 1], [1, 5], [5, 4], [4, 0],
        [3, 7], [7, 6], [6, 2], [2, 3],
        [6, 5], [7, 4], [2, 1], [3, 0]
    ]
    color = [[1, 0, 0] for _ in range(4)]
    color.extend([[0, 1, 0] for _ in range(8)])
    pcd = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(points[:, :3]))
    visual_element = [pcd]
    boxes_in_corner = bbox3d2corner(boxes_in_3d)
    for box in boxes_in_corner:
        line = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(box),
            lines=o3d.utility.Vector2iVector(edge),
        )
        line.colors = o3d.utility.Vector3dVector(color)
        visual_element.append(line)
    o3d.visualization.draw_geometries(visual_element,
                                      zoom=0.02,
                                      front=[-0.93946750203452634, -0.10630439553975832, 0.32573023824928177],
                                      lookat=[-8.8821525693690067, 1.1150945493183868, 4.2441258766639898],
                                      up=[0.33019175326789302, -0.026986804428561113, 0.94352801678625831])


@torch.no_grad()
def main():
    ckpt = torch.load("workspace/pillar/version_0/ckpt/last.ckpt", map_location="cpu")['state_dict']
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
    for k, v in pkl.items():
        image_info, lidar_info = v['image'], v['lidar']
        calib_info: CalibKitti = v['calib']
        annotation: AnnotationKitti = lidar_info['annotation'].remove_invalid()
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
        ret_dict = {
            "lidar_bboxes": predict[:, :7],
            "labels": predict[:, -1],
            "scores": predict[:, -2]
        }
        result_filter = keep_bbox_from_image_range(ret_dict, calib_info.velo_to_cam, calib_info.r0_rect, calib_info.p2, shape_wh)
        result_filter = keep_bbox_from_lidar_range(result_filter, pcd_limit_range)
        # lidar_bboxes = result_filter['lidar_bboxes']
        labels, scores = result_filter['labels'], result_filter['scores']

        bboxes2d, camera_bboxes = result_filter['bboxes2d'], result_filter['camera_bboxes']
        bboxes_corners = bbox3d2corners_camera(camera_bboxes)
        image_points = points_camera2image(bboxes_corners, calib_info.p2)
        img_show = vis_img_3d(cv.imread(img_path), image_points, labels)
        cv.imshow(__name__, img_show)
        cv.waitKey(0)
        # visualize_box3d(lidar_points, predict)
        # break


if __name__ == '__main__':
    main()
