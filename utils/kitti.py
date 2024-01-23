import copy
import os
import pickle
import numba
import numpy as np
import torch

from utils.geometry import projection2crt, get_frustum


class CalibKitti(object):
    def __init__(self):
        self.p0 = None
        self.p1 = None
        self.p2 = None
        self.p3 = None
        self.r0_rect = None
        self.velo_to_cam = None
        self.imu_to_velo = None

    @staticmethod
    def to_homo(p):
        if p.shape == (3, 4):
            return np.concatenate([p, np.array([[0, 0, 0, 1]])], axis=0)
        elif p.shape == (3, 3):
            homo = np.eye(4, dtype=p.dtype)
            homo[:3, :3] = p
            return homo
        else:
            raise NotImplementedError()


def read_calib(file_path):
    calib = CalibKitti()
    with open(file_path, 'r') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    p0 = np.array([item for item in lines[0].split(' ')[1:]], dtype=float).reshape(3, 4)
    p1 = np.array([item for item in lines[1].split(' ')[1:]], dtype=float).reshape(3, 4)
    p2 = np.array([item for item in lines[2].split(' ')[1:]], dtype=float).reshape(3, 4)
    p3 = np.array([item for item in lines[3].split(' ')[1:]], dtype=float).reshape(3, 4)

    r0_rect = np.array([item for item in lines[4].split(' ')[1:]], dtype=float).reshape(3, 3)
    velo_to_cam = np.array([item for item in lines[5].split(' ')[1:]], dtype=float).reshape(3, 4)
    imu_to_velo = np.array([item for item in lines[6].split(' ')[1:]], dtype=float).reshape(3, 4)
    calib.p0 = CalibKitti.to_homo(p0)
    calib.p1 = CalibKitti.to_homo(p1)
    calib.p2 = CalibKitti.to_homo(p2)
    calib.p3 = CalibKitti.to_homo(p3)
    calib.r0_rect = CalibKitti.to_homo(r0_rect)
    calib.velo_to_cam = CalibKitti.to_homo(velo_to_cam)
    calib.imu_to_velo = CalibKitti.to_homo(imu_to_velo)

    return calib


def read_points(file_path, dim=4):
    suffix = os.path.splitext(file_path)[1]
    assert suffix in ['.bin', '.ply']
    if suffix == '.bin':
        return np.fromfile(file_path, dtype=np.float32).reshape(-1, dim)
    else:
        raise NotImplementedError


def write_points(points, file_path):
    assert file_path.endswith(".bin")
    with open(file_path, 'w') as f:
        points.tofile(f)


def camera2lidar(points, r0_rect, velo_to_cam):
    points = np.pad(points, ((0, 0), (0, 1)), "constant", constant_values=1.0)
    rt = np.linalg.inv(r0_rect @ velo_to_cam)
    points = points @ rt.T
    return points[:, :3]


def ground_rectangle_vertex(boxes_corners):
    """
    :param boxes_corners: (8,3)
    :return:(6,4,3)  front back left right bottom up
    """
    rec1 = np.stack([boxes_corners[0], boxes_corners[1], boxes_corners[2], boxes_corners[3]], axis=0)
    rec2 = np.stack([boxes_corners[4], boxes_corners[7], boxes_corners[6], boxes_corners[5]], axis=0)
    rec3 = np.stack([boxes_corners[0], boxes_corners[4], boxes_corners[5], boxes_corners[1]], axis=0)
    rec4 = np.stack([boxes_corners[2], boxes_corners[6], boxes_corners[7], boxes_corners[3]], axis=0)
    rec5 = np.stack([boxes_corners[1], boxes_corners[5], boxes_corners[6], boxes_corners[2]], axis=0)
    rec6 = np.stack([boxes_corners[0], boxes_corners[3], boxes_corners[7], boxes_corners[4]], axis=0)
    ret = np.stack([rec1, rec2, rec3, rec4, rec5, rec6], axis=0)
    return ret


def ground_rectangle_vertex2(bboxes_corners):
    """
    :param bboxes_corners: (n,8,3)
    :return:
    """

    rec1 = np.stack([bboxes_corners[:, 0], bboxes_corners[:, 1], bboxes_corners[:, 2], bboxes_corners[:, 3]],
                    axis=1)  # (n, 4, 3)
    rec2 = np.stack([bboxes_corners[:, 4], bboxes_corners[:, 7], bboxes_corners[:, 6], bboxes_corners[:, 5]],
                    axis=1)  # (n, 4, 3)
    rec3 = np.stack([bboxes_corners[:, 0], bboxes_corners[:, 4], bboxes_corners[:, 5], bboxes_corners[:, 1]],
                    axis=1)  # (n, 4, 3)
    rec4 = np.stack([bboxes_corners[:, 2], bboxes_corners[:, 6], bboxes_corners[:, 7], bboxes_corners[:, 3]],
                    axis=1)  # (n, 4, 3)
    rec5 = np.stack([bboxes_corners[:, 1], bboxes_corners[:, 5], bboxes_corners[:, 6], bboxes_corners[:, 2]],
                    axis=1)  # (n, 4, 3)
    rec6 = np.stack([bboxes_corners[:, 0], bboxes_corners[:, 3], bboxes_corners[:, 7], bboxes_corners[:, 4]],
                    axis=1)  # (n, 4, 3)
    return np.stack([rec1, rec2, rec3, rec4, rec5, rec6], axis=1)


def ground_plane_equation(rectangle_vertex):
    """
    :param rectangle_vertex: (6,4,3)
    :return: (n,4) 法向量以及距离
    """
    vectors = rectangle_vertex[:, :2] - rectangle_vertex[:, 1:3]
    normal = np.cross(vectors[:, 0], vectors[:, 1])  # (6,3)
    normal_distance = (rectangle_vertex[:, 0] * normal).sum(-1)
    plane_params = np.concatenate([normal, -normal_distance[:, None]], axis=-1)
    return plane_params


def ground_plane_equation2(rectangle_vertex):
    """
    :param rectangle_vertex: (n,6,4,3)
    :return:
    """
    vectors = rectangle_vertex[:, :, :2] - rectangle_vertex[:, :, 1:3]
    normal = np.cross(vectors[:, :, 0], vectors[:, :, 1])  # (n, 6, 3)
    normal_distance = (rectangle_vertex[:, :, 0] * normal).sum(-1)
    plane_params = np.concatenate([normal, -normal_distance[:, :, None]], axis=-1)
    return plane_params


@numba.jit(nopython=True)
def points_in_boxes(points, surface):
    """
    :param points:
    :param surface:
    :return: (point - surface_point) * normal
    """
    p_n, s_n = len(points), len(surface)
    mask = np.ones((p_n,), dtype=np.bool_)
    for i in range(p_n):
        x, y, z = points[i]
        for j in range(s_n):
            a, b, c, d = surface[j]
            check = a * x + b * y + c * z + d
            if check >= 0:
                mask[i] = False
                break
    return mask


@numba.jit(nopython=True)
def points_in_boxes2(points, surface):
    """
    :param points:[n, p]
    :param surface: [m,6,4]
    :return:[n,m]
    """
    p_n, b_n = len(points), len(surface)
    s_n = surface.shape[1]
    masks = np.ones((p_n, b_n), dtype=np.bool_)
    for i in range(p_n):
        x, y, z = points[i, :3]
        for j in range(b_n):
            surface_param = surface[j]
            for k in range(s_n):
                a, b, c, d = surface_param[k]
                if a * x + b * y + c * z + d >= 0:
                    masks[i][j] = False
                    break
    return masks


def points_in_boxes_np(points, surface):
    check = (points[:, None, :] * surface[None, :, :3]).sum(-1) + surface[:, 3]
    mask = check.max(-1) < 0
    return mask


def remove_outside_points(points, r0_rect, velo_to_cam, p2, img_shape):
    """
    :param points:
    :param r0_rect:
    :param velo_to_cam:
    :param p2:
    :param img_shape: w, h
    :return:
    """
    c, r, t = projection2crt(p2)
    # pix =  [c|r,t] @ p
    w, h = img_shape
    image_bbox = [0, 0, w, h]
    frustum = get_frustum(image_bbox, c)
    frustum -= t
    # frustum in camera
    frustum = (np.linalg.inv(r) @ frustum.T).T
    # frustum in lidar
    frustum = camera2lidar(frustum, r0_rect, velo_to_cam)
    rectangle_vertex = ground_rectangle_vertex(frustum)
    plane_params = ground_plane_equation(rectangle_vertex)
    mask = points_in_boxes(points[:, :3], plane_params)
    return points[mask]


class AnnotationKitti(object):
    MIN_HEIGHTS = [40, 25, 25]
    MAX_OCCLUSION = [0, 1, 2]
    MAX_TRUNCATION = [0.15, 0.30, 0.50]

    def __init__(self):
        self.names = None
        self.labels = None
        self.truncated = None
        self.occluded = None
        self.alphas = None
        self.bboxes = None  # xmin ymin xmax ymax
        self.dimensions = None  # length height width
        self.locations = None  # bottom center [camera]
        self.rotations = None  # rotation
        self.difficulties = None
        self.is_valid = None
        self.num_of_points = None
        self.valid_bboxes_in_lidar = None  # x y z w l h r

    def judge_difficulty(self):
        truncated = self.truncated
        occluded = self.occluded
        bbox = self.bboxes
        height = bbox[:, 3] - bbox[:, 1]
        difficulties = []
        for h, o, t in zip(height, occluded, truncated):
            difficulty = -1
            for i in range(2, -1, -1):
                if h > AnnotationKitti.MIN_HEIGHTS[i] \
                        and o <= AnnotationKitti.MAX_OCCLUSION[i] \
                        and t <= AnnotationKitti.MAX_TRUNCATION[i]:
                    difficulty = i
            difficulties.append(difficulty)
        self.difficulties = np.array(difficulties, dtype=int)

    def remove_invalid(self):
        assert self.is_valid is not None
        self.names = self.names[self.is_valid]
        self.bboxes = self.bboxes[self.is_valid]
        self.num_of_points = self.num_of_points[self.is_valid]
        self.truncated = self.truncated[self.is_valid]
        self.occluded = self.occluded[self.is_valid]
        self.difficulties = self.difficulties[self.is_valid]
        return self

    def to_dict(self):
        return {
            "bbox": self.bboxes,
            "location": self.locations,
            "dimensions": self.dimensions,
            "rotation_y": self.rotations,
            "name": self.names,
            "difficulty": self.difficulties,
            "alpha": self.alphas
        }


def read_label(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    lines = [line.strip().split(' ') for line in lines]
    annotation = AnnotationKitti()
    annotation.names = np.array([line[0] for line in lines])
    annotation.truncated = np.array([line[1] for line in lines], dtype=float)
    annotation.occluded = np.array([line[2] for line in lines], dtype=int)
    annotation.alphas = np.array([line[3] for line in lines], dtype=float)
    annotation.bboxes = np.array([line[4:8] for line in lines], dtype=float)
    annotation.dimensions = np.array([line[8:11] for line in lines], dtype=float)[:,
                            [2, 0, 1]]  # hwl -> camera coordinates (lhw)
    annotation.locations = np.array([line[11:14] for line in lines], dtype=float)
    annotation.rotations = np.array([line[14] for line in lines], dtype=float)

    return annotation


def boxes_camera2lidar(bboxes, velo_to_cam, r0_rect):
    """
    :param bboxes: location dimension rotation
    :param velo_to_cam:
    :param r0_rect:
    :return:
    """
    x_size, y_size, z_size = bboxes[:, 3:4], bboxes[:, 4:5], bboxes[:, 5:6]
    # xyz_size in lidar
    xyz_size = np.concatenate([z_size, x_size, y_size], axis=1)
    xyz_homo = np.pad(bboxes[:, :3], ((0, 0), (0, 1)), mode="constant", constant_values=1.0)
    # xyz in lidar
    xyz = (xyz_homo @ np.linalg.inv(r0_rect @ velo_to_cam).T)[:, :3]
    bboxes_lidar = np.concatenate([xyz, xyz_size, bboxes[:, 6:]], axis=1)
    return bboxes_lidar


def bbox3d2corner(bboxes):
    """
    bboxes: shape=(n, 7)
    return: shape=(n, 8, 3)
           ^ z   x            6 ------ 5
           |   /             / |     / |
           |  /             2 -|---- 1 |
    y      | /              |  |     | |
    <------|o               | 7 -----| 4
                            |/   o   |/
                            3 ------ 0
    x: front, y: left, z: top
    """
    centers, dims, angles = bboxes[:, :3], bboxes[:, 3:6], bboxes[:, 6]
    bboxes_corners = np.array([[-0.5, -0.5, 0], [-0.5, -0.5, 1.0], [-0.5, 0.5, 1.0], [-0.5, 0.5, 0.0],
                               [0.5, -0.5, 0], [0.5, -0.5, 1.0], [0.5, 0.5, 1.0], [0.5, 0.5, 0.0]],
                              dtype=float)
    bboxes_corners = bboxes_corners[None, :, :] * dims[:, None, :]
    rot_sin, rot_cos = np.sin(angles), np.cos(angles)

    rot_mat = np.array([[rot_cos, rot_sin, np.zeros_like(rot_cos)],
                        [-rot_sin, rot_cos, np.zeros_like(rot_cos)],
                        [np.zeros_like(rot_cos), np.zeros_like(rot_cos), np.ones_like(rot_cos)]],
                       dtype=float)
    rot_mat = np.transpose(rot_mat, (2, 1, 0))

    bboxes_corners = bboxes_corners @ rot_mat
    bboxes_corners += centers[:, None, :]
    return bboxes_corners


def points_in_multi_boxes(points, r0_rect, velo_to_cam, dimensions, locations, rotations, names):
    """
    :param points: 
    :param r0_rect: 
    :param velo_to_cam: 
    :param dimensions: 
    :param locations: 
    :param rotations: 
    :param names: 
    :return: 
    """
    # n_total_bbox = len(dimensions)
    n_valid_bbox = len([item for item in names if item != 'DontCare'])
    locations, dimensions = locations[:n_valid_bbox], dimensions[:n_valid_bbox]
    rotations, names = rotations[:n_valid_bbox], names[:n_valid_bbox]

    bboxes_camera = np.concatenate([locations, dimensions, rotations[:, None]], axis=1)
    bboxes_lidar = boxes_camera2lidar(bboxes_camera, velo_to_cam, r0_rect)
    bboxes_corner = bbox3d2corner(bboxes_lidar)
    rectangle_vertex = ground_rectangle_vertex2(bboxes_corner)
    surface_param = ground_plane_equation2(rectangle_vertex)
    # p_n, b_n
    indices = points_in_boxes2(points[:, :3], surface_param)
    return indices, n_valid_bbox, bboxes_lidar, names


def write_pickle(results, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(results, f)


def read_pickle(file_path, suffix='.pkl'):
    assert os.path.splitext(file_path)[1] == suffix
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def bbox3d2bevcorners(bboxes):
    '''
    bboxes: shape=(n, 7)

                ^ x (-0.5 * pi)
                |
                |                (bird's eye view)
       (-pi)  o |
        y <-------------- (0)
                 \ / (ag)
                  \
                   \

    return: shape=(n, 4, 2)
    '''
    centers, dims, angles = bboxes[:, :2], bboxes[:, 3:5], bboxes[:, 6]

    # 1.generate bbox corner coordinates, clockwise from minimal point
    bev_corners = np.array([[-0.5, -0.5], [-0.5, 0.5], [0.5, 0.5], [0.5, -0.5]], dtype=np.float32)
    bev_corners = bev_corners[None, ...] * dims[:, None, :]  # (1, 4, 2) * (n, 1, 2) -> (n, 4, 2)

    # 2. rotate
    rot_sin, rot_cos = np.sin(angles), np.cos(angles)
    # in fact, -angle
    rot_mat = np.array([[rot_cos, rot_sin],
                        [-rot_sin, rot_cos]])  # (2, 2, n)
    rot_mat = np.transpose(rot_mat, (2, 1, 0))  # (N, 2, 2)
    bev_corners = bev_corners @ rot_mat  # (n, 4, 2)

    # 3. translate to centers
    bev_corners += centers[:, None, :]
    return bev_corners.astype(np.float32)


@numba.jit(nopython=True)
def bevcorner2alignedbbox(bev_corners):
    '''
    bev_corners: shape=(N, 4, 2)
    return: shape=(N, 4)
    '''
    # xmin, xmax = np.min(bev_corners[:, :, 0], axis=-1), np.max(bev_corners[:, :, 0], axis=-1)
    # ymin, ymax = np.min(bev_corners[:, :, 1], axis=-1), np.max(bev_corners[:, :, 1], axis=-1)

    # why we don't implement like the above ? please see
    # https://numba.pydata.org/numba-doc/latest/reference/numpysupported.html#calculation
    n = len(bev_corners)
    alignedbbox = np.zeros((n, 4), dtype=np.float32)
    for i in range(n):
        cur_bev = bev_corners[i]
        alignedbbox[i, 0] = np.min(cur_bev[:, 0])
        alignedbbox[i, 2] = np.max(cur_bev[:, 0])
        alignedbbox[i, 1] = np.min(cur_bev[:, 1])
        alignedbbox[i, 3] = np.max(cur_bev[:, 1])
    return alignedbbox


@numba.jit(nopython=True)
def box_collision_test(boxes, qboxes, clockwise=True):
    """Box collision test.
    Args:
        boxes (np.ndarray): Corners of current boxes. # (n1, 4, 2)
        qboxes (np.ndarray): Boxes to be avoid colliding. # (n2, 4, 2)
        clockwise (bool, optional): Whether the corners are in
            clockwise order. Default: True.
    return: shape=(n1, n2)
    """
    N = boxes.shape[0]
    K = qboxes.shape[0]
    ret = np.zeros((N, K), dtype=np.bool_)
    slices = np.array([1, 2, 3, 0])
    lines_boxes = np.stack((boxes, boxes[:, slices, :]),
                           axis=2)  # [N, 4, 2(line), 2(xy)]
    lines_qboxes = np.stack((qboxes, qboxes[:, slices, :]), axis=2)
    # vec = np.zeros((2,), dtype=boxes.dtype)
    boxes_standup = bevcorner2alignedbbox(boxes)
    qboxes_standup = bevcorner2alignedbbox(qboxes)
    for i in range(N):
        for j in range(K):
            # calculate standup first
            iw = (
                    min(boxes_standup[i, 2], qboxes_standup[j, 2]) -
                    max(boxes_standup[i, 0], qboxes_standup[j, 0]))
            if iw > 0:
                ih = (
                        min(boxes_standup[i, 3], qboxes_standup[j, 3]) -
                        max(boxes_standup[i, 1], qboxes_standup[j, 1]))
                if ih > 0:
                    for k in range(4):
                        for box_l in range(4):
                            A = lines_boxes[i, k, 0]
                            B = lines_boxes[i, k, 1]
                            C = lines_qboxes[j, box_l, 0]
                            D = lines_qboxes[j, box_l, 1]
                            acd = (D[1] - A[1]) * (C[0] -
                                                   A[0]) > (C[1] - A[1]) * (
                                          D[0] - A[0])
                            bcd = (D[1] - B[1]) * (C[0] -
                                                   B[0]) > (C[1] - B[1]) * (
                                          D[0] - B[0])
                            if acd != bcd:
                                abc = (C[1] - A[1]) * (B[0] - A[0]) > (
                                        B[1] - A[1]) * (
                                              C[0] - A[0])
                                abd = (D[1] - A[1]) * (B[0] - A[0]) > (
                                        B[1] - A[1]) * (
                                              D[0] - A[0])
                                if abc != abd:
                                    ret[i, j] = True  # collision.
                                    break
                        if ret[i, j] is True:
                            break
                    if ret[i, j] is False:
                        # now check complete overlap.
                        # box overlap qbox:
                        box_overlap_qbox = True
                        for box_l in range(4):  # point l in qboxes
                            for k in range(4):  # corner k in boxes
                                vec = boxes[i, k] - boxes[i, (k + 1) % 4]
                                if clockwise:
                                    vec = -vec
                                cross = vec[1] * (
                                        boxes[i, k, 0] - qboxes[j, box_l, 0])
                                cross -= vec[0] * (
                                        boxes[i, k, 1] - qboxes[j, box_l, 1])
                                if cross >= 0:
                                    box_overlap_qbox = False
                                    break
                            if box_overlap_qbox is False:
                                break

                        if box_overlap_qbox is False:
                            qbox_overlap_box = True
                            for box_l in range(4):  # point box_l in boxes
                                for k in range(4):  # corner k in qboxes
                                    vec = qboxes[j, k] - qboxes[j, (k + 1) % 4]
                                    if clockwise:
                                        vec = -vec
                                    cross = vec[1] * (
                                            qboxes[j, k, 0] - boxes[i, box_l, 0])
                                    cross -= vec[0] * (
                                            qboxes[j, k, 1] - boxes[i, box_l, 1])
                                    if cross >= 0:  #
                                        qbox_overlap_box = False
                                        break
                                if qbox_overlap_box is False:
                                    break
                            if qbox_overlap_box:
                                ret[i, j] = True  # collision.
                        else:
                            ret[i, j] = True  # collision.
    return ret


def remove_pts_in_bboxes(points, bboxes, rm=True):
    '''
    points: shape=(N, 3)
    bboxes: shape=(n, 7)
    return: shape=(N, n), bool
    '''
    # 1. get 6 groups of rectangle vertexs
    bboxes_corners = bbox3d2corner(bboxes)  # (n, 8, 3)
    bbox_group_rectangle_vertexs = ground_rectangle_vertex2(bboxes_corners)  # (n, 6, 4, 3)

    # 2. calculate plane equation: ax + by + cd + d = 0
    group_plane_equation_params = ground_plane_equation2(bbox_group_rectangle_vertexs)

    # 3. Judge each point inside or outside the bboxes
    # if point (x0, y0, z0) lies on the direction of normal vector(a, b, c), then ax0 + by0 + cz0 + d > 0.
    masks = points_in_boxes2(points, group_plane_equation_params)  # (N, n)

    if not rm:
        return masks

    # 4. remove point insider the bboxes
    masks = np.any(masks, axis=-1)

    return points[~masks]


def limit_period(val, offset=0.5, period=np.pi):
    """
    val: array or float
    offset: float
    period: float
    return: Value in the range of [-offset * period, (1-offset) * period]
    """
    limited_val = val - np.floor(val / period + offset) * period
    return limited_val


def nearest_bev(boxes):
    bboxes_bev = copy.deepcopy(boxes[:, [0, 1, 3, 4]])
    boxes_angle = limit_period(boxes[:, 6].cpu(), offset=0.5, period=np.pi).to(bboxes_bev)
    bboxes_bev = torch.where(torch.abs(boxes_angle[:, None]) > np.pi / 4, bboxes_bev[:, [0, 1, 3, 2]], bboxes_bev)
    bboxes_xy = bboxes_bev[:, :2]
    bboxes_wl = bboxes_bev[:, 2:]
    bboxes_bev_x1y1x2y2 = torch.cat([bboxes_xy - bboxes_wl / 2, bboxes_xy + bboxes_wl / 2], dim=-1)
    return bboxes_bev_x1y1x2y2


def iou2d(bboxes1, bboxes2, metric=0):
    '''
    bboxes1: (n, 4), (x1, y1, x2, y2)
    bboxes2: (m, 4), (x1, y1, x2, y2)
    return: (n, m)
    '''
    bboxes_x1 = torch.maximum(bboxes1[:, 0][:, None], bboxes2[:, 0][None, :])  # (n, m)
    bboxes_y1 = torch.maximum(bboxes1[:, 1][:, None], bboxes2[:, 1][None, :])  # (n, m)
    bboxes_x2 = torch.minimum(bboxes1[:, 2][:, None], bboxes2[:, 2][None, :])
    bboxes_y2 = torch.minimum(bboxes1[:, 3][:, None], bboxes2[:, 3][None, :])

    bboxes_w = torch.clamp(bboxes_x2 - bboxes_x1, min=0)
    bboxes_h = torch.clamp(bboxes_y2 - bboxes_y1, min=0)

    iou_area = bboxes_w * bboxes_h  # (n, m)

    bboxes1_wh = bboxes1[:, 2:] - bboxes1[:, :2]
    area1 = bboxes1_wh[:, 0] * bboxes1_wh[:, 1]  # (n, )
    bboxes2_wh = bboxes2[:, 2:] - bboxes2[:, :2]
    area2 = bboxes2_wh[:, 0] * bboxes2_wh[:, 1]  # (m, )
    if metric == 0:
        iou = iou_area / (area1[:, None] + area2[None, :] - iou_area + 1e-8)
    elif metric == 1:
        iou = iou_area / (area1[:, None] + 1e-8)
    return iou


def iou2d_nearest(bboxes1, bboxes2):
    '''
    bboxes1: (n, 7), (x, y, z, w, l, h, theta)
    bboxes2: (m, 7),
    return: (n, m)
    '''
    bboxes1_bev = nearest_bev(bboxes1)
    bboxes2_bev = nearest_bev(bboxes2)
    iou = iou2d(bboxes1_bev, bboxes2_bev)
    return iou
