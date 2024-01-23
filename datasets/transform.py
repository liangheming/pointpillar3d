import copy
import random
import numba
import numpy as np
from utils.kitti import AnnotationKitti, bbox3d2bevcorners, box_collision_test, read_points, remove_pts_in_bboxes, limit_period


@numba.jit(nopython=True)
def object_noise_core(pts, gt_bboxes_3d, bev_corners, trans_vec, rot_angle, rot_mat, masks):
    '''
    pts: (N, 4)
    gt_bboxes_3d: (n_bbox, 7)
    bev_corners: ((n_bbox, 4, 2))
    trans_vec: (n_bbox, num_try, 3)
    rot_mat: (n_bbox, num_try, 2, 2)
    masks: (N, n_bbox), bool
    return: gt_bboxes_3d, pts
    '''
    # 1. select the noise of num_try for each bbox under the collision test
    n_bbox, num_try = trans_vec.shape[:2]

    # succ_mask: (n_bbox, ), whether each bbox can be added noise successfully. -1 denotes failure.
    succ_mask = -np.ones((n_bbox,), dtype=np.int_)
    for i in range(n_bbox):
        for j in range(num_try):
            cur_bbox = bev_corners[i] - np.expand_dims(gt_bboxes_3d[i, :2], 0)  # (4, 2) - (1, 2) -> (4, 2)
            rot = np.zeros((2, 2), dtype=np.float32)
            rot[:] = rot_mat[i, j]  # (2, 2)
            trans = trans_vec[i, j]  # (3, )
            cur_bbox = cur_bbox @ rot
            cur_bbox += gt_bboxes_3d[i, :2]
            cur_bbox += np.expand_dims(trans[:2], 0)  # (4, 2)
            coll_mat = box_collision_test(np.expand_dims(cur_bbox, 0), bev_corners)
            coll_mat[0, i] = False
            if coll_mat.any():
                continue
            else:
                bev_corners[i] = cur_bbox  # update the bev_corners when adding noise succseefully.
                succ_mask[i] = j
                break
    # 2. points and bboxes noise
    visit = {}
    for i in range(n_bbox):
        jj = succ_mask[i]
        if jj == -1:
            continue
        cur_trans, cur_angle = trans_vec[i, jj], rot_angle[i, jj]
        cur_rot_mat = np.zeros((2, 2), dtype=np.float32)
        cur_rot_mat[:] = rot_mat[i, jj]
        for k in range(len(pts)):
            if masks[k][i] and k not in visit:
                cur_pt = pts[k]  # (4, )
                cur_pt_xyz = np.zeros((1, 3), dtype=np.float32)
                cur_pt_xyz[0] = cur_pt[:3] - gt_bboxes_3d[i][:3]
                tmp_cur_pt_xy = np.zeros((1, 2), dtype=np.float32)
                tmp_cur_pt_xy[:] = cur_pt_xyz[:, :2]
                cur_pt_xyz[:, :2] = tmp_cur_pt_xy @ cur_rot_mat  # (1, 2)
                cur_pt_xyz[0] = cur_pt_xyz[0] + gt_bboxes_3d[i][:3]
                cur_pt_xyz[0] = cur_pt_xyz[0] + cur_trans[:3]
                cur_pt[:3] = cur_pt_xyz[0]
                visit[k] = 1

        gt_bboxes_3d[i, :3] += cur_trans[:3]
        gt_bboxes_3d[i, 6] += cur_angle

    return gt_bboxes_3d, pts


class DBSample(object):
    def __init__(self, db, sample_dict):
        self.db = db
        self.sample_dict = sample_dict

    def __call__(self, points, annotation: AnnotationKitti):
        avoid_coll_boxes = copy.deepcopy(annotation.valid_bboxes_in_lidar)
        sampled_pts, sampled_names, sampled_labels = [], [], []
        sampled_bboxes, sampled_difficulty = [], []
        for k, v in self.sample_dict.items():
            sample_num = v - np.sum(annotation.names == k)
            if sample_num <= 0:
                continue
            idx_list = np.random.randint(low=0, high=len(self.db[k]), size=(sample_num,))
            sample_candidates = [self.db[k][i] for i in idx_list]
            sample_boxes = np.array([item['box3d_lidar'] for item in sample_candidates], dtype=float)
            avoid_coll_boxes_bv_corners = bbox3d2bevcorners(avoid_coll_boxes)
            sampled_cls_bboxes_bv_corners = bbox3d2bevcorners(sample_boxes)
            coll_query_matrix = np.concatenate([avoid_coll_boxes_bv_corners, sampled_cls_bboxes_bv_corners], axis=0)
            coll_mat = box_collision_test(coll_query_matrix, coll_query_matrix)

            n_gt, tmp_bboxes = len(avoid_coll_boxes_bv_corners), []
            for i in range(n_gt, len(coll_mat)):
                if any(coll_mat[i]):
                    coll_mat[i] = False
                    coll_mat[:, i] = False
                else:
                    cur_sample = sample_candidates[i - n_gt]
                    sampled_pts_cur = read_points(cur_sample["path"])
                    sampled_pts_cur[:, :3] += cur_sample["box3d_lidar"][:3]
                    sampled_pts.append(sampled_pts_cur)
                    sampled_names.append(cur_sample["name"])
                    sampled_bboxes.append(cur_sample["box3d_lidar"])
                    tmp_bboxes.append(cur_sample["box3d_lidar"])
                    sampled_difficulty.append(cur_sample["difficulty"])

            if len(tmp_bboxes) == 0:
                tmp_bboxes = np.array(tmp_bboxes).reshape(-1, 7)
            else:
                tmp_bboxes = np.array(tmp_bboxes)
            avoid_coll_boxes = np.concatenate([avoid_coll_boxes, tmp_bboxes], axis=0)
        points = remove_pts_in_bboxes(points, np.stack(sampled_bboxes, axis=0))
        points = np.concatenate([np.concatenate(sampled_pts, axis=0), points], axis=0)
        annotation.valid_bboxes_in_lidar = avoid_coll_boxes.astype(float)
        annotation.names = np.concatenate([annotation.names, np.array(sampled_names)], axis=0)
        annotation.difficulties = np.concatenate([annotation.difficulties, np.array(sampled_difficulty)], axis=0)
        return points, annotation

    def filter_db(self, sample_thresh):
        for k, v in sample_thresh.items():
            self.db[k] = [item for item in self.db[k] if item['num_points_in_gt'] >= v and item['difficulty'] != -1]


class ObjectNoise(object):
    def __init__(self, num_try, translate, rotation):
        self.num_try = num_try
        self.translate = translate
        self.rotation = rotation

    def __call__(self, points, annotations: AnnotationKitti):
        pts, gt_bboxes_3d = points, annotations.valid_bboxes_in_lidar
        n_bbox = len(gt_bboxes_3d)
        trans_vec = np.random.normal(scale=self.translate, size=(n_bbox, self.num_try, 3)).astype(np.float32)
        rot_angle = np.random.uniform(self.rotation[0], self.rotation[1], size=(n_bbox, self.num_try)).astype(
            np.float32)
        rot_cos, rot_sin = np.cos(rot_angle), np.sin(rot_angle)
        rot_mat = np.array([[rot_cos, rot_sin],
                            [-rot_sin, rot_cos]])

        rot_mat = np.transpose(rot_mat, (2, 3, 1, 0))
        bev_corners = bbox3d2bevcorners(gt_bboxes_3d)
        masks = remove_pts_in_bboxes(pts, gt_bboxes_3d, rm=False)
        gt_bboxes_3d, pts = object_noise_core(pts=pts,
                                              gt_bboxes_3d=gt_bboxes_3d.astype(np.float32),
                                              bev_corners=bev_corners,
                                              trans_vec=trans_vec,
                                              rot_angle=rot_angle,
                                              rot_mat=rot_mat,
                                              masks=masks)
        annotations.valid_bboxes_in_lidar = gt_bboxes_3d.astype(float)

        return pts, annotations


class RandFlip(object):
    def __init__(self, ratio=0.5):
        self.ratio = ratio

    def __call__(self, points, annotation: AnnotationKitti):
        random_flip_state = np.random.choice([True, False], p=[self.ratio, 1 - self.ratio])
        if random_flip_state:
            points[:, 1] = -points[:, 1]
            annotation.valid_bboxes_in_lidar[:, 1] = - annotation.valid_bboxes_in_lidar[:, 1]
            annotation.valid_bboxes_in_lidar[:, 6] = - annotation.valid_bboxes_in_lidar[:, 6] + np.pi
        return points, annotation


class GlobalAffine(object):
    def __init__(self, rotation, scale, translation):
        self.rotation = rotation
        self.scale = scale
        self.translation = translation

    def __call__(self, points, annotation: AnnotationKitti):
        rot_angle = np.random.uniform(self.rotation[0], self.rotation[1])
        rot_cos, rot_sin = np.cos(rot_angle), np.sin(rot_angle)
        rot_mat = np.array([[rot_cos, rot_sin],
                            [-rot_sin, rot_cos]])  # (2, 2)
        gt_bboxes_3d = annotation.valid_bboxes_in_lidar
        gt_bboxes_3d[:, :2] = gt_bboxes_3d[:, :2] @ rot_mat.T
        gt_bboxes_3d[:, 6] += rot_angle
        points[:, :2] = points[:, :2] @ rot_mat.T

        scale_factor = np.random.uniform(self.scale[0], self.scale[1])
        gt_bboxes_3d[:, :6] *= scale_factor
        points[:, :3] *= scale_factor

        trans_factor = np.random.normal(scale=self.translation, size=(1, 3))
        gt_bboxes_3d[:, :3] += trans_factor
        points[:, :3] += trans_factor
        annotation.valid_bboxes_in_lidar = gt_bboxes_3d
        return points, annotation


class PointRangeFilter(object):
    def __init__(self, valid_range):
        self.valid_range = valid_range

    def __call__(self, points, annotations):
        x1, y1, z1, x2, y2, z2 = self.valid_range
        flag_x_low = points[:, 0] > x1
        flag_y_low = points[:, 1] > y1
        flag_z_low = points[:, 2] > z1
        flag_x_high = points[:, 0] < x2
        flag_y_high = points[:, 1] < y2
        flag_z_high = points[:, 2] < z2

        keep_mask = flag_x_low & flag_y_low & flag_z_low & flag_x_high & flag_y_high & flag_z_high
        points = points[keep_mask]
        return points, annotations


class ObjectRangeFilter(object):
    def __init__(self, valid_range):
        self.valid_range = valid_range

    def __call__(self, points, annotations: AnnotationKitti):
        x1, y1, z1, x2, y2, z2 = self.valid_range
        gt_bboxes_3d = annotations.valid_bboxes_in_lidar
        flag_x_low = gt_bboxes_3d[:, 0] > x1
        flag_y_low = gt_bboxes_3d[:, 1] > y1
        flag_x_high = gt_bboxes_3d[:, 0] < x2
        flag_y_high = gt_bboxes_3d[:, 1] < y2
        keep_mask = flag_x_low & flag_y_low & flag_x_high & flag_y_high
        gt_bboxes_3d[:, 6] = limit_period(gt_bboxes_3d[:, 6], 0.5, 2 * np.pi)

        annotations.valid_bboxes_in_lidar = gt_bboxes_3d[keep_mask]
        annotations.names = annotations.names[keep_mask]
        annotations.difficulties = annotations.difficulties[keep_mask]
        return points, annotations


class PointShuffle(object):
    def __init__(self):
        super().__init__()

    def __call__(self, points, annotations: AnnotationKitti):
        indices = np.arange(0, len(points))
        np.random.shuffle(indices)
        points = points[indices]
        return points, annotations


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, points, annotations):
        for t in self.transforms:
            points, annotations = t(points, annotations)
        return points, annotations


if __name__ == '__main__':
    from utils.kitti import read_pickle

    db_info = read_pickle("/home/lion/large_data/data/cache/train/db/training_db.pkl")
    db_sample = DBSample(db=db_info, sample_dict={"Car": 15, "Pedestrian": 10, "Cyclist": 10})
    db_sample.filter_db({"Car": 5, "Pedestrian": 10, "Cyclist": 10})
