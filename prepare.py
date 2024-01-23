import os
import sys
import argparse
import cv2 as cv
import open3d as o3d
import numpy as np
from tqdm import tqdm
from utils.kitti import read_points, read_calib, remove_outside_points, read_label, points_in_multi_boxes, write_points, \
    read_pickle, write_pickle

cur_dir = os.path.dirname(os.path.abspath(__file__))


def points_show(points):
    pcd = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(points[:, :3]))
    o3d.visualization.draw_geometries([pcd],
                                      zoom=0.02,
                                      front=[-0.93946750203452634, -0.10630439553975832, 0.32573023824928177],
                                      lookat=[-8.8821525693690067, 1.1150945493183868, 4.2441258766639898],
                                      up=[0.33019175326789302, -0.026986804428561113, 0.94352801678625831])


def prepare(arg):
    ids_file = os.path.join(cur_dir, 'splits', f'{arg.split}.txt')
    with open(ids_file, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    split_name = 'training' if arg.is_train else 'testing'

    reduced_points_save_dir = os.path.join(arg.cache_dir, arg.split, "velodyne_reduced")
    os.makedirs(reduced_points_save_dir, exist_ok=True)
    db_dir = ""
    db_points_dir = ""
    kitti_db_info_train = dict()
    kitti_infos_dict = dict()
    if arg.save_db:
        db_dir = os.path.join(arg.cache_dir, arg.split, "db")
        db_points_dir = os.path.join(arg.cache_dir, arg.split, "db", "velodyne_instance")
        os.makedirs(db_points_dir, exist_ok=True)

    for line in tqdm(lines):
        cur_info_dict = {}
        img_path = os.path.join(arg.data_root, split_name, 'image_2', f'{line}.png')
        lidar_path = os.path.join(arg.data_root, split_name, 'velodyne', f'{line}.bin')
        calib_path = os.path.join(arg.data_root, split_name, 'calib', f'{line}.txt')
        assert os.path.exists(img_path) and os.path.exists(lidar_path) and os.path.exists(calib_path)
        calib_dict = read_calib(calib_path)
        lidar_points = read_points(lidar_path)
        img = cv.imread(img_path)
        h, w = img.shape[:2]
        cur_info_dict['image'] = {
            "shape_wh": (w, h),
            "path": img_path,
            "id": int(line)
        }
        cur_info_dict['calib'] = calib_dict
        reduced_lidar_points = remove_outside_points(lidar_points,
                                                     calib_dict.r0_rect,
                                                     calib_dict.velo_to_cam,
                                                     calib_dict.p2,
                                                     (w, h))
        reduced_lidar_points_file = os.path.join(reduced_points_save_dir, f'{line}.bin')
        write_points(reduced_lidar_points, reduced_lidar_points_file)
        if arg.is_train:
            label_path = os.path.join(arg.data_root, split_name, 'label_2', f'{line}.txt')
            annotation = read_label(label_path)
            annotation.judge_difficulty()
            indices, n_valid_bbox, bboxes_lidar, names = points_in_multi_boxes(reduced_lidar_points,
                                                                               calib_dict.r0_rect,
                                                                               calib_dict.velo_to_cam,
                                                                               annotation.dimensions,
                                                                               annotation.locations,
                                                                               annotation.rotations,
                                                                               annotation.names)
            is_valid = np.zeros(shape=(len(annotation.names),), dtype=bool)
            is_valid[:n_valid_bbox] = True
            num_of_points = np.zeros(shape=(len(annotation.names),), dtype=int)
            num_of_points[:n_valid_bbox] = indices.sum(0)
            annotation.is_valid = is_valid
            annotation.num_of_points = num_of_points
            annotation.valid_bboxes_in_lidar = bboxes_lidar
            cur_info_dict['lidar'] = {
                "annotation": annotation,
                "path": reduced_lidar_points_file
            }
            kitti_infos_dict[int(line)] = cur_info_dict
            if arg.save_db:
                for j in range(n_valid_bbox):
                    db_points = reduced_lidar_points[indices[:, j]]
                    db_points[:, :3] -= bboxes_lidar[j, :3]
                    db_points_saved_name = os.path.join(db_points_dir, f'{int(line)}_{names[j]}_{j}.bin')
                    write_points(db_points, db_points_saved_name)

                    db_info = {
                        "name": names[j],
                        "path": db_points_saved_name,
                        "box3d_lidar": bboxes_lidar[j],
                        "difficulty": annotation.difficulties[j],
                        "num_points_in_gt": len(db_points)
                    }
                    if names[j] not in kitti_db_info_train:
                        kitti_db_info_train[names[j]] = [db_info]
                    else:
                        kitti_db_info_train[names[j]].append(db_info)

    saved_path = os.path.join(arg.cache_dir, arg.split, f"{split_name}_infos.pkl")
    write_pickle(kitti_infos_dict, saved_path)
    if arg.save_db:
        db_save_path = os.path.join(db_dir, f"{split_name}_db.pkl")
        write_pickle(kitti_db_info_train, db_save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset infomation')
    parser.add_argument('--data_root', default='/home/lion/large_data/data/kitti/kitti_OD',
                        help='your data root for kitti')
    parser.add_argument('--cache_dir', default='/home/lion/large_data/data/cache',
                        help='the dir for the saved .pkl file')
    parser.add_argument('--split', default='train',
                        help='the prefix name for the saved .pkl file')
    parser.add_argument('--is_train', action="store_true", help="default is true")
    parser.add_argument('--save_db', action="store_true", help="default is true")
    args = parser.parse_args()
    prepare(args)
