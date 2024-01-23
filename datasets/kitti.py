import os
import torch
from torch.utils.data.dataset import Dataset
from utils.kitti import read_pickle, read_points
from utils.kitti import AnnotationKitti, CalibKitti


class Kitti(Dataset):
    CLASSES = {
        'Pedestrian': 0,
        'Cyclist': 1,
        'Car': 2
    }

    def __init__(self, pkl_path, transform, voxelizer):
        super().__init__()
        self.transform = transform
        self.voxelizer = voxelizer
        self.data_info = read_pickle(pkl_path)
        self.key_list = list(self.data_info.keys())

    def __getitem__(self, item):
        data_item = self.data_info[self.key_list[item]]
        image_info, lidar_info = data_item['image'], data_item['lidar']
        calib_info: CalibKitti = data_item['calib']
        annotation: AnnotationKitti = lidar_info['annotation'].remove_invalid()
        lidar_points = read_points(lidar_info['path'])
        lidar_points, annotation = self.transform(lidar_points, annotation)
        annotation.labels = [Kitti.CLASSES.get(item, -1) for item in annotation.names]
        voxels, cords, point_num_of_voxel = self.voxelizer(lidar_points)
        return torch.from_numpy(voxels).float(), \
            torch.from_numpy(cords), \
            torch.from_numpy(point_num_of_voxel), \
            annotation, calib_info

    def __len__(self):
        return len(self.key_list)

    @staticmethod
    def collect(batches):
        voxels = list()
        cords = list()
        point_num = list()
        annotations = list()
        calibs = list()
        i = 0
        for v, c, pn, an, calib in batches:
            voxels.append(v)
            bs = torch.ones(size=(len(c), 1), dtype=c.dtype) * i
            cords.append(torch.cat([bs, c], dim=-1))
            point_num.append(pn)
            annotations.append(an)
            calibs.append(calib)
            i += 1
        voxels = torch.cat(voxels, dim=0)
        cords = torch.cat(cords, dim=0)
        point_num = torch.cat(point_num, dim=0)
        return voxels, cords, point_num, annotations, calibs
