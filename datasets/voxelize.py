import os
import numba
import numpy as np


@numba.jit(nopython=True)
def voxelize(points, voxels, cords, num_point_per_voxel,
             voxel_range, voxel_size, grid, max_voxel, max_point_num):
    num_points_per_voxel_cords = np.zeros(shape=(grid[0], grid[1], grid[2]), dtype=np.int32)
    cord2idx = -1 * np.ones(shape=(grid[0], grid[1], grid[2]), dtype=np.int32)
    voxel_num = 0
    for i in range(len(points)):
        x, y, z = points[i][:3]
        x_idx = int((x - voxel_range[0]) / voxel_size[0])
        y_idx = int((y - voxel_range[1]) / voxel_size[1])
        z_idx = int((z - voxel_range[2]) / voxel_size[2])
        if x_idx < 0 or x_idx >= grid[0] or y_idx < 0 or y_idx >= grid[1] or z_idx < 0 or z_idx >= grid[2]:
            continue
        voxel_idx = cord2idx[x_idx, y_idx, z_idx]
        point_idx_in_voxel = num_points_per_voxel_cords[x_idx, y_idx, z_idx]
        flag = voxel_idx == -1
        if flag:
            voxel_idx = voxel_num
            cord2idx[x_idx, y_idx, z_idx] = voxel_idx
            voxel_num += 1
            cords[voxel_idx][0] = x_idx
            cords[voxel_idx][1] = y_idx
            cords[voxel_idx][2] = z_idx
        if point_idx_in_voxel >= max_point_num:
            continue
        voxels[voxel_idx][point_idx_in_voxel] = points[i]
        num_points_per_voxel_cords[x_idx, y_idx, z_idx] += 1
        if voxel_num >= max_voxel:
            break
    for i in range(voxel_num):
        voxel_cord = cords[i]
        num_point_per_voxel[i] = num_points_per_voxel_cords[voxel_cord[0], voxel_cord[1], voxel_cord[2]]
    return voxel_num


class Voxelize(object):
    def __init__(self, voxel_size, point_cloud_range, max_num_points, max_voxels):
        self.voxel_size = np.array(voxel_size, dtype=np.float32)
        self.point_cloud_range = np.array(point_cloud_range, dtype=np.float32)
        self.max_num_points = max_num_points
        self.max_voxels = max_voxels
        self.grid = np.round((self.point_cloud_range[3:] - self.point_cloud_range[:3]) / self.voxel_size).astype(
            np.int32)

    def __call__(self, points):
        """
        :param points:
        :return: voxel (N,M,4), cords: (N ,3)
        """
        voxels = np.zeros(shape=(self.max_voxels, self.max_num_points, points.shape[-1]), dtype=np.float32)
        cords = -np.ones(shape=(self.max_voxels, 3), dtype=np.int32)
        num_points_per_voxel = np.zeros(shape=(self.max_voxels,), dtype=np.int32)
        voxel_num = voxelize(points,
                             voxels,
                             cords,
                             num_points_per_voxel,
                             self.point_cloud_range,
                             self.voxel_size,
                             self.grid,
                             self.max_voxels,
                             self.max_num_points)
        return voxels[:voxel_num], cords[:voxel_num], num_points_per_voxel[:voxel_num]
