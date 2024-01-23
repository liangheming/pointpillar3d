import os
from pytorch_lightning import LightningDataModule
from utils.kitti import read_pickle
from datasets.kitti import Kitti
from datasets.transform import Compose, DBSample, ObjectNoise, RandFlip, ObjectRangeFilter, PointRangeFilter, PointShuffle, GlobalAffine
from datasets.voxelize import Voxelize
from torch.utils.data.dataloader import DataLoader


class KittiWrapper(LightningDataModule):
    def __init__(self, hparams):
        super().__init__()
        db_info = read_pickle("/home/lion/large_data/data/cache/train/db/training_db.pkl")
        db_sample = DBSample(db=db_info, sample_dict={"Car": 15, "Pedestrian": 10, "Cyclist": 10})
        db_sample.filter_db({"Car": 5, "Pedestrian": 10, "Cyclist": 10})

        self.t_voxelizer = Voxelize(voxel_size=[0.16, 0.16, 4],
                                    point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1],
                                    max_num_points=32,
                                    max_voxels=16000)
        self.v_voxelizer = Voxelize(
            voxel_size=[0.16, 0.16, 4],
            point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1],
            max_num_points=32,
            max_voxels=40000)
        self.t_transform = Compose(
            transforms=[
                db_sample,
                ObjectNoise(100, [0.25, 0.25, 0.25], [-0.15707963267, 0.15707963267]),
                RandFlip(0.5),
                GlobalAffine(rotation=[-0.78539816, 0.78539816], scale=[0.95, 1.05], translation=[0, 0, 0]),
                PointRangeFilter([0, -39.68, -3, 69.12, 39.68, 1]),
                ObjectRangeFilter([0, -39.68, -3, 69.12, 39.68, 1]),
                PointShuffle()
            ]
        )
        self.v_transform = PointRangeFilter([0, -39.68, -3, 69.12, 39.68, 1])
        self.t_datasets = None
        self.v_datasets = None

    def setup(self, stage: str) -> None:
        self.t_datasets = Kitti(pkl_path="/home/lion/large_data/data/cache/train/training_infos.pkl", transform=self.t_transform, voxelizer=self.t_voxelizer)
        self.v_datasets = Kitti(pkl_path="/home/lion/large_data/data/cache/val/training_infos.pkl", transform=self.v_transform, voxelizer=self.v_voxelizer)

    def train_dataloader(self):
        return DataLoader(self.t_datasets, shuffle=True, pin_memory=True, num_workers=4, batch_size=6, collate_fn=self.t_datasets.collect, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.t_datasets, shuffle=False, pin_memory=True, num_workers=4, batch_size=6, collate_fn=self.t_datasets.collect)
