from datasets.kitti import Kitti
from utils.kitti import read_pickle
from datasets.voxelize import Voxelize
from datasets.transform import DBSample, ObjectNoise, RandFlip, GlobalAffine, ObjectRangeFilter, PointRangeFilter, \
    Compose, PointShuffle
from torch.utils.data.dataloader import DataLoader
from models.pointpillars import PointPillar

if __name__ == '__main__':
    db_info = read_pickle("/home/lion/large_data/data/cache/train/db/training_db.pkl")
    db_sample = DBSample(db=db_info, sample_dict={"Car": 15, "Pedestrian": 10, "Cyclist": 10})
    db_sample.filter_db({"Car": 5, "Pedestrian": 10, "Cyclist": 10})
    voxelizer = Voxelize(voxel_size=[0.16, 0.16, 4],
                         point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1],
                         max_num_points=32,
                         max_voxels=16000)
    transform = Compose(
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
    kitti_data = Kitti(pkl_path="/home/lion/large_data/data/cache/train/training_infos.pkl",
                       transform=transform,
                       voxelizer=voxelizer)

    loader = DataLoader(dataset=kitti_data, batch_size=4, num_workers=4, pin_memory=True, shuffle=True, drop_last=True,
                        collate_fn=kitti_data.collect)

    pillar = PointPillar(3).cuda()
    for v, c, pn, an, ca in loader:
        ret = pillar(v.cuda(), c.cuda(), pn.cuda(), an)
        break
