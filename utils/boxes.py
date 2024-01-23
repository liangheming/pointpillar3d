import cv2 as cv
import numpy as np
import numba

CLASS_NAMES = ['Pedestrian', 'Cyclist', 'Car']


def boxes_rotate_iou(boxes1, boxes2):
    """
    :param boxes1: x y z x_size y_size z_size yaw
    :param boxes2: x y z x_size y_size z_size yaw
    :return:
    """
    iou_bev = np.zeros(shape=(len(boxes1), len(boxes2)))
    # iou_3d = np.zeros(shape=(len(boxes1), len(boxes2)))

    area1 = boxes1[:, 3] * boxes1[:, 4]
    area2 = boxes2[:, 3] * boxes2[:, 4]

    for i in range(len(boxes1)):
        r1 = ((boxes1[i][0], boxes1[i][1]), (boxes1[i][3], boxes1[i][4]), np.rad2deg(boxes1[i][-1]))
        for j in range(len(boxes2)):
            r2 = ((boxes2[j][0], boxes2[j][1]), (boxes2[j][3], boxes2[j][4]), np.rad2deg(boxes2[j][-1]))
            int_pts = cv.rotatedRectangleIntersection(r1, r2)[1]
            if int_pts is not None:
                order_pts = cv.convexHull(int_pts, returnPoints=True)
                int_area = cv.contourArea(order_pts)
                inter = int_area * 1.0 / (area1[i] + area2[j] - int_area + 1e-8)
                iou_bev[i, j] = inter
    return iou_bev


class SimpleMap(object):
    def __init__(self, class_names):
        self.class_names = class_names
        self.class_list = [[] for _ in range(len(class_names))]

    def __call__(self, predicts, gts):
        """
        :param predict: x y z x_size y_size z_size score class
        :param gts: x y z x_size y_size z_size class
        :return:
        """
        if predicts is None:
            predicts = np.zeros(shape=(0, 9))
        if gts is None:
            gts = np.zeros(shape=(0, 8))
        for i in range(len(self.class_names)):
            self.class_list[i].append((
                predicts[predicts[:, -1] == i, :-1],
                gts[gts[:, -1] == i, :-1]
            ))

    def compute(self, ious):
        for i in range(len(self.class_names)):
            gt_num = 0.
            score_and_fp = list()
            for predicts, gts in self.class_list[i]:
                if predicts.shape[0] == 0 and gts.shape[0] == 0:
                    continue
                gt_num += gts.shape[0]
                if predicts.shape[0] == 0:
                    continue
                if gts.shape[0] == 0:
                    score_and_fp.append(np.concatenate([predicts[:, -1:], np.zeros(shape=(len(predicts), 1))], axis=-1))
                    continue
                overlaps = boxes_rotate_iou(predicts[:, :7], gts[:, :7])
                assigned = np.zeros((predicts.shape[0],), dtype=np.bool_)
                for j in range(gts.shape[0]):
                    match_id, match_score = -1, -1
                    for k in range(predicts.shape[0]):
                        if not assigned[k] and overlaps[k, j] > ious[i] and predicts[k, -1] > match_score:
                            match_id = k
                            match_score = predicts[k, -1]
                    if match_id != -1:
                        assigned[match_id] = True
                score_and_fp.append(np.concatenate([predicts[:, -1:], assigned[:, None].astype(predicts.dtype)], axis=-1))
            score_and_fp = np.concatenate(score_and_fp, axis=0)
            sort_idx = score_and_fp[:, 0].argsort()
            fps = score_and_fp[sort_idx, -1]
            tp_cusum = np.cumsum(fps)
            recall = tp_cusum / gt_num
            precision = tp_cusum / (np.array(range(score_and_fp.shape[0])) + 1)
            last_max_precision = precision[-1]
            for j in range(score_and_fp.shape[0] - 1, -1, -1):
                if precision[j] >= last_max_precision:
                    last_max_precision = precision[j]
                precision[j] = last_max_precision
            delta = recall[1:] - recall[:-1]
            mid_precision = (precision[1:] + precision[:-1]) / 2
            map_val = (delta * mid_precision).sum()
            print(map_val)

    def get_ap(self):
        pass


def main():
    from utils.kitti import read_pickle
    simple_map = SimpleMap(class_names=CLASS_NAMES)
    predicts = read_pickle("/home/lion/PycharmProjects/pillar/predict.pkl")
    gts = read_pickle("/home/lion/PycharmProjects/pillar/gt.pkl")
    keys = gts.keys()
    for k in keys:
        p = predicts[k]
        g = gts[k]
        g_mask = np.bitwise_and(np.array([name in CLASS_NAMES for name in g['name']]), g['difficulty'] >= 0)
        p_mask = (p["bbox"][:, 3] - p["bbox"][:, 1]) > 25
        g_label = np.array([-1.0 if name not in CLASS_NAMES else float(CLASS_NAMES.index(name)) for name in g['name']])
        p_label = np.array([-1.0 if name not in CLASS_NAMES else float(CLASS_NAMES.index(name)) for name in p['name']])
        g_inp = np.concatenate([
            g['location'][:, [0, 2, 1]], g['dimensions'][:, [0, 2, 1]], g['rotation_y'][:, None], g_label[:, None]
        ], axis=-1)
        p_inp = np.concatenate([
            p['location'][:, [0, 2, 1]], p['dimensions'][:, [0, 2, 1]], p['rotation_y'][:, None], p['score'][:, None], p_label[:, None]
        ], axis=-1)
        p_inp = p_inp[p_mask]
        g_inp = g_inp[g_mask]
        simple_map(p_inp, g_inp)
    simple_map.compute(ious=[0.5, 0.5, 0.7])


if __name__ == '__main__':
    main()
