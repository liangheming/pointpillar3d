import numpy as np
import cv2 as cv


def nms_rotate_cpu(boxes, scores, iou_threshold, max_output_size):
    keep = []

    order = scores.argsort()[::-1]
    num = boxes.shape[0]

    suppressed = np.zeros((num), dtype=np.int32)

    for _i in range(num):
        if len(keep) >= max_output_size:
            break

        i = order[_i]
        if suppressed[i] == 1:
            continue
        keep.append(i)
        r1 = ((boxes[i, 0], boxes[i, 1]), (boxes[i, 2], boxes[i, 3]), boxes[i, 4])
        area_r1 = boxes[i, 2] * boxes[i, 3]
        for _j in range(_i + 1, num):
            j = order[_j]
            if suppressed[i] == 1:
                continue
            r2 = ((boxes[j, 0], boxes[j, 1]), (boxes[j, 2], boxes[j, 3]), boxes[j, 4])
            area_r2 = boxes[j, 2] * boxes[j, 3]
            inter = 0.0

            int_pts = cv.rotatedRectangleIntersection(r1, r2)[1]
            if int_pts is not None:
                order_pts = cv.convexHull(int_pts, returnPoints=True)

                int_area = cv.contourArea(order_pts)

                inter = int_area * 1.0 / (area_r1 + area_r2 - int_area + 1e-8)

            if inter >= iou_threshold:
                suppressed[j] = 1

    return np.array(keep, np.int64)


def projection2crt(proj):
    """
    :param proj: 投影矩阵
    :return: crt: c is upper triangular
    """
    cr = proj[0:3, 0:3]
    ct = proj[0:3, 3]
    cr_inv = np.linalg.inv(cr)
    r_inv, c_inv = np.linalg.qr(cr_inv)
    c = np.linalg.inv(c_inv)
    r = np.linalg.inv(r_inv)
    t = c_inv @ ct
    return c, r, t


def get_frustum(bbox, c, near=0.001, far=100.0):
    """
    :param bbox:
    :param c:
    :param near:
    :param far:
    :return:
    """
    fku = c[0, 0]
    fkv = c[1, 1]
    u0v0 = c[0:2, 2]
    # (8, 1)
    z_points = np.array(
        [near] * 4 + [far] * 4, dtype=c.dtype)[:, np.newaxis]
    x1, y1, x2, y2 = bbox
    #     x1,y1----x2,y1
    #      |         |
    #     x1,y2----x2,y2
    box_corners = np.array(
        [[x1, y1], [x1, y2], [x2, y2], [x2, y1]],
        dtype=c.dtype)
    near_box_corners = (box_corners - u0v0) / np.array(
        [fku / near, fkv / near], dtype=c.dtype)
    far_box_corners = (box_corners - u0v0) / np.array(
        [fku / far, fkv / far], dtype=c.dtype)
    ret_xy = np.concatenate([near_box_corners, far_box_corners],
                            axis=0)  # [8, 2]
    ret_xyz = np.concatenate([ret_xy, z_points], axis=1)
    return ret_xyz
