import numpy as np


def compute_precision_recall(contours_a, contours_b, theta):
    """
    Count the percentage of points in contours_b that are close to contours_a
    at a threshold theta.

    PR = sum_{z in Cb} delta[d(z,Ca)<theta] / |Cb|

    For precision,
    * contours_a: ground truth
    * contours_b: prediction
    """
    x = np.array(contours_a)
    y = np.array(contours_b)

    hits = 0
    for y_coord in y:
        d = np.square(x[:,0] - y_coord[0]) + np.square(x[:,1] - y_coord[1])
        hits += np.any(d < theta*theta)

    try:
        precision_recall = hits / len(y)
    except ZeroDivisionError:
        precision_recall = 0.

    return precision_recall, hits, len(y)


def compute_f1(precision, recall):
    """
    Compute F1 score.

    F1 = 2 * P * R / (P + R)
    """
    try:
        f1 = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        f1 = 0.
    return f1
