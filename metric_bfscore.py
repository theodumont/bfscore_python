""" For precision, contours_a==GT & contours_b==Prediction
    For recall, contours_a==Prediction & contours_b==GT """

import numpy as np

def calc_precision_recall_base(contours_a, contours_b, theta):
    x = contours_a
    y = contours_b

    xx = np.array(x)
    hits = []
    for yrec in y:
        d = np.square(xx[:,0] - yrec[0]) + np.square(xx[:,1] - yrec[1])
        hits.append(np.any(d < theta*theta))
    top_count = np.sum(hits)

    try:
        precision_recall = top_count / len(y)
    except ZeroDivisionError:
        precision_recall = 0

    return precision_recall, top_count, len(y)


def calc_precision_recall(contours_a, contours_b, theta):
    base_precision_recall, base_hits, base_len = calc_precision_recall_base(contours_a, contours_b, theta)



    x = np.array(contours_a)
    y = np.array(contours_b)

    hits = 0
    for y_coord in y:
        d = np.square(x[:,0] - y_coord[0]) + np.square(x[:,1] - y_coord[1])
        hits += np.any(d < theta*theta)

    try:
        precision_recall = hits / len(y)
    except ZeroDivisionError:
        precision_recall = 0




    assert precision_recall == base_precision_recall
    assert hits == base_hits
    assert len(y) == base_len

    return precision_recall, hits, len(y)



if __name__ == "__main__":
