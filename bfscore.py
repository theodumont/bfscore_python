"""
This is a fork of an open-source implementation of BF-score in Python
(https://github.com/minar09/bfscore_python).

References
----------
* Csurka, G., D. Larlus, and F. Perronnin. "What is a good evaluation measure
for semantic segmentation?" Proceedings of the British Machine Vision
Conference, 2013, pp. 32.1-32.11.
* crosscheck: https://www.mathworks.com/help/images/ref/bfscore.html
* https://github.com/minar09/bfscore_python
"""

import cv2
import numpy as np
import math

from metric_bfscore import compute_precision_recall, compute_f1

major = cv2.__version__.split('.')[0]     # Get opencv version
bDebug = False


""" computes the BF (Boundary F1) contour matching score between the predicted and GT segmentation """


def bfscore(gtfile, prfile, theta=2):

    # print(gtfile)
    gt__ = cv2.imread(gtfile)    # Read GT segmentation
    # for row in gt__:
    #     print(np.unique(row))
    # print(np.unique(gt__))
    print(gt__.shape)
    gt_ = cv2.cvtColor(gt__, cv2.COLOR_BGR2GRAY)    # Convert color space
    print(gt_.shape)

    import sys
    sys.exit(1)

    pr_ = cv2.imread(prfile)    # Read predicted segmentation
    pr_ = cv2.cvtColor(pr_, cv2.COLOR_BGR2GRAY)    # Convert color space

    classes_gt = np.unique(gt_)    # Get GT classes
    classes_pr = np.unique(pr_)    # Get predicted classes

    # Check classes from GT and prediction
    if not np.array_equiv(classes_gt, classes_pr):
        print('Classes are not same! GT:', classes_gt, 'Pred:', classes_pr)

        classes = np.concatenate((classes_gt, classes_pr))
        classes = np.unique(classes)
        classes = np.sort(classes)
        print('Merged classes :', classes)
    else:
        print('Classes :', classes_gt)
        classes = classes_gt    # Get matched classes

    m = np.max(classes)    # Get max of classes (number of classes)
    # Define bfscore variable (initialized with zeros)
    bfscores = np.zeros((m+1), dtype=float)
    areas_gt = np.zeros((m + 1), dtype=float)

    for i in range(m+1):
        bfscores[i] = np.nan
        areas_gt[i] = np.nan

    for target_class in classes:    # Iterate over classes

        if target_class == 0:     # Skip background
            continue

        print(">>> Calculate for class:", target_class)

        gt = gt_.copy()
        gt[gt != target_class] = 0
        # print(gt.shape)

        # contours는 point의 list형태.
        if major == '3':    # For opencv version 3.x
            _, contours, _ = cv2.findContours(
                gt, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)    # Find contours of the shape
        else:    # For other opencv versions
            contours, _ = cv2.findContours(
                gt, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)  # Find contours of the shape

        # contours 는 list of numpy arrays
        contours_gt = []
        for i in range(len(contours)):
            for j in range(len(contours[i])):
                contours_gt.append(contours[i][j][0].tolist())
        if bDebug:
            print('contours_gt')
            print(contours_gt)

        # Get contour area of GT
        if contours_gt:
            area = cv2.contourArea(np.array(contours_gt))
            areas_gt[target_class] = area

        print("\tArea:", areas_gt[target_class])

        # Draw GT contours
        img = np.zeros_like(gt__)
        # print(img.shape)
        img[gt == target_class, 0] = 128  # Blue
        img = cv2.drawContours(img, contours, -1, (255, 0, 0), 1)

        pr = pr_.copy()
        pr[pr != target_class] = 0
        # print(pr.shape)

        # contours는 point의 list형태.
        if major == '3':    # For opencv version 3.x
            _, contours, _ = cv2.findContours(
                pr, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        else:    # For other opencv versions
            contours, _ = cv2.findContours(
                pr, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        # contours 는 list of numpy arrays
        contours_pr = []
        for i in range(len(contours)):
            for j in range(len(contours[i])):
                contours_pr.append(contours[i][j][0].tolist())

        if bDebug:
            print('contours_pr')
            print(contours_pr)

        # Draw predicted contours
        img[pr == target_class, 2] = 128  # Red
        img = cv2.drawContours(img, contours, -1, (0, 0, 255), 1)

        # 3. calculate
        precision, numerator, denominator = calc_precision_recall(
            contours_gt, contours_pr, theta)    # Precision
        print("\tprecision:", denominator, numerator)

        recall, numerator, denominator = calc_precision_recall(
            contours_pr, contours_gt, theta)    # Recall
        print("\trecall:", denominator, numerator)

        f1 = compute_f1(precision, recall)

        print("\tf1:", f1)
        bfscores[target_class] = f1

        # cv2.imshow('image', img)
        # cv2.waitKey(1000)

    cv2.destroyAllWindows()

    # return bfscores[1:], np.sum(bfscores[1:])/len(classes[1:])    # Return bfscores, except for background, and per image score
    return bfscores[1:], areas_gt[1:]    # Return bfscores, except for background


if __name__ == "__main__":

    sample_gt = 'data/gt_1.png'
    # sample_gt = 'data/gt_0.png'

    sample_pred = 'data/crf_1.png'
    # sample_pred = 'data/pred_0.png'

    score, areas_gt = bfscore(sample_gt, sample_pred, 2)    # Same classes
    # score, areas_gt = bfscore(sample_gt, sample_pred, 2)    # Different classes

    # gt_shape = cv2.imread('data/gt_1.png').shape
    # print("Total area:", gt_shape[0] * gt_shape[1])

    total_area = np.nansum(areas_gt)
    print("GT area (except background):", total_area)
    fw_bfscore = []
    for each in zip(score, areas_gt):
        if math.isnan(each[0]) or math.isnan(each[1]):
            fw_bfscore.append(math.nan)
        else:
            fw_bfscore.append(each[0] * each[1])
    print(fw_bfscore)

    print("\n>>>>BFscore:\n")
    print("BFSCORE:", score)
    print("Per image BFscore:", np.nanmean(score))

    print("\n>>>>Weighted BFscore:\n")
    print("Weighted-BFSCORE:", fw_bfscore)
    print("Per image Weighted-BFscore:", np.nansum(fw_bfscore)/total_area)
