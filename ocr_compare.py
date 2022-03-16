import os
import argparse
from itertools import groupby
from collections import defaultdict
from typing import List, Dict, Optional, Any
import json
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from skimage.draw import polygon
import plotnine


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--annotation_csv", type=str, default="./polygon_ocr_346.csv")
    parser.add_argument("--groundtruth_csv", type=str, default="./groundtruth.csv")


    args = parser.parse_args()

    return args


def compute_iou(anno: Dict, groundtruth: Dict) -> float:
    an_points = np.maximum(np.array(anno["value"]["points"]) - 1, 0)
    gt_points = np.maximum(np.array(groundtruth["value"]["points"]) - 1, 0)
    #print(an_points, gt_points)
    an_c, an_r = an_points.T
    gt_c, gt_r = gt_points.T
    try:
        an_rr, an_cc = polygon(an_r, an_c)
        gt_rr, gt_cc = polygon(gt_r, gt_c)
    except TypeError:
        return 0.
    an_mask = np.zeros((100, 100), dtype=np.bool)
    gt_mask = np.zeros((100, 100), dtype=np.bool)
    an_mask[an_rr, an_cc] = True
    gt_mask[gt_rr, gt_cc] = True
    intersect = (an_mask & gt_mask).sum()
    iou = intersect / (an_rr.shape[0] + gt_rr.shape[0] - intersect)

    assert iou >= 0. and iou <= 1.

    return iou


if __name__ == "__main__":
    args = get_args()

    an_df = pd.read_csv(args.annotation_csv)
    gt_df = pd.read_csv(args.groundtruth_csv)

    an_df_dict = {}

    for i, row in an_df.iterrows():
        if row.completed_by_id not in an_df_dict:
            an_df_dict[row.completed_by_id] = {}
        if row.project_id not in an_df_dict[row.completed_by_id]:
            an_df_dict[row.completed_by_id][row.project_id] = {}
        an_df_dict[row.completed_by_id][row.project_id][row.task_id] = json.loads(row.result)

    gt_df_dict = {}

    for i, row in gt_df.iterrows():
        if row.ground_truth:
            if row.project_id not in gt_df_dict:
                gt_df_dict[row.project_id] = {}
            gt_df_dict[row.project_id][row.task_id] = {}
            #print(json.loads(row.result))
            gt_df_dict[row.project_id][row.task_id] = json.loads(row.result)
            #print(gt_df_dict[row.project_id][row.task_id]["result"])

    
    nums_true = {}
    nums_pred = {}
    nums_gt = {}
    ious_true = {}
    for user_id in an_df_dict:
        if user_id not in nums_true:
            nums_true[user_id] = {}
        if user_id not in nums_pred:
            nums_pred[user_id] = {}
        if user_id not in ious_true:
            ious_true[user_id] = {}
        #print("user_id: {}".format(user_id))
        for project_id in an_df_dict[user_id]:
            #print("project_id: {}".format(project_id))
            if project_id not in nums_true[user_id]:
                nums_true[user_id][project_id] = {}
            if project_id not in nums_pred[user_id]:
                nums_pred[user_id][project_id] = {}
            if project_id not in nums_gt:
                nums_gt[project_id] = {}
            if project_id not in ious_true[user_id]:
                ious_true[user_id][project_id] = {}
            for task_id in an_df_dict[user_id][project_id]:
                #print("task_id: {}".format(task_id))
                annotations = an_df_dict[user_id][project_id][task_id]
                groundtruths = gt_df_dict[project_id][task_id]
                #print("task_id: {}, project_id: {}, groundtruth: {}".format(task_id, project_id, groundtruths))
                ious_array = np.zeros(shape=(len(annotations), len(groundtruths)), dtype=np.float)
                for i, anno in enumerate(annotations):
                    #print("anno: {}".format(anno))
                    for j, groundtruth in enumerate(groundtruths):
                        #print("groundtruth: {}".format(groundtruth))
                        #print(anno["id"], groundtruth["id"])
                        #print("groundtruth: {}".format(groundtruth))
                        iou = compute_iou(anno=anno, groundtruth=groundtruth)
                        #print(iou)
                        ious_array[i, j] = iou


                user_index, gt_index = linear_sum_assignment(cost_matrix=ious_array, maximize=True)
                #print(ious_array, user_index, gt_index)
                assert len(ious_array.shape) == 2
                iou_matches = ious_array[user_index, gt_index]
                matches = (iou_matches > 0.4)
                ious_true[user_id][project_id][task_id] = iou_matches[matches]
                #print(iou_matches, matches, matches.sum(), ious_true[user_id][project_id][task_id])
                nums_true[user_id][project_id][task_id] = matches.sum()
                nums_pred[user_id][project_id][task_id] = ious_array.shape[0]
                if task_id not in nums_gt[project_id]:
                    nums_gt[project_id][task_id] = ious_array.shape[1]

    total_true = {}
    total_pred = {}
    total_gt = {}
    precision = {}
    recall = {}
    ious_total = {}
    ious_mean = {}

    for user_id in nums_pred:
        if user_id not in total_true:
            total_true[user_id] = {}
        if user_id not in total_pred:
            total_pred[user_id] = {}
        if user_id not in precision:
            precision[user_id] = {}
        if user_id not in recall:
            recall[user_id] = {}
        if user_id not in ious_total:
            ious_total[user_id] = {}
        if user_id not in ious_mean:
            ious_mean[user_id] = {}
        for project_id in nums_pred[user_id]:
            if project_id not in total_true[user_id]:
                total_true[user_id][project_id] = 0
            if project_id not in total_pred[user_id]:
                total_pred[user_id][project_id] = 0
            if project_id not in total_gt:
                total_gt[project_id] = 0
            if project_id not in ious_total[user_id]:
                ious_total[user_id][project_id] = []
            for task_id in nums_pred[user_id][project_id]:
                total_true[user_id][project_id] += nums_true[user_id][project_id][task_id]
                total_pred[user_id][project_id] += nums_pred[user_id][project_id][task_id]
                total_gt[project_id] += nums_gt[project_id][task_id]
                ious_total[user_id][project_id].extend(ious_true[user_id][project_id][task_id].tolist())

            precision[user_id][project_id] = total_true[user_id][project_id] / total_pred[user_id][project_id]
            recall[user_id][project_id] = total_true[user_id][project_id] / total_gt[project_id]
            if len(ious_total[user_id][project_id]) > 0:
                ious_mean[user_id][project_id] = np.mean(ious_total[user_id][project_id])
            else:
                ious_mean[user_id][project_id] = 0.

    changes_precision = {}
    changes_recall = {}
    changes_iou_means = {}

    for user_id in precision:
        for project_id in precision[user_id]:
            if project_id not in changes_precision:
                changes_precision[project_id] = {}
            if project_id not in changes_recall:
                changes_recall[project_id] = {}
            if project_id not in changes_iou_means:
                changes_iou_means[project_id] = {}
            changes_precision[project_id][user_id] = precision[user_id][project_id]
            changes_recall[project_id][user_id] = recall[user_id][project_id]
            changes_iou_means[project_id][user_id] = ious_mean[user_id][project_id]

    for project_id in changes_precision:
        precision_df = pd.DataFrame(list(changes_precision[project_id].items()), columns=["user_id", "precision"])
        recall_df = pd.DataFrame(list(changes_recall[project_id].items()), columns=["user_id", "recall"])
        iou_df = pd.DataFrame(list(changes_recall[project_id].items()), columns=["user_id", "iou"])

    precision_recall_df = precision_df.merge(recall_df, on="user_id", how="outer")

    precision_recall_iou_df = precision_recall_df.merge(iou_df, on="user_id", how="outer")

    plotnine.ggplot(data=precision_recall_iou_df) + plotnine.geom_histogram(mapping=plotnine.aes(x="precision"))

    plotnine.ggplot(data=precision_recall_iou_df) + plotnine.geom_histogram(mapping=plotnine.aes(x="recall"))

    plotnine.ggplot(data=precision_recall_iou_df) + plotnine.geom_histogram(mapping=plotnine.aes(x="iou"))

    plotnine.ggplot(data=precision_recall_iou_df, mapping=plotnine.aes(x="user_id", y="precision")) + plotnine.geom_bar(stat="identity")

    plotnine.ggplot(data=precision_recall_iou_df, mapping=plotnine.aes(x="user_id", y="recall")) + plotnine.geom_bar(stat="identity")

    plotnine.ggplot(data=precision_recall_iou_df, mapping=plotnine.aes(x="user_id", y="iou")) + plotnine.geom_bar(stat="identity")