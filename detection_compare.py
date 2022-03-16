import os
import argparse
from itertools import groupby
from collections import defaultdict
from typing import List, Dict, Optional, Any
import json
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
import plotnine


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--annotation_csv", type=str, default="./rectangles_342.csv")
    parser.add_argument("--groundtruth_csv", type=str, default="./groundtruth.csv")


    args = parser.parse_args()

    return args


def compute_iou(anno: Dict, groundtruth: Dict) -> float:
    #print(anno["value"]["x"], anno["value"]["y"], anno["value"]["width"], anno["value"]["height"], groundtruth["value"]["x"], groundtruth["value"]["y"], groundtruth["value"]["width"], groundtruth["value"]["height"])
    try:
        x_left = max(anno["value"]["x"], groundtruth["value"]["x"])
        x_right = min(anno["value"]["x"] + anno["value"]["width"], groundtruth["value"]["x"] + groundtruth["value"]["width"])
        y_top = max(anno["value"]["y"], groundtruth["value"]["y"])
        y_bottom = min(anno["value"]["y"] + anno["value"]["height"], groundtruth["value"]["y"] + groundtruth["value"]["height"])
    except TypeError:
        return 0.
    if x_right < x_left or y_bottom < y_top:
        return 0.

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    anno_area = anno["value"]["width"] * anno["value"]["height"]
    groundtruth_area = groundtruth["value"]["width"] * groundtruth["value"]["height"]
    #print(anno_area, groundtruth_area)
    iou = intersection_area/float(anno_area + groundtruth_area - intersection_area)
    #print(intersection_area, anno_area, groundtruth_area, iou)
    assert iou >= 0. and iou <=1

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

    
    gt_df_dict = defaultdict(dict)

    for i, row in gt_df.iterrows():
        gt_df_dict[row.project_id][row.task_id] = row.drop(["project_id", "task_id", "id"]).to_dict()
        gt_df_dict[row.project_id][row.task_id]["result"] = json.loads(gt_df_dict[row.project_id][row.task_id]["result"])

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
        for project_id in an_df_dict[user_id]:
            if project_id not in nums_true[user_id]:
                nums_true[user_id][project_id] = {}
            if project_id not in nums_pred[user_id]:
                nums_pred[user_id][project_id] = {}
            if project_id not in nums_gt:
                nums_gt[project_id] = {}
            if project_id not in ious_true[user_id]:
                ious_true[user_id][project_id] = {}
            for task_id in an_df_dict[user_id][project_id]:
                annotations = an_df_dict[user_id][project_id][task_id]
                groundtruths = gt_df_dict[project_id][task_id]["result"]
                ious_array = np.zeros(shape=(len(annotations), len(groundtruths)), dtype=np.float)
                labels_array = np.zeros(shape=(len(annotations), len(groundtruths)), dtype=np.bool)
                for i, anno in enumerate(annotations):
                    #print("anno: {}".format(anno))
                    for j, groundtruth in enumerate(groundtruths):
                        #print(anno["id"], groundtruth["id"])
                        #print("groundtruth: {}".format(groundtruth))
                        iou = compute_iou(anno=anno, groundtruth=groundtruth)
                        ious_array[i, j] = iou
                        try:
                            labels_array[i, j] = anno["value"]["rectanglelabels"][0] == groundtruth["value"]["rectanglelabels"][0]
                        except KeyError:
                            pass

                        
                user_index, gt_index = linear_sum_assignment(cost_matrix=ious_array, maximize=True)
                #print(ious_array, labels_array, user_index, gt_index)
                assert ious_array.shape[0] == labels_array.shape[0] and ious_array.shape[1] == labels_array.shape[1] and len(ious_array.shape) == 2 and len(labels_array.shape) == 2
                iou_matches = ious_array[user_index, gt_index]
                label_matches = labels_array[user_index, gt_index]
                matches = (iou_matches > 0.4) & label_matches
                ious_true[user_id][project_id][task_id] = iou_matches[matches]
                #print(iou_matches, label_matches, matches, matches.sum(), ious_true[user_id][project_id][task_id])
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