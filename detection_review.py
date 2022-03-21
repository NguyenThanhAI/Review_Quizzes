import os
import argparse
import time
from typing import List, Dict, Tuple
from itertools import groupby, chain
import json
import numpy as np
import pandas as pd
import krippendorff


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--annotation_csv", type=str, default="./rectangles_342.csv")

    args = parser.parse_args()

    return args


def compute_agreement(csv_path: str) -> float:
    df = pd.read_csv(csv_path)
    df_list = df.values.tolist()
    df_list.sort(key=lambda x: x[6])

    labels = []

    #print(1)

    for row in df_list:
                for anno in json.loads(row[2]):
                    try:
                        labels.append(anno["value"]["rectanglelabels"][0])
                    except KeyError:
                        print("Task id: {}, id: {}".format(row[6], row[1]))
    #print(2)
    labels = list(set(labels))
    labels.sort()
    labels_to_index = dict(zip(labels, range(1, len(labels) + 1)))
    #print(3)
    annotations = {}
    image_size = {}
    for task_id, results in groupby(df_list, key=lambda x: x[6]):
        result = dict(list(map(lambda x: (x[3], json.loads(x[2])), list(results))))
        annotations[task_id] = result
        image_size[task_id] = (list(result.values())[0][0]["original_height"], list(result.values())[0][0]["original_width"])
    #print(4)
    agreement = 0
    #print(labels_to_index.values())
    for task_id in annotations:
        array_annotators = []
        height = image_size[task_id][0]
        width = image_size[task_id][1]
        for uid in annotations[task_id]:
            array_anno = np.zeros(shape=(height, width))
            for anno in annotations[task_id][uid]:
                x, y, w, h = anno["value"]["x"], anno["value"]["y"], anno["value"]["width"], anno["value"]["height"]
                try:
                    x, y, w, h = int((x / 100) * width), int((y / 100) * width), int((w / 100) * width), int((h / 100) * height)
                except:
                    #print(file, task_id, uid, anno)
                    continue
                try:
                    l = labels_to_index[anno["value"]["rectanglelabels"][0]]
                except KeyError:
                    l = 1
                #print(l)
                array_anno[y: y + h, x: x + w] = l
            array_anno = np.reshape(array_anno, [-1])
            #array_anno = np.where(array_anno == 0, np.nan, array_anno)
            array_anno = array_anno.tolist()
            array_annotators.append(array_anno)

        alpha = krippendorff.alpha(reliability_data=array_annotators, value_domain=[0] + list(labels_to_index.values()))

        agreement += alpha
    #print(5)
    agreement = agreement / len(annotations.keys())

    return agreement


if __name__ == "__main__":
    args = get_args()
    start = time.time()
    agreement = compute_agreement(csv_path=args.annotation_csv)
    end = time.time()
    print(agreement, end - start)
