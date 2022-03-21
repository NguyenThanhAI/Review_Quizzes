import argparse
import os
from typing import List, Dict, Tuple
from itertools import groupby, chain
import json
import numpy as np
import pandas as pd
from skimage.draw import polygon
import krippendorff
from utils import MulticlassDawidSkeneEM, majority_vote, post_process



def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--annotation_csv", type=str, default="./classification_356.csv")

    args = parser.parse_args()

    return args


def preprocess_classification_annotations_for_estimation(csv_path: str) -> Tuple[List[Dict], List]:
    df = pd.read_csv(csv_path)

    new_df = df[["task_id", "completed_by_id", "result"]]

    new_df = new_df.to_records(index=False).tolist()

    new_df = list(map(lambda x: (x[0], x[1], json.loads(x[2])), new_df))

    new_df = list(filter(lambda x: x[2] != [], new_df))

    new_df = list(map(lambda x: (x[0], x[1], x[2][0]["value"]["choices"][0]), new_df))

    task_list = list(map(lambda x: x[0], new_df))
    categories = list(set(list(map(lambda x: x[2], new_df))))

    #print(len(task_list), len(list(set(task_list))), categories)

    new_df.sort(key=lambda x: x[0])

    dset_to_annotations = []
    for task_id, records in groupby(new_df, key=lambda x: x[0]):
        #print(len(list(records)))
        records = list(records)
        annotations = []
        for rec in records:
            annotations.append({"workerId": rec[1], "annotationData": {"content": rec[2]}})
        dset_annotations = {"datasetObjectId": task_id, "annotations": annotations}
        dset_to_annotations.append(dset_annotations)
    return dset_to_annotations, categories


def aggregate(csv_path: str) -> List[Dict]:
    dset_to_annotations, categories = preprocess_classification_annotations_for_estimation(csv_path=csv_path)
    dawid_skene = MulticlassDawidSkeneEM("classification")

    responses = dawid_skene.update(annotation_payload=dset_to_annotations, label_categories=categories,
                                        label_attribute_name="categories")
    responses = post_process(responses)
    return responses


def review_classification(csv_path: str) -> float:
    df = pd.read_csv(csv_path)
    df_list = df.values.tolist()
    df_list.sort(key=lambda x: x[6])

    labels = []
    for row in df_list:
        for anno in json.loads(row[2]):
            try:
                labels.append(anno["value"]["choices"][0])
            except KeyError:
                print("Task id: {}, id: {}".format(row[6], row[1]))

    labels = list(set(labels))
    labels.sort()
    labels_to_index = dict(zip(labels, range(1, len(labels) + 1)))

    uids = []

    for row in df_list:
        uids.append(row[3])

    uids = list(set(uids))
    uids.sort()

    annotations = {}
    for task_id, results in groupby(df_list, key=lambda x: x[6]):
        result = dict(list(map(lambda x: (x[3], json.loads(x[2])), list(results))))
        annotations[task_id] = result
    
    total_annotations = []

    for task_id in annotations:
        array_annotators = []

        for uid in uids:
            if uid in annotations[task_id]:
                array_annotators.append(labels_to_index[annotations[task_id][uid][0]["value"]["choices"][0]])
            else:
                array_annotators.append(0)

        total_annotations.append(array_annotators)

    alpha = krippendorff.alpha(reliability_data=total_annotations, value_domain=[0] + list(labels_to_index.values()))

    return alpha



if __name__ == "__main__":
    args = get_args()

    responses = aggregate(csv_path=args.annotation_csv)
    print(responses)

    alpha = review_classification(csv_path=args.annotation_csv)

    print(alpha)
