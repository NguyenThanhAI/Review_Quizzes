import os
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pickle


class DataLayer(object):

    def __init__(self):
        self.worker_params = defaultdict(dict)
        self.label_params = defaultdict(dict)

    def put_label_information_s3(self, label_data, dataset_object_id, labeling_job_arn):
        self.label_params[labeling_job_arn][dataset_object_id] = pickle.dumps(label_data)

    def get_label_information_s3(self, dataset_object_id, labeling_job_arn):
        label_data = self.label_params.get(labeling_job_arn, {}).get(dataset_object_id, None)
        if label_data:
            label_data = pickle.loads(label_data)
        return label_data

    def put_worker_information_s3(self, worker_data, worker_id, labeling_job_arn):
        self.worker_params[labeling_job_arn][worker_id] = pickle.dumps(worker_data)

    def get_worker_information_s3(self, worker_id, labeling_job_arn):
        worker_data = self.worker_params.get(labeling_job_arn, {}).get(worker_id, None)
        if worker_data:
            worker_data = pickle.loads(worker_data)
        return worker_data


class MulticlassDawidSkeneEM(object):

    def __init__(
        self,
        labeling_job_arn,
        output_config=None,
        role_arn=None,
        kms_key_id=None,
        identifier="Testing",
    ):
        self.labeling_job_arn = labeling_job_arn
        self.dataset_object_ids = set()
        self.worker_ids = set()
        self.l_ij = defaultdict(dict)
        self.p_prior = None
        self.max_epochs = 20
        self.min_relative_diff = 1e-8
        self.identifier = identifier
        self.data_layer = DataLayer()

    def update(self, annotation_payload, label_categories, label_attribute_name, is_text=False):
        all_worker_prior = 0.7
        p, c_mtrx = self.get_or_initialize_parameters(
            annotation_payload, label_categories, all_worker_prior
        )
        log_likelihood = None
        for epoch in range(self.max_epochs):
            p, p_non_normalized = self.expectation_step(self.l_ij, p, c_mtrx, self.n_classes)
            c_mtrx, worker_priors = self.maximization_step(
                self.l_ij, p, self.n_classes, self.worker_ids, all_worker_prior
            )
            log_likelihood, relative_diff = self.calc_log_likelihood(
                self.l_ij, p_non_normalized, log_likelihood
            )
            if relative_diff is not None and relative_diff < self.min_relative_diff:
                self.put_parameters(p, c_mtrx)
                responses = self.format_responses(
                    p, label_categories, label_attribute_name, is_text
                )
                return responses

            all_worker_prior = sum([worker_priors[j] for j in worker_priors]) / len(worker_priors)

        self.put_parameters(p, c_mtrx)
        responses = self.format_responses(p, label_categories, label_attribute_name, is_text)
        return responses

    def get_or_initialize_parameters(self, annotation_payload, label_categories, all_worker_prior):

        self.label_categories = label_categories
        self.n_classes = len(label_categories)

        for item in annotation_payload:
            i = item["datasetObjectId"]
            self.dataset_object_ids.add(i)
            for annotation in item["annotations"]:
                j = annotation["workerId"]
                self.worker_ids.add(j)
                annotation_content = annotation["annotationData"]["content"]
                self.l_ij[i][j] = self.label_categories.index(annotation_content)

        p = {}
        for i in self.dataset_object_ids:
            item_params = self.initialize_item_parameters(n_classes=self.n_classes)
            p[i] = item_params

        c_mtrx = {}
        for j in self.worker_ids:
            worker_params = self.initialize_worker_params(
                n_classes=self.n_classes, a=all_worker_prior
            )
            c_mtrx[j] = worker_params

        return p, c_mtrx

    def put_parameters(self, p, c_mtrx):
        for i in self.dataset_object_ids:
            pickled_label_data = pickle.dumps(p[i])
            self.data_layer.put_label_information_s3(pickled_label_data, self.labeling_job_arn, i)

        for j in self.worker_ids:
            pickled_worker_data = pickle.dumps(c_mtrx[j])
            self.data_layer.put_worker_information_s3(pickled_worker_data, self.labeling_job_arn, j)

    @staticmethod
    def initialize_item_parameters(n_classes):
        return np.ones(n_classes) / n_classes

    @staticmethod
    def initialize_worker_params(n_classes, a=0.7):
        worker_params = np.ones((n_classes, n_classes)) * ((1 - a) / (n_classes - 1))
        np.fill_diagonal(worker_params, a)
        return worker_params

    @staticmethod
    def expectation_step(l_ij, p, c_mtrx, n_classes):
        p_prior = np.zeros(n_classes)
        for i in p:
            p_prior += p[i]
        p_prior /= p_prior.sum()

        for i in l_ij:
            p[i] = p_prior.copy()
            for j in l_ij[i]:
                annotated_class = l_ij[i][j]
                for true_class in range(n_classes):
                    error_rate = c_mtrx[j][true_class, annotated_class]
                    p[i][true_class] *= error_rate

        p_non_normalized = p.copy()
        for i in p:
            if p[i].sum() > 0:
                p[i] /= float(p[i].sum())
        return p, p_non_normalized

    def maximization_step(self, l_ij, p, n_classes, worker_ids, all_worker_prior):
        all_worker_prior_mtrx = self.initialize_worker_params(n_classes, a=all_worker_prior)

        c_mtrx = {}
        worker_accuracies = {}
        for j in worker_ids:
            c_mtrx[j] = np.zeros((n_classes, n_classes))
        for i in l_ij:
            for j in l_ij[i]:
                annotated_class = l_ij[i][j]
                for true_class in range(n_classes):
                    c_mtrx[j][true_class, annotated_class] += p[i][true_class]

        for j in worker_ids:
            num_annotations = c_mtrx[j].sum()
            worker_accuracies[j] = c_mtrx[j].diagonal().sum() / num_annotations
            worker_prior_mtrx = self.initialize_worker_params(n_classes, a=worker_accuracies[j])
            c_mtrx[j] += (
                worker_prior_mtrx * num_annotations + all_worker_prior_mtrx * num_annotations / 2
            )

            for true_class in range(n_classes):
                if c_mtrx[j][true_class].sum() > 0:
                    c_mtrx[j][true_class] /= float(c_mtrx[j][true_class].sum())

        return c_mtrx, worker_accuracies

    @staticmethod
    def calc_log_likelihood(l_ij, p_non_normalized, prev_log_likelihood=None):
        log_likelihood = 0.0
        relative_diff = None
        for i in l_ij:
            posterior_i = p_non_normalized[i]
            likelihood_i = posterior_i.sum()
            log_likelihood += np.log(likelihood_i)

        if prev_log_likelihood:
            diff = log_likelihood - prev_log_likelihood
            relative_diff = diff / prev_log_likelihood

        return log_likelihood, relative_diff

    def format_responses(self, params, label_categories, label_attribute_name, is_text):
        responses = []
        for dataset_object_id in params:
            label_estimate = params[dataset_object_id]
            confidence_score = round(max(label_estimate), 2)
            label, index = self.retrieve_annotation(label_estimate, label_categories)
            consolidated_annotation = self.transform_to_label(
                label, index, label_attribute_name, confidence_score, is_text
            )
            response = self.build_response(dataset_object_id, consolidated_annotation)
            responses.append(response)
        return responses

    def transform_to_label(
        self, estimated_label, index, label_attribute_name, confidence_score, is_text
    ):
        if is_text:
            return self.transform_to_text_label(
                estimated_label, index, label_attribute_name, confidence_score
            )
        else:
            return self.transform_to_image_label(
                estimated_label, index, label_attribute_name, confidence_score
            )

    def transform_to_image_label(
        self, estimated_label, index, label_attribute_name, confidence_score
    ):
        return {
            label_attribute_name: int(float(index)),
            label_attribute_name
            + "-metadata": {
                "class-name": estimated_label,
                "job-name": self.labeling_job_arn,
                "confidence": confidence_score,
                "type": "groundtruth/text-classification",
                "human-annotated": "yes",
                "creation-date": "date",
            },
        }

    @staticmethod
    def retrieve_annotation(label_estimate, label_categories):
        elem = label_categories[np.argmax(label_estimate, axis=0)]
        index = label_categories.index(elem)
        return elem, index

    @staticmethod
    def build_response(dataset_object_id, consolidated_annotation):
        return {
            "datasetObjectId": dataset_object_id,
            "consolidatedAnnotation": {"content": consolidated_annotation},
        }


def most_common(labels):
    unique_classes, class_votes = np.unique(labels, return_counts=True)
    winning_num_votes = np.max(class_votes)
    winning_class = unique_classes[np.where(class_votes == winning_num_votes)]
    if len(winning_class) == 1:
        return winning_class[0]
    else:
        return np.random.choice(winning_class)


def majority_vote(dset_objects):
    final_labels = []
    for dset_object in dset_objects:
        labels = []
        for annotation in dset_object["annotations"]:
            label = annotation["annotationData"]["content"]
            labels.append(label)
        winner = most_common(labels)
        final_labels.append(
            {
                "datasetObjectId": dset_object["datasetObjectId"],
                "consolidatedAnnotation": {
                    "content": {"categories-metadata": {"class-name": winner}}
                },
            }
        )
    return final_labels
    

def post_process(responses):
    output = list(map(lambda x: {"task_id": x["datasetObjectId"], "label": x["consolidatedAnnotation"]["content"]["categories-metadata"]["class-name"]}, responses))
    return output
    