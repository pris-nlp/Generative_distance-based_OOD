from typing import List
import os
import json
import pandas as pd
import itertools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.backend import set_session
import numpy as np
import random as rn
from sklearn.decomposition import PCA

SEED = 20190222
tf.set_random_seed(SEED)

def set_allow_growth(device="1"):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    config.gpu_options.visible_device_list=device
    sess = tf.Session(config=config)
    set_session(sess)  # set this TensorFlow session as the default session for Keras

def load_data(dataset):
    texts = []
    labels = []
    partition_to_n_row = {}
    for partition in ['train', 'valid', 'test']:
        with open("data/" + dataset + "/" + partition + ".seq.in") as fp:
            lines = fp.read().splitlines()
            texts.extend(lines)
            partition_to_n_row[partition] = len(lines)
        with open("data/" + dataset + "/" + partition + ".label") as fp:
            labels.extend(fp.read().splitlines())

    df = pd.DataFrame([texts, labels]).T
    df.columns = ['text', 'label']
    return df, partition_to_n_row

def get_score(cm):
    fs = []
    n_class = cm.shape[0]
    for idx in range(n_class):
        TP = cm[idx][idx]
        r = TP / cm[idx].sum() if cm[idx].sum()!=0 else 0
        p = TP / cm[:, idx].sum() if cm[:, idx].sum()!=0 else 0
        f = 2*r*p/(r+p) if (r+p)!=0 else 0
        fs.append(f*100)

    f = np.mean(fs).round(2)
    f_seen = np.mean(fs[:-1]).round(2)
    f_unseen = round(fs[-1], 2)
    print("Overall(macro): ", f)
    print("Seen(macro): ", f_seen)
    print("=====> Uneen(Experiment) <=====: ", f_unseen)
    
    return f, f_seen, f_unseen

def mahalanobis_distance(x: np.ndarray,
                         y: np.ndarray,
                         covariance: np.ndarray) -> float:
    """
    Calculate the mahalanobis distance.

    Params:
        - x: the sample x, shape (num_features,)
        - y: the sample y (or the mean of the distribution), shape (num_features,)
        - covariance: the covariance of the distribution, shape (num_features, num_features)
    
    Returns:
        - score: the mahalanobis distance in float
    
    """
    num_features = x.shape[0]

    vec = x - y
    cov_inv = np.linalg.inv(covariance)
    bef_sqrt = np.matmul(np.matmul(vec.reshape(1, num_features), cov_inv), vec.reshape(num_features, 1))
    return np.sqrt(bef_sqrt).item()

def confidence(features: np.ndarray,
               means: np.ndarray,
               distance_type: str,
               cov: np.ndarray = None) -> np.ndarray:
    """
    Calculate mahalanobis or euclidean based confidence score for each class.

    Params:
        - features: shape (num_samples, num_features)
        - means: shape (num_classes, num_features)
        - cov: shape (num_features, num_features) or None (if use euclidean distance)
    
    Returns:
        - confidence: shape (num_samples, num_classes)
    """
    assert distance_type in ("euclidean", "mahalanobis")

    num_samples = features.shape[0]
    num_features = features.shape[1]
    num_classes = means.shape[0]
    if distance_type == "euclidean":
        cov = np.identity(num_features)
    
    features = features.reshape(num_samples, 1, num_features).repeat(num_classes, axis=1)  # (num_samples, num_classes, num_features)
    means = means.reshape(1, num_classes, num_features).repeat(num_samples, axis=0)  # (num_samples, num_classes, num_features)
    vectors = features - means  # (num_samples, num_classes, num_features)
    cov_inv = np.linalg.inv(cov)
    bef_sqrt = np.matmul(np.matmul(vectors.reshape(num_samples, num_classes, 1, num_features), cov_inv),
                         vectors.reshape(num_samples, num_classes, num_features, 1)).squeeze()
    result = np.sqrt(bef_sqrt)
    result[np.isnan(result)] = 1e12  # solve nan
    return result

def get_test_info(texts: pd.Series,
                  label: pd.Series,
                  label_mask: pd.Series,
                  softmax_prob: np.ndarray,
                  softmax_classes: List[str],
                  lof_result: np.ndarray = None,
                  gda_result: np.ndarray = None,
                  gda_classes: List[str] = None,
                  save_to_file: bool = False,
                  output_dir: str = None) -> pd.DataFrame:
    """
    Return a pd.DataFrame, including the following information for each test instances:
        - the text of the instance
        - label & masked label of the sentence
        - the softmax probability for each seen classes (sum up to 1)
        - the softmax prediction
        - the softmax confidence (i.e. the max softmax probability among all seen classes)
        - (if use lof) lof prediction result (1 for in-domain and -1 for out-of-domain)
        - (if use gda) gda mahalanobis distance for each seen classes
        - (if use gda) the gda confidence (i.e. the min mahalanobis distance among all seen classes)
    """
    df = pd.DataFrame()
    df['label'] = label
    df['label_mask'] = label_mask
    for idx, _class in enumerate(softmax_classes):
        df[f'softmax_prob_{_class}'] = softmax_prob[:, idx]
    df['softmax_prediction'] = [softmax_classes[idx] for idx in softmax_prob.argmax(axis=-1)]
    df['softmax_confidence'] = softmax_prob.max(axis=-1)
    if lof_result is not None:
        df['lof_prediction'] = lof_result
    if gda_result is not None:
        for idx, _class in enumerate(gda_classes):
            df[f'm_dist_{_class}'] = gda_result[:, idx]
        df['gda_prediction'] = [gda_classes[idx] for idx in gda_result.argmin(axis=-1)]
        df['gda_confidence'] = gda_result.min(axis=-1)
    df['text'] = [text for text in texts]

    if save_to_file:
        df.to_csv(os.path.join(output_dir, "test_info.csv"))
    
    return df

def estimate_best_threshold(seen_m_dist: np.ndarray,
                            unseen_m_dist: np.ndarray) -> float:
    """
    Given mahalanobis distance for seen and unseen instances in valid set, estimate
    a best threshold (i.e. achieving best f1 in valid set) for test set.
    """
    lst = []
    for item in seen_m_dist:
        lst.append((item, "seen"))
    for item in unseen_m_dist:
        lst.append((item, "unseen"))
    lst = sorted(lst, key=lambda item: item[0])

    threshold = 0.
    tp, fp, fn = len(unseen_m_dist), len(seen_m_dist), 0
    def compute_f1(tp, fp, fn):
        p = tp / (tp + fp + 1e-10)
        r = tp / (tp + fn + 1e-10)
        return (2 * p * r) / (p + r + 1e-10)
    f1 = compute_f1(tp, fp, fn)

    for m_dist, label in lst:
        if label == "seen":  # fp -> tn
            fp -= 1
        else:  # tp -> fn
            tp -= 1
            fn += 1
        if compute_f1(tp, fp, fn) > f1:
            f1 = compute_f1(tp, fp, fn)
            threshold = m_dist + 1e-10

    print("estimated threshold:", threshold)
    return threshold
