# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error, roc_auc_score


def BCE(targets, preds, weights=None):
    if weights is None:
        weights = np.ones(len(preds))
    weights_norm = weights / np.sum(weights)
    return np.sum(
        -np.log(preds + 10 ** (-15)) * targets * weights_norm
        - np.log(1 - preds + 10 ** (-15)) * (1 - targets) * weights_norm
    )


def BCE_se(targets, preds, weights=None):
    if weights is None:
        weights = np.ones(len(preds))
    n = len(preds)
    BCE_est = BCE(targets, preds, weights)
    weights_2 = weights**2
    return np.sqrt(
        (n / (n - 1))
        * np.sum(
            (
                weights_2
                * (
                    -np.log(preds + 10 ** (-15)) * targets
                    - np.log(1 - preds + 10 ** (-15)) * (1 - targets)
                    - BCE_est
                )
                ** 2
            )
        )
        / (np.sum(weights) ** 2)
    )


def BCE_seg(targets, preds, weights, segments, segments_order=None):
    if segments_order is None:
        segments_order = np.unique(segments)
    res = {}
    for seg in segments_order:
        idx_seg = np.where(segments == seg)[0]
        res[seg] = BCE(targets[idx_seg], preds[idx_seg], weights[idx_seg])
    return res


def BCE_seg_se(targets, preds, weights, segments, segments_order=None):
    if segments_order is None:
        segments_order = np.unique(segments)
    res = {}
    for seg in segments_order:
        idx_seg = np.where(segments == seg)[0]
        res[seg] = BCE_se(targets[idx_seg], preds[idx_seg], weights[idx_seg])
    return res


def mse_se(targets, preds):
    n = len(targets)
    theta = mean_squared_error(targets, preds)
    return np.sqrt(
        (n / (n - 1)) * np.sum(((targets - preds) ** 2 - theta) ** 2) / (n**2)
    )


def mse_seg(targets, preds, segments, segments_order=None):
    if segments_order is None:
        segments_order = np.unique(segments)
    res = {}
    for seg in segments_order:
        idx_seg = np.where(segments == seg)[0]
        res[seg] = mean_squared_error(targets[idx_seg], preds[idx_seg])
    return res


def mse_seg_se(targets, preds, segments, segments_order=None):
    if segments_order is None:
        segments_order = np.unique(segments)
    res = {}
    for seg in segments_order:
        idx_seg = np.where(segments == seg)[0]
        res[seg] = mse_se(targets[idx_seg], preds[idx_seg])
    return res


def log_loss_se(targets, preds, labels=None):
    n = len(targets)
    if labels is None:
        theta = log_loss(targets, preds, normalize=True)  # estimate mean
    else:
        theta = log_loss(targets, preds, normalize=True, labels=labels)  # estimate mean

    return np.sqrt(
        (n / (n - 1))
        * np.sum((np.log(preds[np.arange(n), targets] + 10 ** (-15)) - theta) ** 2)
        / (n**2)
    )


def log_loss_seg(targets, preds, segments, segments_order=None):
    labels = np.unique(targets)
    if segments_order is None:
        segments_order = np.unique(segments)
    res = {}
    for seg in segments_order:
        idx_seg = np.where(segments == seg)[0]
        res[seg] = log_loss(targets[idx_seg], preds[idx_seg], labels=labels)
    return res


def log_loss_seg_se(targets, preds, segments, segments_order=None):
    labels = np.unique(targets)
    if segments_order is None:
        segments_order = np.unique(segments)
    res = {}
    for seg in segments_order:
        idx_seg = np.where(segments == seg)[0]
        res[seg] = log_loss_se(targets[idx_seg], preds[idx_seg], labels=labels)
    return res


def accuracy_score_seg(targets, preds, segments, segments_order=None):
    if segments_order is None:
        segments_order = np.unique(segments)
    res = {}
    for seg in segments_order:
        idx_seg = np.where(segments == seg)[0]
        res[seg] = accuracy_score(targets[idx_seg], preds[idx_seg])
    return res


def roc_auc_score_se(targets, preds, multi_class="raise", B=1000):
    n = len(targets)
    boot_results = np.zeros(B)
    for b in range(B):

        id_boot = np.random.choice(n, n, replace=True)
        try:
            boot_results[b] = roc_auc_score(
                targets[id_boot], preds[id_boot], multi_class=multi_class
            )
        except ValueError:
            boot_results[b] = np.nan
    B_non_na = np.sum(~np.isnan(boot_results))
    return np.sqrt(
        np.nansum((boot_results - np.nanmean(boot_results)) ** 2) / (B_non_na - 1)
    )


def roc_auc_score_seg(
    targets, preds, segments, segments_order=None, multi_class="raise"
):
    if segments_order is None:
        segments_order = np.unique(segments)
    res = {}
    labels = np.unique(targets)
    for seg in segments_order:
        idx_seg = np.where(segments == seg)[0]
        res[seg] = roc_auc_score(
            targets[idx_seg], preds[idx_seg], multi_class=multi_class, labels=labels
        )
    return res


def roc_auc_score_seg_se(
    targets, preds, segments, segments_order=None, multi_class="raise", B=1000
):
    if segments_order is None:
        segments_order = np.unique(segments)
    res = {}
    for seg in segments_order:
        idx_seg = np.where(segments == seg)[0]
        res[seg] = roc_auc_score_se(
            targets[idx_seg], preds[idx_seg], multi_class=multi_class, B=B
        )
    return res
