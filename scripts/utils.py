import logging
import os
import time
import numpy as np
import pandas as pd
import requests
import sympy
import torch
from sklearn import feature_selection


def nrmse(y_true, y_pred):
    """y, y_pred should be (num_samples,)"""
    assert y_true.shape == y_pred.shape, "y_true and y_pred must have the same shape"
    var = torch.var(y_true)
    return (torch.sqrt(torch.mean((y_true - y_pred) ** 2)) / var).item()


def MSE(y, y_pred):
    return torch.mean(torch.square(y - y_pred)).item()


def Relative_Error(y, y_pred):
    return torch.mean(torch.abs((y - y_pred) / y)).item()


def nrmse_np(y_true, y_pred):
    """y, y_pred should be (num_samples,)"""
    assert y_true.shape == y_pred.shape, "y_true and y_pred must have the same shape"
    var = np.var(y_true)
    return np.sqrt(np.mean((y_true - y_pred) ** 2)) / var


def R_Square(y, y_pred):
    """y, y_pred should be same shape (num_samples,) or (num_samples, 1)"""
    return (1 - torch.sum(torch.square(y - y_pred)) / torch.sum(torch.square(y - torch.mean(y)))).item()


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def get_top_k_features(X, y, k=10):
    if y.ndim == 2:
        y = y[:, 0]
    # if X.shape[1] <= k:
    #     return [i for i in range(X.shape[1])]
    else:
        kbest = feature_selection.SelectKBest(feature_selection.r_regression, k=k)
        kbest.fit(X, y)
        scores = kbest.scores_
        # scores = corr(X, y)
        top_features = np.argsort(-np.abs(scores))
        print("keeping only the top-{} features. Order was {}".format(k, top_features))
        return list(top_features[:k])
