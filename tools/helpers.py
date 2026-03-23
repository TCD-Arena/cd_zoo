import numpy as np
import pandas as pd
import os

from os import listdir
from os.path import isfile, join
import pickle
from yaml import safe_load
from omegaconf import OmegaConf
from pathlib import Path
import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl



def simple_sample_display(sample_data):
    fig, axs = plt.subplots(len(sample_data.columns),1)
    cmap = mpl.colormaps['plasma']
    # Take colors at regular intervals spanning the colormap.
    for n,s in enumerate(sample_data.columns):
        axs[n].set_ylabel(s, fontstyle="normal", fontsize=8,rotation=45)
        rgba = cmap(1/(n+1))
        axs[n].plot(sample_data[s].values, linewidth=2, color=rgba)
        axs[n].get_xaxis().set_ticks([])
        axs[n].get_yaxis().set_ticks([])
    axs[n].get_yaxis().set_ticks([])
    axs[n].set_xlabel("Timesteps")
    plt.show()

def basic_preprocess(test_data, cfg):
    for i, sample in enumerate(test_data):
        if cfg.remove_trailing_nans:
            sample = remove_trailing_nans(sample)
        if cfg.normalize:
            sample = (sample - sample.mean()) / sample.std()
        if cfg.cut_at is not None:
            sample = sample[: cfg.cut_at]
        test_data[i] = sample
    return test_data


def remove_trailing_nans(sample_prep):
    """
    Removes samples until the last non-nan value.
    """
    check_trailing_nans = np.where(sample_prep.isnull().values.any(axis=1) == 0)[0]
    if not len(check_trailing_nans) == 0:  # A ts is completely 0:
        sample_prep = sample_prep[
            check_trailing_nans.min() : check_trailing_nans.max() + 1
        ]
    else:
        sample_prep = sample_prep.fillna(value=0)
    assert sample_prep.isnull().sum().max() == 0, "Check nans!"
    return sample_prep


def load_single_samples(cfg):
    # Load data. replace this here if necessary.
    onlyfiles = [f for f in listdir(cfg.data_path) if isfile(join(cfg.data_path , f))]
    test_data = sorted([x for x in onlyfiles if "data" in x])
    test_labels = sorted([x for x in onlyfiles if "label" in x])

    if cfg.only_single_sample_index >= 0:
        test_data = test_data[cfg.only_single_sample_index : cfg.only_single_sample_index + 1]
        test_labels = test_labels[cfg.only_single_sample_index : cfg.only_single_sample_index + 1]

    test_data = [
        pd.read_csv(cfg.data_path + sample, index_col=0).T for sample in test_data
    ]
    test_labels = [
        pd.read_csv(cfg.data_path + sample, index_col=0).astype(bool).values
        for sample in test_labels
    ]

    return test_data, test_labels


def load_joint_samples(cfg, index_col="datetime"):
    """
    Loads and transforms the data.
    if you have an index_col= specify it.
    """

    data = pickle.load(open(cfg.data_path + "labels.p", "rb"))
    if cfg.only_single_sample_index >= 0:
        data = data[cfg.only_single_sample_index : cfg.only_single_sample_index + 1]
    # This is not ram efficient but faster to process.
    Y = [graph_to_label_tensor(sample, human_readable=True) for sample in data]
    # To fix double col names due to human readable format.
    Y_names = [[m[1] for m in sample.columns.values] for sample in Y]
    unique_nodes = list(set([item for sublist in Y_names for item in sublist]))
    unique_nodes = (
        ([index_col] + [str(x) for x in unique_nodes])
        if index_col
        else [str(x) for x in unique_nodes]
    )
    data = pd.read_csv(
        cfg.data_path + "data.csv",
        index_col=index_col if index_col else None,
        usecols=unique_nodes,
    )
    X = []
    for sample in Y:
        X.append(data[[str(m[1]) for m in sample.columns]].T)
    return X, Y


def summary_transform(pred, cfg):
    if cfg.map_to_summary_graph == "max":
        prediction = pred.max(axis=2)
    elif cfg.map_to_summary_graph == "mean":
        prediction = pred.mean(axis=2)
    return prediction


def graph_to_label_tensor(G_sample, human_readable=False):
    nodes = sorted(G_sample.nodes)
    labels = np.zeros((len(nodes), len(nodes)))

    for n, x in enumerate(nodes):
        for m, y in enumerate(nodes):
            if (x, y) in G_sample.edges:
                labels[m, n] = 1
    if human_readable:
        labels = pd.DataFrame(labels, columns=nodes, index=nodes)
        labels = pd.concat(
            [pd.concat([labels], keys=["Cause"], axis=1)], keys=["Effect"]
        )
        return labels
    else:
        return labels