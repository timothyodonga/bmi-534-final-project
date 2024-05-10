# %%
import ast
from datetime import datetime
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    roc_auc_score,
)
import torch.nn.functional as F
from dataset.dataset import CustomDataset
from torch.utils.data import DataLoader

from model import BaselineCNN, MLP
from routines.train import train_one_epoch
from routines.validate import validate_model
from routines.test import test_model
import kfolds
import argparse


# %%

parser = argparse.ArgumentParser()
parser.add_argument(
    "-data",
    type=str,
    help="Path to the csv file containing  preprocessed defog data",
)

args = parser.parse_args()
print(args)

data = args.data
# %%
df = pd.read_csv(data)


# %%
df["label_"] = df["fog"].apply(lambda x: int(ast.literal_eval(x)[-1]))
# %%
print(df["label_"].value_counts(normalize=True))
df["features"] = df["features"].apply(lambda x: ast.literal_eval(x))

# %%
print("Printing the folds")
k_fold_split = kfolds.defog_5_fold
print(k_fold_split)

subjects = list(df["Subject"].unique())

# %%
for key, value in k_fold_split.items():
    train = value["train"]
    test = value["test"]

    print(train)
    print(test)

    df_train = df[df["Subject"].isin(train)]
    print(df_train["label_"].value_counts(normalize=True))

    df_test = df[df["Subject"].isin(test)]
    print(df_test["label_"].value_counts(normalize=True))

    features_train = np.array(df_train["features"].tolist())
    label_train = np.array(df_train["label_"].tolist())

    features_test = np.array(df_test["features"].tolist())
    label_test = np.array(df_test["label_"].tolist())

    print(features_train.shape)

    train_tensor = {
        "samples": torch.as_tensor(features_train)
        # .unsqueeze(2)
        .permute(0, 2, 1).float(),
        "labels": torch.as_tensor(label_train).float(),
    }

    test_tensor = {
        # "samples": torch.as_tensor(features_test).unsqueeze(2).permute(0, 2, 1).float(),
        "samples": torch.as_tensor(features_test).permute(0, 2, 1).float(),
        "labels": torch.as_tensor(label_test).float(),
    }

    torch.save(train_tensor, f"defog_train_fold_{key}.pt")
    torch.save(test_tensor, f"defog_test_fold_{key}.pt")

# %%
