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

# %%
data = r"C:\Users\timot\OneDrive\Desktop\EMORY\Spring 2024\BMI-534\project-code\code\bmi-534-final-project\harth_preprocessed_data_206_window.csv"
# %%
df = pd.read_csv(data)


# %%
def standardize_classes(x):
    if x == 1:
        return 0
    if x == 2:
        return 1
    elif x == 3:
        return 2
    elif x == 4:
        return 3
    elif x == 5:
        return 4
    elif x == 6:
        return 5
    elif x == 7:
        return 6
    elif x == 8:
        return 7
    elif x == 13:
        return 8
    elif x == 14:
        return 9
    elif x == 130:
        return 10
    elif x == 140:
        return 11


# %%
df["label_"] = df["label"].apply(lambda x: int(ast.literal_eval(x)[-1]))
df["label_"] = df["label_"].apply(lambda x: int(standardize_classes(x)))
# %%
print(df["label_"].value_counts(normalize=True))
df["features"] = df["features"].apply(lambda x: ast.literal_eval(x))

# %%
print("Printing the folds")
k_fold_split = kfolds.harth_5_fold
print(k_fold_split)

subjects = list(df["subject"].unique())

# %%
for key, value in k_fold_split.items():
    train = value["train"]
    test = value["test"]

    print(train)
    print(test)

    df_train = df[df["subject"].isin(train)]
    print(df_train["label_"].value_counts(normalize=True))

    df_test = df[df["subject"].isin(test)]
    print(df_test["label_"].value_counts(normalize=True))

    features_train = np.array(df_train["features"].tolist())
    label_train = np.array(df_train["label_"].tolist())

    features_test = np.array(df_test["features"].tolist())
    label_test = np.array(df_test["label_"].tolist())

    train_tensor = {
        "samples": torch.as_tensor(features_train).permute(0, 2, 1).float(),
        "labels": torch.as_tensor(label_train).float(),
    }

    test_tensor = {
        "samples": torch.as_tensor(features_test).permute(0, 2, 1).float(),
        "labels": torch.as_tensor(label_test).float(),
    }

    torch.save(train_tensor, f"harth_train_fold_{key}.pt")
    torch.save(test_tensor, f"harth_test_fold_{key}.pt")

# %%
