# %%
import ast
from datetime import datetime
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    roc_auc_score,
)
import torch.nn.functional as F
from dataset.dataset import CustomDataset
from torch.utils.data import DataLoader
from model import *
from routines.test import test_model
import kfolds
from utils import standardize_classes
import os

# %%
config = {
    "BATCH_SIZE": 64,
    "learning_rate": 0.0001,
    "weight_decay": 0.0001,
    "NUM_EPOCHS": 50,
    "train_valid_split": 0.8,
    "data": r"C:\Users\timot\OneDrive\Desktop\EMORY\Spring 2024\BMI-534\project-code\code\bmi-534-final-project\csv_files\harth_preprocessed_data_150_window.csv",
    "num_sensor_channels": 6,
    "class_weights": [
        0.15076889696509452,
        0.01749342301537794,
        0.04228255268368094,
        0.011716525182392667,
        0.009465433538553334,
        0.12665780694855036,
        0.5197851970383228,
        0.07057036695505953,
        0.04228255268368094,
        0.004041116324482656,
        0.0047191559762414905,
        0.00021697268856282715,
    ],
    "k": 5,
    "model_name": f"mlp_harth",
    "saved_models": {
        "0": r"saved_models\mlp_harth_0_5_20240422_124416.pth",
        "1": r"saved_models\mlp_harth_1_5_20240422_124416.pth",
        "2": r"saved_models\mlp_harth_2_5_20240422_124416.pth",
        "3": r"saved_models\mlp_harth_3_5_20240422_124416.pth",
        "4": r"saved_models\mlp_harth_4_5_20240422_124416.pth",
    },
}


df_performance = pd.DataFrame()


# %%


# %%
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
df = pd.read_csv(config["data"])
print(df.head())

df["label_"] = df["label"].apply(lambda x: int(ast.literal_eval(x)[-1]))
df["label_"] = df["label_"].apply(lambda x: int(standardize_classes(x)))

print(df["label_"].value_counts(normalize=True))
df["features"] = df["features"].apply(lambda x: ast.literal_eval(x))

print("Printing the folds")
if config["k"] == 5:
    k_fold_split = kfolds.harth_5_fold
else:
    k_fold_split = kfolds.harth_5_fold

print(k_fold_split)

subjects = list(df["subject"].unique())

# %%
for key, value in k_fold_split.items():
    print(key)
    print(value)
    train = value["train"]
    test = value["test"]

    df_test = df[df["subject"].isin(test)]
    df_test["label_"].value_counts(normalize=True)

    # %%

    test_data = CustomDataset(
        features=df_test["features"].tolist(),
        labels=df_test["label_"].tolist(),
        train_valid_split=1.0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # %%s
    # model = BaselineCNN(num_sensor_channels=6, num_output_classes=12)
    # model = BaselineCNNSmall(num_sensor_channels=6, num_output_classes=12)
    # model = MLPSmall(num_sensor_channels=6, num_output_classes=12)
    model = MLP(num_sensor_channels=6, num_output_classes=12)
    model.to(device)
    print(model)
    # %%
    # criterion = nn.CrossEntropyLoss()
    class_weights = 1 / torch.tensor(config["class_weights"], dtype=torch.float)
    class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)
    # criterion = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    model_name = config["model_name"]
    best_vloss = np.inf
    loss_fn = criterion

    # %%
    model.load_state_dict(torch.load(config["saved_models"][key]))
    model.to(device)

    test_dataloader = DataLoader(test_data, batch_size=1024, shuffle=False)

    out_batch, out_label, out_pred, out_prob = test_model(
        model=model,
        test_dataloader=test_dataloader,
        criterion=criterion,
        device=device,
        f1_type="macro",
    )

    lbl = pd.DataFrame({"label": out_label})
    print(lbl.value_counts())

    lbl = pd.DataFrame({"label": out_pred})
    print(lbl.value_counts())

    f1_value = f1_score(
        y_true=out_label, y_pred=out_pred, average="macro", zero_division=np.nan
    )
    print(f1_value)
    acc_value = accuracy_score(y_true=out_label, y_pred=out_pred)
    print(acc_value)
    roc_value = roc_auc_score(
        y_true=out_label, y_score=out_prob, multi_class="ovr", average="macro"
    )

    f1_per_class = f1_score(
        y_true=out_label, y_pred=out_pred, average=None, zero_division=np.nan
    )
    print(f1_per_class)

    new_data = {
        "fold": key,
        "f1": f1_value,
        "acc": acc_value,
        "auc": roc_value,
        "f1 per class": f1_per_class,
        "model": config["saved_models"][key],
    }

    df_performance = pd.concat(
        [df_performance, pd.DataFrame(new_data, index=[0])], ignore_index=True
    )
# %%
file_path = f'{config["model_name"]}_performance_150_{timestamp}.csv'
if os.path.exists(file_path):
    df_performance.to_csv(file_path, index=False, mode="a")
else:
    df_performance.to_csv(file_path, index=False)
