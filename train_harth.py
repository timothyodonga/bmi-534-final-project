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

from model import BaselineCNN, MLP, BaselineCNNSmall
from routines.train import train_one_epoch
from routines.validate import validate_model
from routines.test import test_model
import kfolds

# %%
config = {
    "BATCH_SIZE": 64,
    "learning_rate": 0.0001,
    "weight_decay": 0.0001,
    "NUM_EPOCHS": 50,
    "train_valid_split": 0.8,
    "data": r"C:\Users\timot\OneDrive\Desktop\EMORY\Spring 2024\BMI-534\project-code\code\bmi-534-final-project\harth_preprocessed_data_150_window.csv",
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
    "model_name": f"cnn_small_harth",
}


df_performance = pd.DataFrame()


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

    # %%

    df_train = df[df["subject"].isin(train)]
    df_train["label_"].value_counts(normalize=True)

    df_test = df[df["subject"].isin(test)]
    df_test["label_"].value_counts(normalize=True)

    # %%
    df_train_train_X = df_train["features"].iloc[
        : int(len(df_train) * config["train_valid_split"])
    ]
    df_train_train_y = df_train["label_"].iloc[
        : int(len(df_train) * config["train_valid_split"])
    ]

    df_train_train_y.value_counts(normalize=True)

    print("Spliting the data into train<>validation")
    df_train_valid_X = df_train["features"].iloc[
        int(len(df_train) * config["train_valid_split"]) :
    ]
    df_train_valid_y = df_train["label_"].iloc[
        int(len(df_train) * config["train_valid_split"]) :
    ]
    print(df_train_valid_y.value_counts(normalize=True))

    # %%

    train_data = CustomDataset(
        features=df_train_train_X.tolist(),
        labels=df_train_train_y.tolist(),
        train_valid_split=1.0,
    )
    valid_data = CustomDataset(
        features=df_train_valid_X.tolist(),
        labels=df_train_valid_y.tolist(),
        train_valid_split=1.0,
    )
    test_data = CustomDataset(
        features=df_test["features"].tolist(),
        labels=df_test["label_"].tolist(),
        train_valid_split=1.0,
    )

    # %%

    train_dataloader = DataLoader(
        train_data, batch_size=config["BATCH_SIZE"], shuffle=True
    )

    validation_dataloader = DataLoader(
        valid_data, batch_size=config["BATCH_SIZE"], shuffle=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # %%s
    # model = BaselineCNN(num_sensor_channels=6, num_output_classes=12)
    model = BaselineCNNSmall(num_sensor_channels=6, num_output_classes=12)
    # model = MLP(num_sensor_channels=6, num_output_classes=12)
    model.to(device)
    print(model)
    # %%
    criterion = nn.CrossEntropyLoss()
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

    for epoch_number in range(config["NUM_EPOCHS"]):
        print("EPOCH {}:".format(epoch_number))
        model.train(True)
        model.zero_grad()

        print("===============")
        print("Training now")
        avg_loss = train_one_epoch(
            train_dataloader=train_dataloader,
            optimizer=optimizer,
            device=device,
            criterion=criterion,
            model=model,
        )

        print(f"Epoch {epoch_number} TRAIN-LOSS: {avg_loss}")

        best_vloss = validate_model(
            model=model,
            validation_dataloader=validation_dataloader,
            device=device,
            loss_fn=criterion,
            best_vloss=best_vloss,
            epoch_number=epoch_number,
            key=key,
            k=config["k"],
            timestamp=timestamp,
            model_name=model_name,
        )

    # %%
    model.load_state_dict(
        torch.load(f"{config['model_name']}_{key}_{config['k']}_{timestamp}.pth")
    )
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

    f1_value = f1_score(y_true=out_label, y_pred=out_pred, average="macro")
    print(f1_value)
    acc_value = accuracy_score(y_true=out_label, y_pred=out_pred)
    print(acc_value)
    roc_value = roc_auc_score(
        y_true=out_label, y_score=out_prob, multi_class="ovr", average="macro"
    )

    new_data = {
        "fold": key,
        "f1": f1_value,
        "acc": acc_value,
        "auc": roc_value,
        "model": f"{config['model_name']}_{key}_{config['k']}_{timestamp}.pth",
    }

    df_performance = pd.concat(
        [df_performance, pd.DataFrame(new_data, index=[0])], ignore_index=True
    )
# %%
df_performance.to_csv("harth_cnn_small_performance_.csv", index=False, mode="a")
