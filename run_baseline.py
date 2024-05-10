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

# from model import BaselineCNN
from routines.train import train_one_epoch
from routines.validate import validate_model
from routines.test import test_model

# %%
config = {
    "BATCH_SIZE": 64,
    "learning_rate": 0.0001,
    "weight_decay": 0.0001,
    "NUM_EPOCHS": 1,
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
    "k": 10,
    "model_name": f"cnn_harth",
}


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

# %%
subjects = list(df["subject"].unique())
train_subjects = subjects[: int(len(subjects) * 0.8)]
test_subjects = subjects[int(len(subjects) * 0.8) :]

# %%
# train_subjects = train_subjects[: int(len(train_subjects) * 0.9)]
# valid_subjects = train_subjects[int(len(train_subjects) * 0.9) :]


# %%

df_train = df[df["subject"].isin(train_subjects)]
df_train["label_"].value_counts(normalize=True)

df_test = df[df["subject"].isin(test_subjects)]
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

train_dataloader = DataLoader(train_data, batch_size=config["BATCH_SIZE"], shuffle=True)

validation_dataloader = DataLoader(
    valid_data, batch_size=config["BATCH_SIZE"], shuffle=False
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineCNN(nn.Module):
    def __init__(
        self,
        num_sensor_channels,
        num_output_classes=2,
        # num_units_lstm=128,
        # num_lstm_layers=2,
        filter_size=5,
        num_filters=64,
    ):
        super(BaselineCNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=num_filters, kernel_size=(filter_size, 1)
        )
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(
            in_channels=num_filters,
            out_channels=num_filters,
            kernel_size=(filter_size, 1),
        )
        self.bn2 = nn.BatchNorm2d(num_filters)
        self.conv3 = nn.Conv2d(
            in_channels=num_filters,
            out_channels=num_filters,
            kernel_size=(filter_size, 1),
        )
        self.bn3 = nn.BatchNorm2d(num_filters)
        self.conv4 = nn.Conv2d(
            in_channels=num_filters,
            out_channels=num_filters,
            kernel_size=(filter_size, 1),
        )
        self.bn4 = nn.BatchNorm2d(num_filters)

        self.dropout1 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(
            in_features=64 * 134 * num_sensor_channels, out_features=84
        )
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=84, out_features=84)

        self.dropout3 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(in_features=84, out_features=num_output_classes)

        self.num_output_classes = num_output_classes
        self.num_sensor_channels = num_sensor_channels

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))

        # print("Printing the shape of x after the 4 convolution operations")
        # print(x.shape)

        # TODO - Remove the hardcoded 134 number.
        x = x.view(-1, 64 * 134 * self.num_sensor_channels)
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        x = self.dropout3(x)
        x = self.fc3(x)

        return x


# %%s
model = BaselineCNN(num_sensor_channels=6, num_output_classes=12)
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
        key="0",
        k=config["k"],
        timestamp=timestamp,
        model_name=model_name,
    )

# %%
key = "0"
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

lbl = pd.DataFrame({"label": out_pred})


print(f"F1 score: {f1_score(y_true=out_label, y_pred=out_pred, average='macro')}")
print(f"Accuracy: {accuracy_score(y_true=out_label, y_pred=out_pred)}")
print(
    f"AUC ROC: {roc_auc_score(y_true=out_label, y_score=out_prob, multi_class='ovr')}"
)

# %%
