# %%
import pandas as pd
import os
import torch

# from config import config_defog
from utils import opp_sliding_windowX
import os

# TODO - Make sure that the processed data conforms to the shape expected
# Num of samples , Num channels, Ts value - for now use a Ts value of 206 which is derived from the TF-C code
# This is a hyper parameter that can be tuned
# The preprocessing code is in the FOG wearables repo

# For this dataset, right now we don't really need to know the subject information so
# just preprocess it

# %%
# df = pd.read_parquet(
#     r"C:\Users\timot\OneDrive\Desktop\EMORY\Fall 2023\CS-598R -Rotation Project\Code\KaggleFOG\data\unlabeled\00c4c9313d.parquet"
# )

# %%
# The daily living dataset, has series for subject who are in the defog dataset. Careful with pre-training and then training on this dataset. If you pretrain and then test, you might end up leaking information
config = {
    "dir": r"C:\Users\timot\OneDrive\Desktop\EMORY\Fall 2023\CS-598R -Rotation Project\Code\KaggleFOG\data\unlabeled",
    "save_path": r"./",
    "sample": True,
}
dir = config["dir"]

print(os.listdir(dir))

# %%
# Need to concatenate the dataframes and compute the mean and value values

# For now just run it for 10 files in the daily living dataset. Get the code working and then run it for all the 65

df = pd.DataFrame()
print(df.head())

subj_files = [file for file in os.listdir(dir)]

# %%

for i in range(len(subj_files)):
    file = subj_files[i]
    print(file)
    df_ = pd.read_parquet(os.path.join(dir, file))
    # print(df_.head())
    df = pd.concat([df, df_], ignore_index=True)

    # TODO - Remove this in the code when the full code actually works
    if i == 4:
        break
    # print(df.head())
# %%
min_values = df.min()
# %%
min_values_df_sample = pd.DataFrame(min_values).T
min_values_df_sample[["AccV", "AccML", "AccAP"]].to_csv(
    f"{config['save_path']}/daily_living_min_values_sample.csv", index=False
)
# %%
max_values = df.max()
# %%
max_values_df_sample = pd.DataFrame(max_values).T
max_values_df_sample[["AccV", "AccML", "AccAP"]].to_csv(
    f"{config['save_path']}/daily_living_max_values_sample.csv", index=False
)

# %%
# Now to segment the data into 206 time chunks. Just do it for 5 subjects

max_values_daily = pd.read_csv(
    f"{config['save_path']}/daily_living_max_values_sample.csv"
).to_numpy()
min_values_daily = pd.read_csv(
    f"{config['save_path']}/daily_living_min_values_sample.csv"
).to_numpy()

# %%
sliding_window_length = 206
sliding_window_step = 206


segmented_features = []

for i in range(len(subj_files)):
    file = subj_files[i]
    print(file)
    df_ = pd.read_parquet(os.path.join(dir, file))
    df_ = df_[["AccV", "AccML", "AccAP"]]

    features = df_[["AccV", "AccML", "AccAP"]].to_numpy()
    diffs = max_values_daily - min_values_daily
    features = (features - min_values_daily) / diffs

    features = opp_sliding_windowX(
        features,
        sliding_window_length,
        sliding_window_step,
    )
    segmented_features.append(features)
    if config["sample"] == True:
        if i == 4:
            break

# %%
import numpy as np

features_ = np.vstack(segmented_features)

# %%
features_tensor = torch.tensor(features_)

# %%
features_tensor_ = features_tensor.permute(0, 2, 1)

# %%
# Create dummy labels
lbl = np.random.randint(0, 1, size=features_.shape[0])
train_tensor = {
    "samples": torch.as_tensor(features_).permute(0, 2, 1).float(),
    "labels": torch.as_tensor(lbl).float(),
}

# %%
if config["sample"]:
    torch.save(train_tensor, f"{config['save_path']}/daily_living_sample.pt")
else:
    torch.save(train_tensor, f"{config['save_path']}/daily_living.pt")
# %%
# np.random.randint(0,1,size = 10)
