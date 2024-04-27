# %%
import pandas as pd
import os
from utils import opp_sliding_windowX


# TODO - Need to figure out if it is necessary to normalize the data
# %%
data_path = r"C:\Users\timot\OneDrive\Desktop\EMORY\Spring 2024\BMI-534\project-code\data\harth\harth"
sliding_window_length = 150
sliding_window_step = 150

# %%
os.listdir(data_path)

# %%
df = pd.DataFrame()
csv_files = [file for file in os.listdir(data_path) if file.endswith(".csv")]

# %%
for file in csv_files:
    print(file)
    print(f"Processing : {file}")
    subject_id = file.split(".")[0]
    df_ = pd.read_csv(f"{data_path}/{file}")
    print(df_.head())

    features = opp_sliding_windowX(
        df_[
            [
                "back_x",
                "back_y",
                "back_z",
                "thigh_x",
                "thigh_y",
                "thigh_z",
            ]
        ].to_numpy(),
        sliding_window_length,
        sliding_window_step,
    )

    label = opp_sliding_windowX(
        df_[
            [
                "label",
            ]
        ].to_numpy(),
        sliding_window_length,
        sliding_window_step,
    )

    data = pd.DataFrame()
    for i in range(features.shape[0]):
        df__ = pd.DataFrame(
            {
                # "Time of sample in millisecond": [timesteps],
                "subject": [subject_id],
                "features": [features[i].tolist()],
                "label": [label[i].tolist()],
            }
        )
        data = pd.concat([data, df__], ignore_index=True)
    print(f"Finished processing: {file}")
    df = pd.concat([df, data], ignore_index=True)
# %%
df.to_csv("harth_preprocessed_data_150_window.csv", index=False)

# %%
