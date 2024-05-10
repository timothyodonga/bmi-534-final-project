# %%
import pandas as pd
import os
from utils import opp_sliding_windowX
from sklearn.decomposition import PCA


# TODO - Need to figure out if it is necessary to normalize the data
# %%
data_path = r"C:\Users\timot\OneDrive\Desktop\EMORY\Spring 2024\BMI-534\project-code\data\harth\harth"
sliding_window_length = 300
sliding_window_step = 300

# %%
os.listdir(data_path)

# %%
df = pd.DataFrame()
csv_files = [file for file in os.listdir(data_path) if file.endswith(".csv")]

max_values_harth = pd.read_csv("csv_files/defog_max_values.csv").to_numpy()
min_values_harth = pd.read_csv("csv_files/defog_min_values.csv").to_numpy()

# %%
for file in csv_files:
    print(file)
    print(f"Processing : {file}")
    subject_id = file.split(".")[0]
    df_ = pd.read_csv(f"{data_path}/{file}")
    print(df_.head())

    # Normalize
    features = df_[
        [
            "back_x",
            "back_y",
            "back_z",
            "thigh_x",
            "thigh_y",
            "thigh_z",
        ]
    ].to_numpy()

    diffs = max_values_harth - min_values_harth
    features = (features - min_values_harth) / diffs

    pca = PCA(n_components=3)
    pca.fit(features)
    print("Printing the amount of variance explained by first component")
    print(pca.explained_variance_ratio_)

    features_reduced = pca.fit_transform(features)

    features = opp_sliding_windowX(
        features_reduced,
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
df.to_csv("harth_preprocessed_data_pca_3pcs_206_window.csv", index=False)

# %%
