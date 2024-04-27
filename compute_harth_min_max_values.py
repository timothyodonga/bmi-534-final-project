# %%
import pandas as pd
import os

data_path = r"C:\Users\timot\OneDrive\Desktop\EMORY\Spring 2024\BMI-534\project-code\data\harth\harth"
sliding_window_length = 206
sliding_window_step = 206
# %%
print(os.listdir(data_path))

# %%
df = pd.DataFrame()
print(df.head())

subj_files = [file for file in os.listdir(data_path)]

# %%
for i in range(len(subj_files)):
    file = subj_files[i]
    print(file)
    df_ = pd.read_csv(os.path.join(data_path, file))
    # print(df_.head())
    df = pd.concat([df, df_], ignore_index=True)
# %%
min_values = df[["back_x", "back_y", "back_z", "thigh_x", "thigh_y", "thigh_z"]].min()

# %%
min_values_df = pd.DataFrame(min_values).T
min_values_df.to_csv("harth_min_values.csv", index=False)


# %%
max_values = df[["back_x", "back_y", "back_z", "thigh_x", "thigh_y", "thigh_z"]].max()

# %%
max_values_df = pd.DataFrame(max_values).T
max_values_df.to_csv("harth_max_values.csv", index=False)

# %%
