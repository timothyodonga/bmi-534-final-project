import torch
import numpy as np
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(
        self,
        features,
        labels,
        train_flag=True,
        train_valid_split=0.8,
    ):
        self.train_flag = train_flag
        self.data_x = np.array(features)
        self.data_y = np.array(labels)

        # Get the last value of fog row - the label is the last element for each segmented window
        self.train_valid_cutoff = int(train_valid_split * len(self.data_x))
        print(f"Train cutoff index: {self.train_valid_cutoff}")

        print(self.data_x.shape)
        print(self.data_y.shape)

        self.data_x_train = self.data_x[: self.train_valid_cutoff, :, :]
        self.data_y_train = self.data_y[: self.train_valid_cutoff]

        self.data_x_valid = self.data_x[self.train_valid_cutoff :, :, :]
        self.data_y_valid = self.data_y[self.train_valid_cutoff :]

    def __len__(self):
        if self.train_flag:
            return len(self.data_x_train)
        else:
            return len(self.data_x_valid)

    def __getitem__(self, idx):
        if self.train_flag:
            return (
                torch.as_tensor(self.data_x_train[idx]).float().unsqueeze(0),
                torch.as_tensor(self.data_y_train[idx]).float(),
            )
        else:
            return (
                torch.as_tensor(self.data_x_valid[idx]).float().unsqueeze(0),
                torch.as_tensor(self.data_y_valid[idx]).float(),
            )


class CustomTestDatasetTwo(Dataset):
    def __init__(
        self,
        features,
        labels,
        subjects,
    ):
        self.data_x = np.array(features)
        self.data_y = np.array(labels)
        self.subjects = np.array(subjects).astype(int)

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, idx):
        return (
            torch.as_tensor(self.data_x[idx]).float().unsqueeze(0),
            torch.as_tensor(self.data_y[idx]).float(),
            torch.as_tensor(self.subjects[idx]),
        )
