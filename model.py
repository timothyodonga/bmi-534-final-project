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


class BaselineCNNSmall(nn.Module):
    def __init__(
        self,
        num_sensor_channels,
        num_output_classes=2,
        # num_units_lstm=128,
        # num_lstm_layers=2,
        filter_size=5,
        num_filters=64,
    ):
        super(BaselineCNNSmall, self).__init__()
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

        self.dropout1 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(
            in_features=64 * 142 * num_sensor_channels, out_features=84
        )

        self.dropout3 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(in_features=84, out_features=num_output_classes)

        self.num_output_classes = num_output_classes
        self.num_sensor_channels = num_sensor_channels

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        x = x.view(-1, 64 * 142 * self.num_sensor_channels)
        # x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.fc3(x)

        return x


class MLP(nn.Module):
    def __init__(self, num_sensor_channels, num_output_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features=150 * num_sensor_channels, out_features=1800)
        self.fc2 = nn.Linear(in_features=1800, out_features=900)
        self.fc3 = nn.Linear(in_features=900, out_features=450)
        self.fc4 = nn.Linear(in_features=450, out_features=num_output_classes)
        self.num_sensor_channels = num_sensor_channels

    def forward(self, x):
        x = x.view(-1, 150 * self.num_sensor_channels)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        out = self.fc4(x)

        return out
