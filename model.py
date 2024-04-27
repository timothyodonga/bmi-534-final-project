import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn
import torch
from torch.nn import TransformerEncoder, TransformerEncoderLayer
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
        embed_length=142,
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
            in_features=64 * embed_length * num_sensor_channels, out_features=84
        )

        self.dropout3 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(in_features=84, out_features=num_output_classes)

        self.num_output_classes = num_output_classes
        self.num_sensor_channels = num_sensor_channels
        self.embed_length = embed_length
        self.num_filters = num_filters

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        x = x.view(-1, self.num_filters * self.embed_length * self.num_sensor_channels)
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


"""Two contrastive encoders"""


class TFC(nn.Module):
    def __init__(self, configs):
        super(TFC, self).__init__()

        encoder_layers_t = TransformerEncoderLayer(
            configs.TSlength_aligned,
            dim_feedforward=2 * configs.TSlength_aligned,
            nhead=2,
        )
        self.transformer_encoder_t = TransformerEncoder(encoder_layers_t, 2)

        self.projector_t = nn.Sequential(
            nn.Linear(configs.TSlength_aligned, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )

        encoder_layers_f = TransformerEncoderLayer(
            configs.TSlength_aligned,
            dim_feedforward=2 * configs.TSlength_aligned,
            nhead=2,
        )
        self.transformer_encoder_f = TransformerEncoder(encoder_layers_f, 2)

        self.projector_f = nn.Sequential(
            nn.Linear(configs.TSlength_aligned, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )

    def forward(self, x_in_t, x_in_f):
        """Use Transformer"""
        x = self.transformer_encoder_t(x_in_t)
        h_time = x.reshape(x.shape[0], -1)

        """Cross-space projector"""
        z_time = self.projector_t(h_time)

        """Frequency-based contrastive encoder"""
        f = self.transformer_encoder_f(x_in_f)
        h_freq = f.reshape(f.shape[0], -1)

        """Cross-space projector"""
        z_freq = self.projector_f(h_freq)

        return h_time, z_time, h_freq, z_freq


"""Downstream classifier only used in finetuning"""


class target_classifier(nn.Module):
    def __init__(self, configs):
        super(target_classifier, self).__init__()
        self.logits = nn.Linear(2 * 128, 64)
        self.logits_simple = nn.Linear(64, configs.num_classes_target)

    def forward(self, emb):
        emb_flat = emb.reshape(emb.shape[0], -1)
        emb = torch.sigmoid(self.logits(emb_flat))
        pred = self.logits_simple(emb)
        return pred
