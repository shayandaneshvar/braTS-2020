# -*- coding: utf-8 -*-
"""SurvivalPrediction.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1TqTfCZkTbxzWn_8Zv882jY4CVWDx-wEf
"""

# Survival Prediction MOdel


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SurvivalPRED(nn.Module):
    def __init__(self, in_channels=4, out_channels=3):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)

        return x