import torch
import torch.nn as nn
import torch.nn.functional as F

class CropRecommendationNet(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_p=0.2):
        super(CropRecommendationNet, self).__init__()
        # Architecture: 7 -> 64 -> 128 -> 64 -> num_classes
        self.fc1 = nn.Linear(input_dim, 64)
        self.dropout1 = nn.Dropout(p=dropout_p)

        self.fc2 = nn.Linear(64, 128)
        self.dropout2 = nn.Dropout(p=dropout_p)

        self.fc3 = nn.Linear(128, 64)
        self.dropout3 = nn.Dropout(p=dropout_p)

        self.fc4 = nn.Linear(64, output_dim)

    def forward(self, x):
        # Use SELU to match original training
        x = F.selu(self.fc1(x))
        x = self.dropout1(x)
        x = F.selu(self.fc2(x))
        x = self.dropout2(x)
        x = F.selu(self.fc3(x))
        x = self.dropout3(x)
        # Output logits (softmax applied externally)
        return self.fc4(x)