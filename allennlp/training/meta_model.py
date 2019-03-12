import torch
import torch.nn as nn
import torch.nn.functional as F


class FCMetaNet(nn.Module):

    def __init__(self, first_layer_size):
        super(FCMetaNet, self).__init__()
        self.fc1 = nn.Linear(first_layer_size, 4096) #input dimension both output and fc layer output
        self.bn1 = nn.BatchNorm1d(4096)

        self.fc2 = nn.Linear(4096, 7000)
        self.bn2 = nn.BatchNorm1d(7000)
        self.fc3 = nn.Linear(7000, 2048)
        self.bn3 = nn.BatchNorm1d(2048)

        self.fc4 = nn.Linear(2048, 512)
        self.bn4 = nn.BatchNorm1d(512)

        self.fc5 = nn.Linear(512, 512)
        self.bn5 = nn.BatchNorm1d(512)

        self.fc6 = nn.Linear(512, 64)
        self.bn6 = nn.BatchNorm1d(64)

        self.fc7 = nn.Linear(64, 2)


    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = self.bn1(x)

        #x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = self.bn2(x)

        x = F.dropout(x, training=self.training)

        x = F.relu(self.fc3(x))
        x = self.bn3(x)

        #x = F.dropout(x, training=self.training)
        x = F.relu(self.fc4(x))
        x = self.bn4(x)
        x = F.dropout(x, training=self.training)

        x = F.relu(self.fc5(x))
        x = self.bn5(x)
        #x = F.dropout(x, training=self.training)

        x = F.relu(self.fc6(x))
        x = self.bn6(x)

        x = self.fc7(x)

        return F.log_softmax(x, dim = 1)