import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        """
        Example        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=5, stride=1, padding=2)
        self.conv1_s = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1)
        self.conv2_s = nn.Conv2d(in_channels=6, out_channels=6, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=6, out_channels=10, kernel_size=3, stride=1, padding=1)
        self.conv3_s = nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3, stride=2, padding=1)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(10, 10)
        """

    @staticmethod
    def forward(x):
        """
        Example

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv1_s(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv2_s(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv3_s(x))

        x = self.flatten(x)
        x = self.fc1(x)
        x = F.softmax(x)
        """

        return x


def train():
    return True


def test():
    return True


if __name__ == '__main__':
    print("Hi there")
