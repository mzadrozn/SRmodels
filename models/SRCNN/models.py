from torch import nn


class SRCNN(nn.Module):
    def __init__(self, num_chanels=1):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(num_chanels, 64, kernel_size=(9, 9), padding='same')
        self.conv2 = nn.Conv2d(64, 32, kernel_size=(1, 1), padding='same')
        self.conv3 = nn.Conv2d(32, num_chanels, kernel_size=(5, 5), padding='same')
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x