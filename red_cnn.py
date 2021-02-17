import torch.nn as nn


class RED_CNN(nn.Module):
    def __init__(self, channel=96):
        super(RED_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, channel, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(channel, channel, kernel_size=5, stride=1, padding=0)
        self.conv3 = nn.Conv2d(channel, channel, kernel_size=5, stride=1, padding=0)
        self.conv4 = nn.Conv2d(channel, channel, kernel_size=5, stride=1, padding=0)
        self.conv5 = nn.Conv2d(channel, channel, kernel_size=5, stride=1, padding=0)

        self.deconv1 = nn.ConvTranspose2d(channel, channel, kernel_size=5, stride=1, padding=0)
        self.deconv2 = nn.ConvTranspose2d(channel, channel, kernel_size=5, stride=1, padding=0)
        self.deconv3 = nn.ConvTranspose2d(channel, channel, kernel_size=5, stride=1, padding=0)
        self.deconv4 = nn.ConvTranspose2d(channel, channel, kernel_size=5, stride=1, padding=0)
        self.deconv5 = nn.ConvTranspose2d(channel, 1, kernel_size=5, stride=1, padding=0)

        self.relu = nn.ReLU()

    def forward(self, x):
        # encoder
        res1 = x
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        res2 = out
        out = self.relu(self.conv3(out))
        out = self.relu(self.conv4(out))
        res3 = out
        out = self.relu(self.conv5(out))
        # decoder
        out = self.deconv1(out)
        out += res3
        out = self.deconv2(self.relu(out))
        out = self.deconv3(self.relu(out))
        out += res2
        out = self.deconv4(self.relu(out))
        out = self.deconv5(self.relu(out))
        out += res1
        out = self.relu(out)
        return out