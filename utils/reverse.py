import torch
import torch.nn as nn


class ReverseVGG16(nn.Module):

    def __init__(self):
        super(ReverseVGG16, self).__init__()

        self.TransConv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=0, output_padding=0),
            nn.ReLU(inplace=True)
        )

        self.TransConv2 = nn.Sequential(
            # nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True)
        )

        self.TransConv3 = nn.Sequential(
            # nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=0, output_padding=0),
            nn.ReLU(inplace=True)
        )

        self.TransConv4 = nn.Sequential(
            # nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True)
        )

        self.TransConv5 = nn.Sequential(
            # nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True)
        )

        self.TransConv6 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):

        x = self.TransConv1(x)
        x = self.TransConv2(x)
        x = self.TransConv3(x)
        x = self.TransConv4(x)
        x = self.TransConv5(x)
        x = self.TransConv6(x)

        return x
