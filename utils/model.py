import torch.nn as nn
import torch
import os


class CNN3D8CONV4MAXPOOL3FC(nn.Module): # even smaller
    def __init__(
        self,
        st_feature: int = 16,
        input_image_dim: list = [50, 46, 41],
        input_channel: int = 4
    ) -> None:

        super().__init__()

        self.st_feature = st_feature
        self.input_image_dim = input_image_dim
        self.input_channel = input_channel

        self.convs = nn.Sequential(
            nn.Conv3d(self.input_channel, st_feature, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1)),
            nn.BatchNorm3d(st_feature),
            nn.LeakyReLU(),
            nn.Conv3d(st_feature, st_feature * 2, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(st_feature*2),
            nn.LeakyReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            nn.Conv3d(st_feature * 2, st_feature * 2, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(st_feature*2),
            nn.LeakyReLU(),
            nn.Conv3d(st_feature * 2, st_feature * 4, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(st_feature*4),
            nn.LeakyReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            nn.Conv3d(st_feature * 4, st_feature * 4, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(st_feature* 4),
            nn.LeakyReLU(),
            nn.Conv3d(st_feature * 4, st_feature * 8, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(st_feature* 8),
            nn.LeakyReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            nn.Conv3d(st_feature * 8, st_feature * 8, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(st_feature* 8),
            nn.LeakyReLU(),
            nn.Conv3d(st_feature * 8, st_feature * 8, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(st_feature* 8),
            nn.LeakyReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),

        )

        self.input_size = torch.tensor(input_image_dim)
        self.fc_in_feature = (self.input_size/(2**4)).int().prod().item()
        self.fcs = nn.Sequential(nn.Linear(self.fc_in_feature * self.st_feature * 8, self.st_feature * 8),
                                 nn.Dropout3d(0.5),
                                 nn.Linear(self.st_feature * 8 , int(self.st_feature * 8 /2)),
                                 nn.Dropout3d(0.5),
                                 nn.Linear(int(self.st_feature * 8 /2), 1)
                                 )

        self.sigmoid = nn.Sigmoid()


    def forward(self, x):

        x = self.convs(x)
        x = x.view(x.size(0), -1)
        x = self.fcs(x)

        return x


CNN3D_1CHANNEL_8CONV4MP_3FC_F4 = CNN3D8CONV4MAXPOOL3FC(input_channel=1, st_feature=4)
CNN3D_3CHANNEL_8CONV4MP_3FC_F4 = CNN3D8CONV4MAXPOOL3FC(input_channel=3, st_feature=4)
CNN3D_3CHANNEL_8CONV4MP_3FC_F7 = CNN3D8CONV4MAXPOOL3FC(input_channel=3, st_feature=7)
