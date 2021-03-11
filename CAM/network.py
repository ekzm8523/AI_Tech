from torch import nn

class VGG16_CAM(nn.Module):
    def __init__(self):
        super(VGG16_CAM, self).__init__()

        self.CAM_conv = nn.Conv2d(512,1024, kernel_size=(3,3), stride=(1,1), padding=(1,1), groups=2)
        self.CAM_relu = nn.ReLU(inplace=True)
        self.CAM_gap = nn.AvgPoll2d(kernel_size=14, stride=14)
        self.CAM_fc = nn.Linear(in_features=1024, out_features=1000, bias=True)

    def forward(self, x):
        x = self.CAM_conv(x)
        x = self.CAM_relu(x)
        x = self.CAM_gap(x)
        x = self.CAM_fc(x)

        return x

model = VGG16_CAM()


