from torch import nn, optim
from torch.nn import functional as F

class Model(nn.Module):
    def __init__(
            self,
            name='cnn',
            xdim=[1,28,28],
            ksize=3,
            cdims=[32,64],
            hdims=[1024,128],
            ydim=10,
            init_weight="he",
            init_bias="zero",
    ):
        super(Model,self).__init__()

        self.name = name
        self.xdim = xdim
        self.ksize = ksize
        self.cdims = cdims
        self.hdims = hdims
        self.ydim = ydim
        self.init_weight = init_weight
        self.init_bias = init_bias

        # Convolutional layers
        layer_list = []
        prev_cdim = xdim[0]
        for cdim in self.cdims:
            layer_list.append(
                nn.Conv2d(
                    in_channels=prev_cdim,
                    out_channels=cdim,
                    kernel_size=self.ksize,
                    stride=(1,1),
                    padding=self.ksize//2,
                )
            )
            layer_list.append(nn.BatchNorm2d(cdim))
            layer_list.append(nn.ReLU(True))
            layer_list.append(nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)))
            layer_list.append(nn.Dropout2d(p=0.5))
            prev_cdim = cdim
        # Dense layers
        layer_list.append(nn.Flatten())
        prev_hdim = prev_cdim*(self.xdim[1]//(2**len(self.cdims)))*(self.xdim[2]//(2**len(self.cdims)))
        for hdim in self.hdims:
            layer_list.append(nn.Linear(prev_hdim,hdim,bias=True))
            layer_list.append(nn.ReLU(True))
            prev_hdim = hdim

        # Final layer (without activation function)
        layer_list.append(nn.Linear(prev_hdim,self.ydim,bias=True))

        # Concatenate all layer + layer name
        self.net = nn.Sequential()
        for l_idx, layer in enumerate(layer_list):
            layer_name = "%s_%02d"%(type(layer).__name__.lower(),l_idx)
            self.net.add_module(layer_name,layer)
        self.init_param()

    def init_param(self):
        init_weight_method = {
            "he": nn.init.kaiming_normal_,
            "xavier": nn.init.xavier_normal_,
        }
        assert(
            self.init_weight in init_weight_method.keys()
        ), f"Select the weight initialization method in {list(init_weight_method.keys())}"
        init_bias_method = {
            "zero": nn.init.zeros_,
            "uniform": nn.init.uniform_,
        }
        assert(
            self.init_bias in init_bias_method.keys()
        ), f"Select the bias initialization method in {list(init_bias_method.keys())}"

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weight_method[self.init_weight](m.weight)
                init_bias_method[self.init_bias](m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init_weight_method[self.init_weight](m.weight)
                init_bias_method[self.init_bias](m.bias)


    def forward(self, X):
        return self.net(X)
