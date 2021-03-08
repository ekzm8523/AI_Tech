import torch
import numpy as np
import random
from network import VGG11
import base64, copy

torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
random.seed(0)

import warnings
warnings.filterwarnings('ignore')

class Calculator:
    def __init__(self, model):
        self.answer = 0
        layers = [b'Y29udjNfMQ==\n', b'cG9vbDM=\n', b'Y29udjVfMg==\n']
        for l in layers:
            self.hook = model._modules[base64.decodebytes(l).decode()].register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.answer += self._get_answer(output)

    def _get_answer(self, layer):
        _, A, B, C = layer.shape
        return A * (B - C // 3)

    def unregister_forward_hook(self):
        self.hook.remove()


def calc_answer(model):
    model_test = copy.deepcopy(model)
    ans_calculator = Calculator(model_test)

    x = torch.rand(1,3,224,224)
    model_test(x)
    print("Your answer is : %d" % ans_calculator.answer)


if __name__ == "__main__":
    model = VGG11(num_classes=1000)
    x = torch.randn((1,3,224,224))

    out = model(x)

    print("Output tensor shape is :", out.shape)
    calc_answer(model)
