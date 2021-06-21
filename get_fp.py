
from torchstat import stat
import torchvision.models as models
model = models.resnet101()
stat(model, (3, 512, 512))

from torchvision.models import resnet101, vgg16_bn
from thop import profile
import torch
from thop import clever_format
model = resnet101()
input = torch.randn(1, 3, 512, 512)
macs, params = profile(model, inputs=(input, ))
macs, params = clever_format([macs, params], "%.3f")
print(macs)
print(params)

from ptflops import get_model_complexity_info
flops, params = get_model_complexity_info(model, (3, 512, 512), as_strings=True, print_per_layer_stat=True)
print('Flops:  ' + flops)
print('Params: ' + params)