# -*- coding: utf-8 -*-
"""pytorch classifier.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1KRxKTF9vrsyC9eDQQ6NjtMAWu1Z_L6Lu
"""

import torch
import torchvision
from torch.utils.mobile_optimizer import optimize_for_mobile

model = torchvision.models.resnet18(pretrained=True)
model.eval()
example = torch.rand(1, 3, 244, 244)
traced_script_module = torch.jit.trace(model, example)
optimized_traced_model = optimize_for_mobile(traced_script_module)
optimized_traced_model._save_for_lite_interpreter("app/src/main/assets/model.pt")