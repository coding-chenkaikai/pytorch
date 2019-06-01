# -*- coding: UTF-8 -*-

import torch
import numpy as np

import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.autograd import Variable

print(torch.__version__)
print('gpu: ', torch.cuda.is_available())

