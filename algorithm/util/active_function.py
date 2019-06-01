# -*- coding: UTF-8 -*-

import torch
import numpy as np

import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable

x = Variable(torch.linspace(-5, 5, 100))
x_np = x.data.numpy()
y_sigmoid = torch.sigmoid(x).data.numpy()
y_tanh = torch.tanh(x).data.numpy()
y_relu = torch.relu(x).data.numpy()

x_leakyrelu = Variable(torch.linspace(-50, 50, 100))
y_leakyrelu = F.leaky_relu(x_leakyrelu).data.numpy()

plt.figure(1, figsize=(8, 6))
plt.subplot(221)
plt.plot(x_np, y_sigmoid, 'r-', label='sigmoid')
plt.ylim((-0.2, 1.2))
plt.legend(loc='upper left')
plt.grid(True)

plt.subplot(222)
plt.plot(x_np, y_tanh, 'r-', label='tanh')
plt.ylim((-1.2, 1.2))
plt.legend(loc='upper left')
plt.grid(True)

plt.subplot(223)
plt.plot(x_np, y_relu, 'r-', label='relu')
plt.ylim((-1, 5))
plt.legend(loc='upper left')
plt.grid(True)

plt.subplot(224)
plt.plot(x_leakyrelu.data.numpy(), y_leakyrelu, 'r-', label='leakyrelu')
plt.ylim((-1, 5))
plt.legend(loc='upper left')
plt.grid(True)
plt.show()