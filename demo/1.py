import torch.nn.functional as F
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

x = torch.linspace(-5, 5, 200)
x = Variable(x)
x_np = x.data.numpy()

y1 = F.relu(x).data.numpy()

plt.figure(1, figsize=(8, 6))
plt.subplot(221)
plt.plot(x_np, y1, c='red', label='relu')
plt.ylim((-1, 5))
plt.legend(loc='best')
plt.show()
