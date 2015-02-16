__author__ = 'julenka'

import numpy as np
from matplotlib import pyplot as plt
from scipy import signal as sc

# When you convolve with moving average you get periodicity?
Random = np.random.normal(0,1,100)
plt.plot(np.arange(Random.size), Random)
plt.xlabel('samples')
plt.ylabel('random')
plt.show()

M = np.linspace(10,50,5)
x = np.random.randn(100,1)
Random_Conv = []

plot_rows = len(M) + 1
plt.subplot(plot_rows, 1, 1)
plt.ylabel('original')
plt.xlabel('samples')
plt.plot(x)
for i,m in enumerate(M):
    h = 1.0/m * np.ones(m).reshape(m, 1)
    print h.shape, x.shape
    convolved = sc.convolve2d(x, h, 'valid')
    Random_Conv.append(convolved)
    plt.subplot(plot_rows, 1, i + 2)
    plt.ylabel("M = %i" % m)
    plt.xlabel("samples")
    plt.plot(convolved)

plt.show()