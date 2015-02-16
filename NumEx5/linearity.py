__author__ = 'julenka'
import numpy as np
from scipy import constants as c
from matplotlib import pyplot as plt

alpha = 0.999
h = pow(alpha, np.arange(100))
y = np.linspace(0,999,1000)

x1 = np.sin((y*2*c.pi*40)/1000)
y1 = np.convolve(x1,h, 'valid')

plt.subplot(1,3,1)
plt.title("h")
plt.plot(h)
plt.subplot(3,3,2)
plt.title("x1")
plt.plot(x1)
plt.subplot(3,3,3)
plt.title("x1 * h")
plt.plot(y1)

x2 = np.sin((y*2*c.pi*80)/1000)
y2 = np.convolve(x2,h,'valid')


plt.subplot(3,3,5)
plt.title("x2")
plt.plot(x2)
plt.subplot(3,3,6)
plt.title("x2 * h")
plt.plot(y2)

# if system is linear, ax1 + bx2 must be equal to ay1 + by2
a = 2
b = -10
xx = a * x1 + b * x2
yy = np.convolve(xx,h,'valid')


plt.subplot(3,3,8)
plt.title("h * (ax1 + bx2)")
plt.plot(yy)
plt.subplot(3,3,9)
plt.title("ay1 + by2")
plt.plot(a * y1 + b * y2)

error = abs(yy - (a*y1 + b*y2))
maxError = max(error)
print maxError
plt.show()

