import numpy as np
import matplotlib.pyplot as plt

theta = np.arange(0, 2*np.pi, 0.01)
a = 1
b = 2

xpos = a*np.cos(theta)
ypos = b*np.cos(theta)

new_xpos = xpos*np.cos(np.pi/2)+ypos*np.sin(np.pi/2)
new_ypos = -xpos*np.sin(np.pi/2)+ypos*np.cos(np.pi/2)

plt.plot(xpos, ypos, 'b-')
plt.plot(new_xpos, new_ypos, 'r-')

plt.show()
