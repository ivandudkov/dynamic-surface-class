import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

x = np.linspace(-6, 6, 25)
y = np.linspace(-6, 6, 25)

# General equation for an ellipsis
# x^2/a^2 + y^2/b^2 = 1
# or
# x^2/a^2 + y^2/b^2 - 1 = 0
#
# If our center has an offset:
# (x-x0)^2/a^2 + (y-y0)^2/b^2 = 1

# Creating cartesian coordinate space
# using meshgrid method
x, y = np.meshgrid(x, y)

# a should be a > b or a = b
a = 3
b = 2

xf1 = np.sqrt(a ** 2 - b ** 2)
yf1 = 0
xf2 = np.sqrt(a ** 2 - b ** 2) * -1
yf2 = 0

# Ellipsis equation
X = x ** 2 / a ** 2  # X-coordinate
Y = y ** 2 / b ** 2  # Y-coordinate

# Solution grid of the Ellipsis equation
solution = X + Y

# Parametrization
t = np.linspace(0, 2 * np.pi, 100)
x2 = a * np.cos(t)
y2 = b * np.sin(t)

# Plotting
plt.contour(x, y, solution, [1])
plt.scatter(x2, y2, marker='o')
plt.scatter(xf1, yf1, marker='o')
plt.scatter(xf2, yf2, marker='o')
plt.show()

# Example: https://stackoverflow.com/questions/25050899/producing-an-array-from-an-ellipse
import numpy as np
from matplotlib import pyplot
# Let's introduce some values of interest:
# Centers:
x0 = 4; a = 5  # x center and half width (greater axis of an ellipse)
y0 = 2; b = 3  # y center and half width (smaller axis of an ellipse)


x = np.linspace(-10, 10, 100)  # x - values
y = np.linspace(-5, 5, 100)[:,None]  # y - values as a "column" array

ellipse = ((x-x0)/a)**2 + ((y-y0)/b)**2 <= 1  # Boolean array, where True for points inside the ellipse

# Thanks to NumPy's broadcasting rules,
# the contributions of x and y are summed together
# in a simple way (y[:,None] essentially makes y
# a column vector of y value, while x remain a row vector).
# There is also no need for larger 2D intermediate array,
# as would be needed with numpy.meshgrid().

pyplot.imshow(ellipse, extent=(x[0], x[-1], y[0][0], y[-1][0]), origin="lower")
pyplot.show()
