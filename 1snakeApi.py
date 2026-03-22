import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage import data
from skimage.filters import gaussian
from skimage.segmentation import active_contour

img = data.astronaut()
img = rgb2gray(img)

# circle (x, y) and radius r of the initial snake
# (220, 100) and r=100
t = np.linspace(0, 2*np.pi, 400)
x = 220 + 100 * np.cos(t)
y = 100 + 100 * np.sin(t)

# construct the snake and display it
init = np.array([x, y]).T
print(init.shape)
print(init)

snake = active_contour(gaussian(img, 3), snake=init, alpha=0.015, beta=1, gamma=0.001, w_line=0, w_edge=1)
plt.figsize(7, 7)
plt.imshow(img, cmap=plt.cm.gray)
plt.plot(init[:, 0], init[:, 1], '--r', lw=3)
plt.plot(snake[:, 0], snake[:, 1], '-b', lw=3)
plt.title('Active Contour Model')
plt.xticks([]), plt.yticks([]), plt.axis('off')
plt.show()