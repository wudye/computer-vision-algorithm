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
# print(init)

"""
1. gaussian(img, 3)
作用: 对输入图像 img 进行高斯模糊处理
sigma=3: 高斯核的标准差为3，用于平滑图像，减少噪声影响
2. snake=init
作用: 蛇模型的初始轮廓
说明: 需要预先定义一个初始的轮廓点集作为迭代的起点
3. alpha=0.015
作用: 轮廓的弹性系数（控制轮廓的收缩力）
说明: 值越大，轮廓越趋向于收缩（变小），增加连续性约束
典型范围: 0.01-0.1
4. beta=1
作用: 轮廓的刚性系数（控制轮廓的弯曲度）
说明: 值越大，轮廓越平滑，减少曲率变化（避免轮廓过度弯曲）
典型范围: 0.1-2
5. gamma=0.001
作用: 步长参数（控制迭代速度和精度）
说明: 值越小，迭代越精确但收敛越慢；值越大，收敛快但可能不稳定
典型范围: 0.001-0.1
6. w_line=0
作用: 线能量权重（控制轮廓靠近图像亮线还是暗线）
说明: 0表示不使用线能量
正值: 轮廓趋向亮线
负值: 轮廓趋向暗线
7. w_edge=1
作用: 边缘能量权重（控制轮廓吸引到边缘的强度）
说明: 1表示使用边缘能量，轮廓会被强梯度（边缘）吸引
"""
# 优化方案：保持初始轮廓位置，调整参数以获得更好的收敛
# 方法1：低弹性，中等刚性，精确收敛
snake1 = active_contour(gaussian(img, 3), snake=init, alpha=0.01, beta=1, gamma=0.001, w_line=0, w_edge=5, max_num_iter=2500)
# 方法2：中等弹性，低刚性，强边缘吸引
snake2 = active_contour(gaussian(img, 3), snake=init, alpha=0.05, beta=0.1, gamma=0.01, w_line=0, w_edge=15, max_num_iter=2500)
# 方法3：标准配置
snake3 = active_contour(gaussian(img, 3), snake=init, alpha=0.015, beta=10, gamma=0.001, w_line=0, w_edge=10, max_num_iter=2500)

# 绘制对比图
fig, axes = plt.subplots(1, 3, figsize=(15, 15))

# 方法1
axes[0].imshow(img, cmap=plt.cm.gray)
axes[0].plot(init[:, 0], init[:, 1], '--r', lw=2)
axes[0].plot(snake1[:, 0], snake1[:, 1], '-b', lw=2)
axes[0].set_title('Method 1: Low Elasticity\n(α=0.01, β=1, γ=0.001, w_edge=5)')
axes[0].axis('off')

# 方法2
axes[1].imshow(img, cmap=plt.cm.gray)
axes[1].plot(init[:, 0], init[:, 1], '--r', lw=2)
axes[1].plot(snake2[:, 0], snake2[:, 1], '-b', lw=2)
axes[1].set_title('Method 2: Flexible + Strong Edge\n(α=0.05, β=0.1, γ=0.01, w_edge=15)')
axes[1].axis('off')

# 方法3
axes[2].imshow(img, cmap=plt.cm.gray)
axes[2].plot(init[:, 0], init[:, 1], '--r', lw=2)
axes[2].plot(snake3[:, 0], snake3[:, 1], '-b', lw=2)
axes[2].set_title('Method 3: Standard\n(α=0.015, β=10, γ=0.001, w_edge=10)')
axes[2].axis('off')

plt.tight_layout()
plt.show()