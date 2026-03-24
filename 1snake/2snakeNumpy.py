import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path

def getGaussianPE(src):
    """
    Transform the image gradient information into an external
     energy field to guide the Snake algorithm in locating object edges.
    origial img：
    80  90  85
    95  100 92
    88  93  87
    3 * 3 kernel:
    -1  0  1
    -2  0  2
    -1  0  1
    convolution result:
    dx = (-1×80) + (0×90) + (1×85) +
     (-2×95) + (0×100) + (2×92) +
     (-1×88) + (0×93) + (1×87)

       = -80 + 0 + 85
         -190 + 0 + 184
         -88 + 0 + 87

       = 5 - 6 - 1
       = -2
    dx = -2 express the pixel to be small along the  right side of the image 表示像素向右变暗
    dx < 0：表示向右变暗（左边更亮）✅
    dx > 0：表示向右变亮（右边更亮）✅
    dx = 0：局部平坦或对称区域，x 向梯度弱 ✅

    dx = -2（向右变暗）
        ↓
    E = dx² = 4（边缘强度）
       ↓
    E_ext = -E = -4（外部能量）
    外部能量场 E_ext 的分布：
    区域	            原始梯度	        E = dx²	    E_ext = -E	能量高低
    左边（边缘）	    大（如2）	        4	        -4	        最低 ✓
    中间	            小（如0.5）	    0.25	    -0.25	    较高
    右边（平坦）	    很小（如0.1）	    0.01	    -0.01	    最高
    重新理解 fx 的方向
    如果轮廓点在边缘和中间之间：
    位置：     边缘(E_ext=-4) ←── 轮廓点 ←── 中间(E_ext=-0.25)
                              ↑
                           计算fx的地方
    fx = 右边能量 - 左边能量
   = (-0.25) - (-4)
   = +3.75
   fx = +3.75 的含义：
    能量向右增加
    轮廓受向左的力（向更低能量）
    轮廓向边缘移动 ✓
    fx值	能量变化	轮廓受力方向	移动目标
    +2	向右增加	向左	向左边缘
    -2	向右递减	向右	向右边缘
    0	平坦	无力	停留

    """
    imblur = cv.GaussianBlur(src, ksize=(5, 5), sigmaX=3)
    dx = cv.Sobel(imblur, cv.CV_16S, 1, 0)  # X方向上取一阶导数，16位有符号数，卷积核3x3
    dy = cv.Sobel(imblur, cv.CV_16S, 0, 1)
    E = dx**2 + dy**2
    return E

# def getDiagCycleMat(alpha, beta, n):
    """ 这个函数计算的是 Snake 模型的内部能量矩阵，用于：

    弹性能量（Elastic energy）→ 由 alpha 参数控制
    刚性能量（Rigidity energy）→ 由 beta 参数控制
    总能量 = 内部能量 + 外部能量
    E_total = E_internal + E_external

    E_internal = E_elastic + E_rigidity
               = α|v'|² + β|v''|²
    其中：

    α (alpha)：弹性系数，控制轮廓拉伸
    β (beta)：刚性系数，控制轮廓弯曲

    5对角循环矩阵的结构
    主对角线：    a, a, a, ..., a
    第一副对角线：  b, b, b, ..., b
    第二副对角线：  c, c, c, ..., c
    具体矩阵示例（N=6）：
    [ a  b  c  0  c  b ]
    [ b  a  b  c  0  c ]
    [ c  b  a  b  c  0 ]
    [ 0  c  b  a  b  c ]
    [ c  0  c  b  a  b ]
    [ b  c  0  c  b  a ]
    参数	含义	物理意义
    alpha	弹性系数	控制轮廓拉伸（连续性）
    beta	刚性系数	控制轮廓弯曲（平滑性）
    n	轮廓点数	矩阵维度 n×n
    主对角线系数 a：
    a = 2α + 6β

    来源：
    - 弹性项贡献：2α（来自一阶导数平方）
    - 刚性项贡献：6β（来自二阶导数平方）
    第一副对角线系数 b：
    b = -(α + 4β)

    来源：
    - 弹性项贡献：-α
    - 刚性项贡献：-4β
    第二副对角线系数 c：
    c = β
    来源：
    - 刚性项贡献：β

    np.roll(np.eye(n), 1, 0)：

    向下滚动1行
    主对角线移到上方




    """
  #  a = 2 * alpha + 6 * beta
  #  b = -(alpha + 4 * beta)
  #  c = beta
    """
    生成 n×n 单位矩阵乘以 a：
    [ a  0  0  ...  0 ]
    [ 0  a  0  ...  0 ]
    [ 0  0  a  ...  0 ]
    [ ...          ...]
    [ 0  0  0  ...  a ]
    """
# diag_mat_a = a * np.eye(n)

    """
    np.roll(np.eye(n), 1, 0)：
    
    向下滚动1行
    主对角线移到上方
    np.roll(np.eye(n), -1, 0)：
    
    向上滚动1行
    主对角线移到下方
    [ 0  1  0  0  ...  1 ]
    [ 1  0  1  0  ...  0 ]
    [ 0  1  0  1  ...  0 ]
    [ 0  0  1  0  ...  0 ]
    [ ...          ...]
    [ 1  0  0  0  ...  1 ]  ← 循环特性

    """
    # diag_mat_b = b * np.roll(np.eye(n), 1, 0) + b * np.roll(np.eye(n), -1, 0)
    """"[0  0  1  0...  0  1]
    [1  0  0  1...  0  0]
    [0  1  0  0...  0  0]
    [0  0  1  0...  0  0]
    [......]
    [0  1  0  0...  0  0]
    [1  0  0  1...  0  0]
    """
    # diag_mat_c = c * np.roll(np.eye(n), 2, 0) + c * np.roll(np.eye(n), -2, 0)
    """
   最终矩阵（alpha=0.5, beta=0.1, n=6）：
    a = 2*0.5 + 6*0.1 = 1.6
    b = -(0.5 + 4*0.1) = -0.9
    c = 0.1
    
    [ 1.6 -0.9  0.1  0.0  0.1 -0.9 ]
    [-0.9  1.6 -0.9  0.1  0.0  0.1 ]
    [ 0.1 -0.9  1.6 -0.9  0.1  0.0 ]
    [ 0.0  0.1 -0.9  1.6 -0.9  0.1 ]
    [ 0.1  0.0  0.1 -0.9  1.6 -0.9 ]
    [-0.9  0.1  0.0  0.1 -0.9  1.6 ]
   """
    # return diag_mat_a + diag_mat_b + diag_mat_c

def getDiagCycleMat(alpha, beta, n):
    """ 计算5对角循环矩阵 """
    a = 2 * alpha + 6 * beta
    b = -(alpha + 4 * beta)
    c = beta
    diag_mat_a = a * np.eye(n)
    diag_mat_b = b * np.roll(np.eye(n), 1, 0) + b * np.roll(np.eye(n), -1, 0)
    diag_mat_c = c * np.roll(np.eye(n), 2, 0) + c * np.roll(np.eye(n), -2, 0)
    return diag_mat_a + diag_mat_b + diag_mat_c

def getCircleContour(centre=(0, 0), radius=(1, 1), N=200):
    """ 以参数方程的形式，获取n个离散点围成的圆形/椭圆形轮廓 输入：中心centre=（x0, y0）, 半轴长radius=(a, b)， 离散点数N 输出：由离散点坐标(x, y)组成的2xN矩阵 """
    t = np.linspace(0, 2 * np.pi, N)
    x = centre[0] + radius[0] * np.cos(t)
    y = centre[1] + radius[1] * np.sin(t)
    return np.array([x, y])

def getRectContour(pt1=(0, 0), pt2=(50, 50)):
    """ 根据左上、右下两个顶点来计算矩形初始轮廓坐标 由于Snake模型适用于光滑曲线，故这里用不到该函数 """
    pt1, pt2 = np.array(pt1), np.array(pt2)
    r1, c1, r2, c2 = pt1[0], pt1[1], pt2[0], pt2[1]
    a, b = r2 - r1, c2 - c1
    length = (a + b) * 2 + 1
    x = np.ones((length), np.float)
    x[:b] = r1
    x[b:a + b] = np.arange(r1, r2)
    x[a + b:a + b + b] = r2
    x[a + b + b:] = np.arange(r2, r1 - 1, -1)
    y = np.ones((length), np.float)
    y[:b] = np.arange(c1, c2)
    y[b:a + b] = c2
    y[a + b:a + b + b] = np.arange(c2, c1, -1)
    y[a + b + b:] = c1
    return np.array([x, y])

def snake(img, snake, alpha=0.5, beta=0.1, gamma=0.1, max_iter=2500, convergence=0.01):
    """ 根据Snake模型的隐式格式进行迭代 输入：弹力系数alpha，刚性系数beta，迭代步长gamma，最大迭代次数max_iter，收敛阈值convergence 输出：由收敛轮廓坐标(x, y)组成的2xN矩阵， 历次迭代误差list """
    x, y, errs = snake[0].copy(), snake[1].copy(), []
    n = len(x)
    # 计算5对角循环矩阵A，及其相关逆阵
    A = getDiagCycleMat(alpha, beta, n)
    """
    A · v = F_internal
    A · v = 内部力
    其中 v 是轮廓点坐标向量
    参数组合	特点	适用场景
    大alpha，大beta	强拉伸、强平滑	平滑边界，噪声图像
    大alpha，小beta	强拉伸、允许转折	多边形物体
    小alpha，大beta	允许伸缩、强平滑	弯曲边界，噪声图像
    小alpha，小beta	弱约束	自由变形
    Snake 算法的迭代公式：

    (A + γI) · v_new = γv_old + F_ext

    解方程：
    v_new = (A + γI)⁻¹ · (γv_old + F_ext)
            ↑
            这就是 inv
    """
    # gamma	迭代步长	控制收敛速度,  np.linalg.inv() - 矩阵求逆
    inv = np.linalg.inv(A + gamma * np.eye(n))

    # 初始化
    y_max, x_max = img.shape
    max_px_move = 1.0
    # 计算负高斯势能矩阵，及其梯度
    E_ext = -getGaussianPE(img)
    fx = cv.Sobel(E_ext, cv.CV_16S, 1, 0)
    fy = cv.Sobel(E_ext, cv.CV_16S, 0, 1)
    T = np.max([abs(fx), abs(fy)])
    fx, fy = fx / T, fy / T
    for g in range(max_iter):
        x_pre, y_pre = x.copy(), y.copy()
        # 限制坐标在图像范围内
        """
        对于每个元素：
        if value < min:
            value = min
        elif value > max:
            value = max
        else:
            value = value  # 保持不变
        """
        x_clipped = np.clip(x, 0, x_max - 1)
        y_clipped = np.clip(y, 0, y_max - 1)
        # 转换为8位无符号整数	(0 -255)
        i, j = np.uint8(y_clipped), np.uint8(x_clipped)

        """
        Snake 迭代核心公式详细解释
        @	矩阵乘法	应用内部约束
        gamma * x	惯性项	保持当前位置的趋势
        fx[i, j], fy[i, j]	外力	向边缘收敛的力
        gamma * x + fx	总力向量	惯性 + 外力
        inv	逆矩阵	内部能量约束
        xn, yn	新轮廓点坐标	下一时刻位置
        E_internal = α|v'|² + β|v''|²
        E_external = -|∇I|²
        能量最小化
        目标：找到 v 使 E 最小
        ∂E/∂v = ∂E_internal/∂v + ∂E_external/∂v = 0
        离散化方程
        时间演化方程：
        γ(v_new - v_old) + A·v_new + F_ext = 0
        (A + γI)·v_new = γ·v_old + F_ext
        v_new = (A + γI)⁻¹ · (γ·v_old + F_ext)
                   ↑           ↑          ↑
                   inv         inertia   external_force
         代码实现
        xn = inv @ (gamma * x + fx[i, j])
        yn = inv @ (gamma * y + fy[i, j]
        """
        try:
            xn = inv @ (gamma * x + fx[i, j])
            yn = inv @ (gamma * y + fy[i, j])
        except Exception as e:
            print(f"索引超出范围: {e}")
            break
        # 判断收敛
        x, y = xn, yn
        err = np.mean(0.5 * np.abs(x_pre - x) + 0.5 * np.abs(y_pre - y))
        errs.append(err)
        if err < convergence:
            print(f"Snake迭代{g}次后，趋于收敛。\t err = {err:.3f}")
            break
    return x, y, errs

def main():
    # 0 for grayscale, 1 for color
    src = cv.imread("./circle.png", 0)
    if src is None:
        raise FileNotFoundError(f"Failed to decode image file")
    print(src.shape)
    print(type(src))


    """
    高斯模糊是线性的，两次模糊等价于一次更大核的模糊：
    GaussianBlur₁ × GaussianBlur₂ = GaussianBlur₃
    σ_total² = σ₁² + σ₂²
    次数	核大小	sigmaX	说明
    第1次（main）	(3, 3)	5	主要模糊
    第2次（getGaussianPE）	(5, 5)	3	再次模糊
    等价单次	≈ (7, 7)	≈ 5.83	综合效果
    σ_total = √(5² + 3²) = √(25 + 9) = √34 ≈ 5.83
    """

    img = cv.GaussianBlur(src, (3, 3), 5)
    # img = src.copy()
    # 构造初始轮廓线
    # 根据图像大小调整初始轮廓位置
    h, w = img.shape
    init = getCircleContour((w//2, h//2), (w//4, h//4), N=800)
    print(w//4, w/2, w//2)
    # Snake Model - 使用更小的gamma避免快速收敛导致越界
    x, y, errs = snake(img, snake=init, alpha=0.05, beta=1, gamma=0.01)

    plt.figure() # 绘制轮廓图
    plt.imshow(img, cmap="gray")
    plt.plot(init[0], init[1], '--r', lw=1)
    plt.plot(x, y, 'g', lw=1)
    plt.xticks([]), plt.yticks([]), plt.axis("off")
    plt.figure() # 绘制收敛趋势图
    plt.plot(range(len(errs)), errs)
    plt.show()

if __name__ == '__main__':
    main()
