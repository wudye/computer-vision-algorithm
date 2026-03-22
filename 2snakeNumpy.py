import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def getCircleContour(center=(0,0), radius=(1, 1), N = 100):

    t = np.linspace(0, 2*np.pi, N)
    x = center[0] + radius[0] * np.cos(t)
    y = center[1] + radius[1] * np.sin(t)
    return np.array([x, y]).T

# 4 times differentiable energy function
def getDiagCycleMat(alpha, beta, n):
    a = 2 * alpha + 6 * beta
    b = -(alpha + 4 * beta)
    c = beta

    diag_mat_a = a * np.eye(n)
    diag_mat_b = b * np.roll(np.eye(n), 1, 0) + b * np.roll(np.eye(n), -1, 0)
    diag_mat_c = c * np.roll(np.eye(n), 2, 0) + c * np.roll(np.eye(n), -2, 0)
    return diag_mat_a + diag_mat_b + diag_mat_c


def getGaussianPE(src):
    imblur = cv.GaussianBlur(src, (5, 5), 3)
    dx = cv.Sobel(imblur, cv.CV_16S, 1, 0)
    dy = cv.Sobel(imblur, cv.CV_16S, 0, 1)
    E = dx**2 + dy**2
    return E

def snake(img, snake, alpha=0.5, beta=0.1, gamma=0.1, max_iterations=2500, convergence=0.01):
    x, y,  errs = snake[0].copy(), snake[1].copy(), []
    n = len(x)
    A = getDiagCycleMat(alpha, beta, n)
    inv = np.linalg.inv(A + gamma * np.eye(n))

    y_max, x_max = img.shape
    max_px_move = 1.0

    E_ext = -getGaussianPE(img)
    fx = cv.Sobel(E_ext, cv.CV_16S, 1, 0)
    fy = cv.Sobel(E_ext, cv.CV_16S, 0, 1)
    T = np.max([abs(fx), abs(fy)])
    fx, fy = fx / T, fy / T
    
    for g in range(max_iterations):
        x_prev, y_prev = x.copy(), y.copy()
        i,j = np.uint8t(y), np.uint8t(x)
        try:
            xn = inv @ (gamma * x + fx[i, j])
            yn = inv @ (gamma * y + fy[i, j])
        except IndexError:
            print('Snake went out of bounds.')
            break
        x, y = xn, yn
        err = np.mean(0.5 * np.abs(x - x_prev) + 0.5 * np.abs(y - y_prev))
        errs.append(err)
        if err < convergence:
            print(f'Converged after {g} iterations.')
            break
    return x, y, errs


def main():
    src = cv.imread('circle.png')
    img = cv.GaussianBlur(src, (3, 3), 5)

    init = getCircleContour((140, 95), (110, 80), N = 200)

    x, y, errs = snake(img, init, alpha=0.1,  beta=1, gamma=0.001, w_line=0, w_edge=1, max_iterations=0.1)
    plt.figure(figsize=(7, 7))
    plt.imshow(img, cmap=plt.cm.gray)
    plt.plot(init[0], init[1], '--r', lw=1)
    plt.plot(x, y, 'g', lw=1)
    plt.xticks([]), plt.yticks([]), plt.axis("off")
    plt.figure() # 绘制收敛趋势图
    plt.plot(range(len(errs)), errs)
    plt.show()

if __name__ == '__main__':
    main()
