import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def getGaussianPE(src):
    imblur = cv.GaussianBlur(src, ksize=(5, 5), sigmaX=3)
    dx = cv.Sobel(imblur, cv.CV_16S, 1, 0)  
    dy = cv.Sobel(imblur, cv.CV_16S, 0, 1)
    E = dx**2 + dy**2
    return E

def getDiagCycleMat(alpha, beta, n):
    a = 2 * alpha + 6 * beta
    b = -(alpha + 4 * beta)
    c = beta
    diag_mat_a = a * np.eye(n)
    diag_mat_b = b * np.roll(np.eye(n), 1, 0) + b * np.roll(np.eye(n), -1, 0)
    diag_mat_c = c * np.roll(np.eye(n), 2, 0) + c * np.roll(np.eye(n), -2, 0)
    return diag_mat_a + diag_mat_b + diag_mat_c

def getCircleContour(centre=(0, 0), radius=(1, 1), N=200):
    t = np.linspace(0, 2 * np.pi, N)
    x = centre[0] + radius[0] * np.cos(t)
    y = centre[1] + radius[1] * np.sin(t)
    return np.array([x, y])

def getRectContour(pt1=(0, 0), pt2=(50, 50)):
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
    x, y, errs = snake[0].copy(), snake[1].copy(), []
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
    for g in range(max_iter):
        x_pre, y_pre = x.copy(), y.copy()
        i, j = np.uint8(y), np.uint8(x)
        try:
            xn = inv @ (gamma * x + fx[i, j])
            yn = inv @ (gamma * y + fy[i, j])
        except Exception as e:
            print("index out of boundary")
        x, y = xn, yn
        err = np.mean(0.5 * np.abs(x_pre - x) + 0.5 * np.abs(y_pre - y))
        errs.append(err)
        if err < convergence:
            print(f"Snake inter{g} times，approximate the result。\t err = {err:.3f}")
            break
    return x, y, errs

def main():
    src = cv.imread("circle.png", 0)
    img = cv.GaussianBlur(src, (3, 3), 5)
    h, w = img.shape
    init = getCircleContour((w//2, h//2), (w//4, h//4), N=800)
    x, y, errs = snake(img, snake=init, alpha=0.1, beta=1, gamma=0.1)

    plt.figure() 
    plt.imshow(img, cmap="gray")
    plt.plot(init[0], init[1], '--r', lw=1)
    plt.plot(x, y, 'g', lw=1)
    plt.xticks([]), plt.yticks([]), plt.axis("off")
    plt.figure() 
    plt.plot(range(len(errs)), errs)
    plt.show()

if __name__ == '__main__':
    main()
