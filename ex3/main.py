import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, img_as_ubyte, util, filters
import time
from scipy.signal import correlate2d, medfilt2d
import math

start_time = time.time()

img = io.imread("./img/lena.png")
img_gray = img_as_ubyte(color.rgb2grey(img))


def correl2d(img_in_method, window):
    """
    使用滤波器实现图像的空间相关
    mode='same' 表示输出尺寸等于输入尺寸
    boundary=‘fill’ 表示滤波前，用常量值填充原始图像的边缘，默认常量值为0

    :param img_in_method:
    :param window:
    :return:
    """
    s = correlate2d(img_in_method, window, mode='same', boundary='fill')
    return s.astype(np.uint8)


def mean_filtering_and_compare_each_size_filter(img_gray_in_method):
    mean_filter1 = np.ones((3, 3)) / (3 ** 2)
    mean_filter2 = np.ones((5, 5)) / (5 ** 2)
    mean_filter3 = np.ones((9, 9)) / (9 ** 2)
    img1 = correl2d(img_gray_in_method, mean_filter1)
    img2 = correl2d(img_gray_in_method, mean_filter2)
    img3 = correl2d(img_gray_in_method, mean_filter3)
    plt.subplot(131)
    io.imshow(img1)
    plt.title("mean_filtering 3x3")
    plt.subplot(132)
    io.imshow(img2)
    plt.title("mean_filtering 5x5")
    plt.subplot(133)
    io.imshow(img3)
    plt.title("mean_filtering 9x9")
    plt.show()


def gauss(i, j, sigma):
    return 1 / (2 * math.pi * sigma ** 2) * math.exp(-(i ** 2 + j ** 2) / (2 * sigma ** 2))


def gauss_filter(radius, sigma):
    window = np.zeros((radius * 2 + 1, radius * 2 + 1))
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            window[i + radius][j + radius] = gauss(i, j, sigma)
    return window / np.sum(window)


def gauss_filtering_and_compare_each_size_filter(img_gray_in_method):
    gauss_filter1 = gauss_filter(3, 1.0)
    gauss_filter2 = gauss_filter(5, 1.0)
    gauss_filter3 = gauss_filter(9, 1.0)
    # 生成滤波结果
    img1 = correl2d(img_gray_in_method, gauss_filter1)
    img2 = correl2d(img_gray_in_method, gauss_filter2)
    img3 = correl2d(img_gray_in_method, gauss_filter3)
    plt.subplot(131)
    io.imshow(img1)
    plt.title("gauss_filtering 3x3")
    plt.subplot(132)
    io.imshow(img2)
    plt.title("gauss_filtering 5x5")
    plt.subplot(133)
    io.imshow(img3)
    plt.title("gauss_filtering 9x9")
    plt.show()


def median_filtering(img_gray_in_method, filter_size):
    return medfilt2d(img_gray_in_method, filter_size)


def compare_different_type_filter(img_gray):
    noise_img = util.random_noise(img_gray, mode='s&p', seed=None, clip=True)
    after_median_filter_img = median_filtering(noise_img, 3)
    after_mean_filter_img = correlate2d(noise_img, np.ones((3, 3)) / (3 ** 2))
    after_gauss_filter_img = correlate2d(noise_img, gauss_filter(3, 1.0))
    plt.subplot(224)
    io.imshow(noise_img)
    plt.title("noise_img")
    plt.subplot(223)
    io.imshow(after_median_filter_img)
    plt.title("after_median_filter_img")
    plt.subplot(222)
    io.imshow(after_mean_filter_img)
    plt.title("after_mean_filter_img")
    plt.subplot(221)
    io.imshow(after_gauss_filter_img)
    plt.title("after_gauss_filter_img")
    plt.show()


def sobel_filtering(img_in_method, threshold, sobel_filter):
    sobel_img = correlate2d(img_in_method, sobel_filter, mode='same', boundary='fill')
    sobel_img[sobel_img < threshold] = 0
    return sobel_img.astype(np.uint8)


def save_and_show_sobel_v_and_sobel_h(img_gray_in_method):
    g_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_filter_v = sobel_filtering(img_gray_in_method.copy(), 100, g_x)
    g_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    sobel_filter_h = sobel_filtering(img_gray_in_method.copy(), 100, g_y)
    plt.subplot(131)
    io.imshow(sobel_filter_v)
    plt.title("sobel_filter_v")
    plt.subplot(132)
    io.imshow(sobel_filter_h)
    plt.title("sobel_filter_h")
    plt.subplot(133)
    sobel_img = (abs(sobel_filter_v.astype(np.float)) + abs(sobel_filter_h.astype(np.float))).astype(np.uint8)
    io.imshow(sobel_img)
    plt.title("sobel")
    plt.show()

    io.imsave("./img/lena_sobel_v.png", sobel_filter_v)
    io.imsave("./img/lena_sobel_h.png", sobel_filter_h)
    io.imsave("./img/lena_sobel.png", sobel_img)


def laplace_edge(img_in_method):
    laplace_filter = np.array([
        [1, 1, 1],
        [1, -8, 1],
        [1, 1, 1],
    ])

    img_in_method = np.pad(img_in_method, (1, 1), mode='constant', constant_values=0)
    m, n = img_in_method.shape
    output_image = correlate2d(img_in_method, laplace_filter, mode='same', boundary='fill')
    output_image = output_image[1:m - 1, 1:n - 1]
    return output_image


def laplace_sharpen(img_gray_in_method):
    plt.subplot(131)
    io.imshow(img_gray_in_method)
    plt.title("img_gray")

    plt.subplot(132)
    laplace_edge_img = laplace_edge(img_gray_in_method)
    io.imshow(laplace_edge_img)
    plt.title("laplace_edge_img")

    plt.subplot(133)
    c = 0.5
    laplace_sharpen_img = img_gray_in_method + c * laplace_edge_img
    laplace_sharpen_img[laplace_sharpen_img <= 0] = 0
    laplace_sharpen_img[laplace_sharpen_img >= 255] = 255
    laplace_sharpen_img = laplace_sharpen_img.astype(np.uint8)
    io.imshow(laplace_sharpen_img)
    plt.title("laplace_sharpen_img")
    plt.show()

    io.imsave("./img/lena_laplace_edge_img.png", laplace_edge_img)
    io.imsave("./img/lena_laplace_sharpen_img.png", laplace_sharpen_img)


mean_filtering_and_compare_each_size_filter(img_gray.copy())

gauss_filtering_and_compare_each_size_filter(img_gray.copy())

compare_different_type_filter(img_gray.copy())

save_and_show_sobel_v_and_sobel_h(img_gray.copy())

laplace_sharpen(img_gray.copy())
