from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

start_time = time.time()


def Basis(img):
    img_height, img_width = img.shape
    count = np.zeros(256, np.float)
    for i in range(0, img_width):
        for j in range(0, img_height):
            pixel = int(img[i, j] * 255)
            count[pixel] = count[pixel] + 1
    return count


def Histogram(Probability_Density=False):
    img = io.imread('./img/lena.png', as_gray=True)
    img_height, img_width = img.shape
    count = Basis(img)
    pro = count / (img_height * img_width)
    x = np.linspace(0, 255, 256)
    y = count
    _, ax = plt.subplots(1, 2)
    ax[0].bar(x, height=y)
    ax[0].set_title('Histogram of grayscale image')
    ax[0].set_xlabel('Pixel')
    ax[0].set_ylabel('Number of pixels')
    ax[0].grid(axis='y', linestyle='--')
    if Probability_Density:
        ax[1].plot(x, pro, 'r-', label='Probability_Density')
    plt.legend()
    plt.show()


def image_information():
    img = io.imread('./img/lena.png')
    img_mean = np.mean(img)
    img_var = np.var(img)
    img_std = np.std(img)
    img_e = entropy_gray()
    print("Mean=", img_mean, "\nvariance=", img_var, "\nstandard deviation=", img_std, "\nentropy=", img_e)


def entropy_gray():
    img = io.imread('./img/lena.png', as_gray=True)
    img_height, img_width = img.shape
    count = Basis(img) + 1e-8
    count = count / (img_height * img_width)
    h = 0
    for i in range(0, 255):
        h = h + (-1.0) * np.sum(count[i] * np.log2(count[i]))
    return h


def Histogram_equ():
    img = io.imread('./img/lena.png', as_gray=True)
    img_height, img_width = img.shape
    imgs = np.zeros_like(img)
    counts = np.zeros(256, np.float)
    equ = np.zeros(256, np.float)
    count = Basis(img)
    pro = count / (img_height * img_width)
    sum = 0
    for i in range(0, 255):
        sum = sum + pro[i]
        equ[i] = np.around(255 * sum)
    for i in range(0, img_width):
        for j in range(0, img_height):
            imgs[i, j] = equ[int(img[i, j] * 255)]
            pixel = int(imgs[i, j])
            counts[pixel] = counts[pixel] + 1
    x = np.linspace(0, 255, 256)
    y1 = count
    y2 = counts
    _, ax = plt.subplots(2, 2)
    ax0, ax1, ax2, ax3 = ax.ravel()
    ax0.imshow(img, plt.cm.gray)
    ax1.bar(x, height=y1)
    ax2.imshow(imgs, plt.cm.gray)
    ax3.bar(x, height=y2)
    plt.show()


def Quantify(bit):
    img = io.imread('./img/lena.png', as_gray=True)
    temp = 256 / (2 ** bit)
    inter_pixel = np.floor(img * 255 / temp)
    Transform(inter_pixel, bit, temp)

def Transform(pixel, bit, temp):
    new_img = np.multiply(pixel, temp)
    new_img = np.array(new_img, np.uint8)
    io.imsave('lena_trans_%d.png' % bit, new_img)


if __name__ == "__main__":
    Histogram(True)
    image_information()
    Histogram_equ()
    trans_list = [1, 3, 6]
    for bit in tqdm(trans_list):
        Quantify(bit)

    print(time.time() - start_time)
