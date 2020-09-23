import numpy as np
import math
import matplotlib.pyplot as plt
from skimage import io, color, img_as_ubyte
import time

# -------------------- the promote after code refactor --------------------#
# old program cost time: 5.2849647998809814
# refactor program cost time: 2.3496077060699463
# -------------------- the promote after code refactor --------------------#

start_time = time.time()

img = io.imread("./img/lena.png")
img_gray = img_as_ubyte(color.rgb2grey(img))


def count_histogram_andPDF(count_img):
    hist, bins = np.histogram(count_img.ravel(), 256, [0, 256])
    pdf = hist / count_img.ravel().shape
    return hist, pdf


def count_gray_img_average(count_img):
    x, y = count_img.shape
    row_sum = np.ones(x, dtype=int)
    col_sum = np.ones(y, dtype=int)

    count_average = count_img.dot(row_sum).dot(col_sum) / (x * y)
    return round(count_average)


def count_gray_img_variance(count_img):
    count_average = count_gray_img_average(count_img)
    x, y = count_img.shape

    count_img = count_img.astype(int)
    count_variance = (count_img - count_average) * (count_img - count_average)
    row_sum = np.ones(x, dtype=int)
    col_sum = np.ones(y, dtype=int)
    count_variance = count_variance.dot(row_sum).dot(col_sum) / (x * y)

    return count_variance


def count_gray_img_standard_deviation(count_img):
    count_standard_deviation = math.sqrt(count_gray_img_variance(count_img))
    return count_standard_deviation


def count_gray_img_entropy(pdf):
    pdf_masked = np.ma.masked_less_equal(pdf, 0)
    sum_arr = np.ones(pdf.shape, dtype=int)

    count_entropy_mask = (np.log2(pdf_masked) * (-1) * pdf_masked).dot(sum_arr)
    return count_entropy_mask.item()


def count_cumulative_distribution(hist):
    cumulative_distribution = hist.cumsum()
    cumulative_distribution_mask = np.ma.masked_equal(cumulative_distribution, 0)
    cumulative_distribution_mask = (cumulative_distribution_mask - cumulative_distribution_mask.min()) * 255 / (
            cumulative_distribution_mask.max() - cumulative_distribution_mask.min())
    cumulative_distribution_mapping = np.ma.filled(cumulative_distribution_mask, 0).astype('uint8')
    return cumulative_distribution_mapping[img_gray]


def create_mapping_arr():
    gray1bit_mapping = np.zeros(256, dtype=np.uint8)
    gray3bit_mapping = np.zeros(256, dtype=np.uint8)
    gray6bit_mapping = np.zeros(256, dtype=np.uint8)
    add_num_1bit = 128
    add_num_3bit = 32
    add_num_6bit = 4
    value_1bit = 0
    value_3bit = 0
    value_6bit = 0
    for i in range(256):
        if i % add_num_1bit == 0 and i / add_num_1bit != 0:
            value_1bit += add_num_1bit
            if value_1bit == 128:
                value_1bit = 255
        if i % add_num_3bit == 0 and i / add_num_3bit != 0:
            value_3bit += add_num_3bit
            if value_3bit == 224:
                value_3bit = 255
        if i % add_num_6bit == 0 and i / add_num_6bit != 0:
            value_6bit += add_num_6bit
            if value_6bit == 252:
                value_6bit = 255
        gray1bit_mapping[i] = value_1bit
        gray3bit_mapping[i] = value_3bit
        gray6bit_mapping[i] = value_6bit
    img_1bit = gray1bit_mapping[img_gray]
    img_3bit = gray3bit_mapping[img_gray]
    img_6bit = gray6bit_mapping[img_gray]
    return img_1bit, img_3bit, img_6bit


print("------------------ task1 start ------------------")
lena_hist, lena_PDF = count_histogram_andPDF(img_gray)
# print("hist: " + str(lena_hist))
# print("pdf: " + str(lena_PDF))
print("------------------ task1 over ------------------")

print("------------------ task2 start ------------------")
plt.subplot(121)
plt.hist(img_gray.ravel(), 256)
plt.title("Gray histogram")

plt.subplot(122)
plt.plot(np.arange(0, 256, 1), lena_PDF)
plt.title("PDF")
plt.show()
print("------------------ task2 over ------------------")

print("------------------ task3 start ------------------")
average = count_gray_img_average(img_gray)
variance = count_gray_img_variance(img_gray)
standard_deviation = count_gray_img_standard_deviation(img_gray)
entropy = count_gray_img_entropy(lena_PDF)

print("average = " + str(average))
print("variance = " + str(variance))
print("standard_deviation = " + str(standard_deviation))
print("entropy = " + str(entropy))

# average = 92
# variance = 1855.4552001953125
# standard_deviation = 43.07499506901089
# entropy = 7.200259275246712

print("------------------ task3 start ------------------")

print("------------------ task4 start ------------------")

# cdf = lena_PDF.cumsum()
# plt.plot(np.arange(0, 256, 1), cdf)
# plt.title("cdf")
# plt.show()

lena_img2 = count_cumulative_distribution(lena_hist)
plt.subplot(221)
plt.hist(img_gray.ravel(), 256)
plt.title("Origin Gray histogram")

plt.subplot(222)
io.imshow(img_gray)
plt.title("Origin Image")

plt.subplot(223)
plt.hist(lena_img2.ravel(), 256)
plt.title("New Gray histogram")

plt.subplot(224)
io.imshow(lena_img2)
plt.title("New Image")
plt.show()
print("------------------ task4 over ------------------")

print("------------------ task5 start ------------------")

img_1bit, img_3bit, img_6bit = create_mapping_arr()

plt.subplot(131)
io.imshow(img_1bit)
plt.title("1bit")
plt.subplot(132)
io.imshow(img_3bit)
plt.title("3bit")
plt.subplot(133)
io.imshow(img_6bit)
plt.title("6bit")
plt.show()

io.imsave("./img/lena1bit.png", img_1bit)
io.imsave("./img/lena3bit.png", img_3bit)
io.imsave("./img/lena6bit.png", img_6bit)
print("------------------ task5 over ------------------")

end_time = time.time()

print("program cost time: " + str(end_time - start_time))
