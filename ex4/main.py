import numpy as np
import matplotlib.pyplot as plt
from pip._vendor.msgpack.fallback import xrange
from scipy.signal import correlate2d
from skimage import io, color, img_as_ubyte
from skimage.measure import label
from skimage.color import label2rgb, rgb2hsv
from skimage.segmentation import slic
import time
from skimage import morphology
from sklearn.cluster import KMeans

start_time = time.time()

img_lung = io.imread("./img/lung.png")
img_lung_gray = img_as_ubyte(color.rgb2grey(img_lung))

img_cat = io.imread("./img/cat.jpg")
img_cat_gray = img_as_ubyte(color.rgb2grey(img_cat))


def remove_black_holes_in_lung(lung_mask_reversal):
    label_img = label(lung_mask_reversal)
    # image_label_overlay = label2rgb(label_img, rough_segmentation_img_in_method)
    label_count, _ = np.histogram(label_img.ravel(), label_img.max(), [0, label_img.max()])

    max_label = max(label_count)

    keep_label_list = list()

    for label_index in range(label_count.size):
        if label_count[label_index] < max_label:
            keep_label_list.append(label_index)

    masks = np.zeros(shape=label_img.shape)

    for keep_label in keep_label_list:
        mask = label_img == keep_label
        masks += mask

    io.imshow(masks)
    plt.title("lung_masks_remove_black_holes")
    plt.show()
    return masks


def lung_segmentation(img_gray_in_method):
    rough_segmentation_img = bi_model_thresholding(img_gray_in_method)
    lung_masks = remove_noise_regions(rough_segmentation_img)
    lung_masks_remove_black_holes = remove_black_holes_in_lung(lung_masks == 0)

    closing = morphology.binary_closing(lung_masks_remove_black_holes, morphology.square(3))
    io.imshow(closing)
    plt.title("closing")
    plt.show()


def bi_model_thresholding(img_in_method):
    hist, _ = np.histogram(img_in_method.ravel(), 256, [0, 256])
    hist = list(hist)
    max_value = max(hist)
    max_value_index = hist.index(max_value)
    second_max_value = max(hist[max_value_index + 1:])
    second_max_value_index = hist.index(second_max_value)

    min_value = min(
        hist[max_value_index: second_max_value_index]) if max_value_index <= second_max_value_index else min(
        hist[second_max_value_index: max_value_index])

    threshold = hist.index(min_value)

    img_in_method = img_in_method <= threshold
    return img_in_method


def otsu_thresholding(img_in_method, th_begin=0, th_end=256, th_step=1):
    max_g = 0
    suitable_th = 0
    for threshold in xrange(th_begin, th_end, th_step):
        bin_img = img_in_method > threshold
        bin_img_inv = img_in_method <= threshold
        fore_pix = np.sum(bin_img)
        back_pix = np.sum(bin_img_inv)
        if 0 == fore_pix:
            break
        if 0 == back_pix:
            continue

        w0 = float(fore_pix) / img_in_method.size
        u0 = float(np.sum(img_in_method * bin_img)) / fore_pix
        w1 = float(back_pix) / img_in_method.size
        u1 = float(np.sum(img_in_method * bin_img_inv)) / back_pix
        # intra-class variance
        g = w0 * w1 * (u0 - u1) * (u0 - u1)
        if g > max_g:
            max_g = g
            suitable_th = threshold

    img_in_method = img_in_method <= suitable_th
    return img_in_method


def remove_noise_regions(rough_segmentation_img_in_method):
    label_img = label(rough_segmentation_img_in_method)
    # image_label_overlay = label2rgb(label_img, rough_segmentation_img_in_method)
    label_count, _ = np.histogram(label_img.ravel(), label_img.max(), [0, label_img.max()])

    max_label = max(label_count)

    keep_label_list = list()

    for label_index in range(label_count.size):
        if label_count[label_index] * 10 > max_label:
            keep_label_list.append(label_index)

    x, y = label_img.shape
    for i in range(x):
        if label_img[i][0] in keep_label_list:
            keep_label_list.remove(label_img[i][0])
        if label_img[i][x - 1] in keep_label_list:
            keep_label_list.remove(label_img[i][x - 1])
    for i in range(y):
        if label_img[0][i] in keep_label_list:
            keep_label_list.remove(label_img[0][i])
        if label_img[0][y - 1] in keep_label_list:
            keep_label_list.remove(label_img[0][y - 1])

    masks = np.zeros(shape=label_img.shape)

    for keep_label in keep_label_list:
        mask = label_img == keep_label
        masks += mask

    io.imshow(masks)
    plt.title("masks")
    plt.show()
    return masks


def create_data_mat_by_rgb(img_in_method):
    row_in_method, col_in_method, channel = img_in_method.shape
    data = []
    for i in range(row_in_method):
        for j in range(col_in_method):
            x, y, z = img_in_method[i, j, :]
            data.append([x, y, z])
    return np.mat(data), row_in_method, col_in_method


def create_data_mat_by_rgb_and_coordinates(img_in_method):
    row_in_method, col_in_method, channel = img_in_method.shape
    data = []
    for i in range(row_in_method):
        for j in range(col_in_method):
            x, y, z = img_in_method[i, j, :]
            data.append([x, y, z, i, j])
    return np.mat(data), row_in_method, col_in_method


def create_data_mat_by_hsv_and_coordinates(img_in_method):
    img_hsv = rgb2hsv(img_in_method)
    row_in_method, col_in_method, channel = img_hsv.shape
    data = []
    for i in range(row_in_method):
        for j in range(col_in_method):
            x, y, z = img_hsv[i, j, :]
            data.append([x * 360, y, z, i, j])
    return np.mat(data), row_in_method, col_in_method


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


def over_segmentation_by_k_mean(img_in_method):
    data_mat, row, col = create_data_mat_by_hsv_and_coordinates(img_in_method)
    k_res = KMeans(n_clusters=100).fit(data_mat)
    mask = k_res.labels_.reshape([row, col])

    mask_boundary_in_method = laplace_edge(mask)

    for i in range(row):
        for j in range(col):
            if mask_boundary_in_method[i][j] != 0:
                img_in_method[i, j, 0] = 255
                img_in_method[i, j, 1] = 0
                img_in_method[i, j, 2] = 0
    io.imshow(img_in_method)
    plt.title("k_mean")
    plt.show()


def over_segmentation_by_slic(img_in_method):
    row, col, _ = img_in_method.shape
    segments = slic(img_in_method, n_segments=100)
    mask_boundary = laplace_edge(segments)

    for i in range(row):
        for j in range(col):
            if mask_boundary[i][j] != 0:
                img_in_method[i, j, 0] = 255
                img_in_method[i, j, 1] = 0
                img_in_method[i, j, 2] = 0
    io.imshow(img_in_method)
    plt.title("slic")
    plt.show()


def cat_segmentation(img_in_method):
    over_segmentation_by_k_mean(img_in_method.copy())
    over_segmentation_by_slic(img_in_method.copy())


if __name__ == "__main__":
    lung_segmentation(img_lung_gray.copy())
    cat_segmentation(img_cat.copy())
