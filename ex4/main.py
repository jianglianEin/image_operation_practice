import numpy as np
import matplotlib.pyplot as plt
from pip._vendor.msgpack.fallback import xrange
from skimage import io, color, img_as_ubyte
from skimage.measure import label
from skimage.color import label2rgb
import time
from skimage import morphology

start_time = time.time()

img = io.imread("./img/lung.png")
img_gray = img_as_ubyte(color.rgb2grey(img))


def lung_segmentation(img_gray_in_method):
    rough_segmentation_img = otsu_thresholding(img_gray_in_method)
    lung_masks = remove_noise_regions(rough_segmentation_img)

    closing = morphology.binary_closing(lung_masks, morphology.square(14))
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

if __name__ == "__main__":
    lung_segmentation(img_gray.copy())
