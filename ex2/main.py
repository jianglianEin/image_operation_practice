import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, img_as_ubyte
import time

start_time = time.time()

img = io.imread("./img/lena.png")
img_gray = img_as_ubyte(color.rgb2grey(img))


def draw_circle(img_gray_in_method):
    r = 120
    circle_img = np.zeros([2 * r, 2 * r], dtype=np.uint8)
    left_top_x, left_top_y = 180, 180

    for i in range(2 * r):
        for j in range(2 * r):
            lx = abs(i - r)
            ly = abs(j - r)
            l = (pow(lx, 2) + pow(ly, 2)) ** 0.5

            if l < r:
                circle_img[i, j] = img_gray_in_method[left_top_x + i, left_top_y + j]
    return circle_img


def rotate_circle(img_gray_in_method, angle):
    circle = draw_circle(img_gray_in_method)

    w, h = circle.shape
    new_w = int(w * abs(np.cos(angle)) + h * abs(np.sin(angle))) + 1
    new_h = int(w * abs(np.sin(angle)) + h * abs(np.cos(angle))) + 1
    rotate_img = np.zeros((new_h, new_w, 3), dtype=int)

    trans = np.array([[1, 0, 0], [0, -1, 0], [-0.5 * new_w, 0.5 * new_h, 1]])
    trans = trans.dot(
        np.array([[np.cos(angle), np.sin(angle), 0], [-np.sin(angle), np.cos(angle), 0], [0, 0, 1]]))
    trans = trans.dot(np.array([[1, 0, 0], [0, -1, 0], [0.5 * w, 0.5 * h, 1]]))

    for x in range(new_w):
        for y in range(new_h):
            rotate_position = np.array([x, y, 1]).dot(trans)
            if 0 <= rotate_position[0] < w and 0 <= rotate_position[1] < h:
                # 最邻近内插
                rotate_img[y][x] = circle[int(rotate_position[1])][int(rotate_position[0])]

    start_pos = int((new_w - w) / 2)
    end_pos = int(new_w - start_pos)
    rotate_result = img_as_ubyte(rotate_img)[:, :, 0][start_pos: end_pos, start_pos: end_pos]

    img_result = img_gray_in_method.copy()
    left_top_x, left_top_y = 180, 180

    for i in range(w):
        for j in range(h):
            if rotate_result[i][j] != 0:
                img_result[left_top_x + i, left_top_y + j] = rotate_result[i, j]
    return img_result


def img_vertically_and_horizontal_process(img_gray_in_method):
    img_roi = img_gray_in_method[200:400, 200:400]
    img_roi_turn_horizontal = img_roi.copy()
    img_roi_turn_vertically = img_roi.copy()
    x, y = img_roi.shape
    for i in range(x):
        for j in range(y):
            img_roi_turn_vertically[i, j] = img_roi[-1 * i + 199, 1 * j]
    for i in range(x):
        for j in range(y):
            img_roi_turn_horizontal[i, j] = img_roi[1 * i, -1 * j + 199]
    img_gray_vertically = img_gray_in_method.copy()
    img_gray_horizontal = img_gray_in_method.copy()
    img_gray_vertically[200:400, 200:400] = img_roi_turn_vertically
    img_gray_horizontal[200:400, 200:400] = img_roi_turn_horizontal
    io.imsave("./img/lena_vertically.png", img_gray_vertically)
    io.imsave("./img/lena_horizontal.png", img_gray_horizontal)


def rotate_and_scale_img(img_gray_in_method):
    angle = 45 * np.pi / 180
    w, h = img_gray_in_method.shape
    new_w = int(w * abs(np.cos(angle)) + h * abs(np.sin(angle))) + 1
    new_h = int(w * abs(np.sin(angle)) + h * abs(np.cos(angle))) + 1
    trans_point_position = 200
    new_trans_point_position = 2 * int(trans_point_position * abs(np.cos(angle))) + 1

    rotate_img = np.zeros((new_h, new_w, 3), dtype=int)
    result_img = np.zeros((int(new_h * 0.95), int(new_w * 1.05), 3), dtype=int)

    rotate_step = np.array([[1, 0, 0], [0, -1, 0], [-0.5 * new_w, new_trans_point_position, 1]])
    rotate_step = rotate_step.dot(
        np.array([[np.cos(angle), np.sin(angle), 0], [-np.sin(angle), np.cos(angle), 0], [0, 0, 1]]))
    rotate_step = rotate_step.dot(np.array([[1, 0, 0], [0, -1, 0], [trans_point_position, trans_point_position, 1]]))
    scale_step = np.array([[0.95, 0, 0], [0, 1.05, 0], [0, 0, 1]])
    for x in range(new_w):
        for y in range(new_h):
            rotate_position = np.array([x, y, 1]).dot(rotate_step)
            if 0 <= rotate_position[0] < w and 0 <= rotate_position[1] < h:
                rotate_img[y][x] = img_gray_in_method[int(rotate_position[1])][int(rotate_position[0])]

    for x in range(int(new_w * 1.05)):
        for y in range(int(new_h * 0.95)):
            step_position = np.array([x, y, 1]).dot(scale_step)
            if 0 <= step_position[0] < new_w and 0 <= step_position[1] < new_h:
                result_img[y][x] = rotate_img[int(step_position[1])][int(step_position[0])]

    result_img = img_as_ubyte(result_img)[:, :, 0]

    io.imsave("./img/lena_rotate_and_scale.png", result_img)


if __name__ == "__main__":
    img_vertically_and_horizontal_process(img_gray.copy())

    angle_30 = 30 * np.pi / 180
    angle_minus_60 = -60 * np.pi / 180
    img_rotate_30_angle_result = rotate_circle(img_gray.copy(), angle_30)
    img_rotate_minus_60_angle_result = rotate_circle(img_gray.copy(), angle_minus_60)
    io.imsave("./img/img_rotate_30_angle_result.png", img_rotate_30_angle_result)
    io.imsave("./img/img_rotate_minus_60_angle_result.png", img_rotate_minus_60_angle_result)

    rotate_and_scale_img(img_gray.copy())

    end_time = time.time()

    print("program cost time: " + str(end_time - start_time))
