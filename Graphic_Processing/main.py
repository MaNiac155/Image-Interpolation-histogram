import numpy as np
import matplotlib.pyplot as plt
import pylab
import math

img = plt.imread("pic/jean.jpg")
print(img.shape)
height, width, mode = img.shape[0], img.shape[1], img.shape[2]
desWidth = int(width * 2)
desHeight = int(height * 2)

# 画灰度图、RGB三个通道的图
# b = np.array([0.299, 0.587, 0.114])
# b=np.array([0,0,0])
# x=np.dot(img,b)
# plt.imsave("pic/new.jpg",x,cmap="gray")
# plt.subplot(2,2,1)
# plt.imshow(img)
#
# for i in range(height):
#     for j in range(width):
#         img[i,j,2]=0
#         img[i,j,1]=0
# plt.subplot(2,2,2)
# plt.imshow(img)
# img = plt.imread("pic/Lei.jpg")
#
# for i in range(height):
#     for j in range(width):
#         img[i,j,0]=0
#         img[i,j,2]=0
# plt.subplot(2,2,3)
# plt.imshow(img)
# img = plt.imread("pic/Lei.jpg")
#
# for i in range(height):
#     for j in range(width):
#         img[i,j,0]=0
#         img[i,j,1]=0
# plt.subplot(2,2,4)
# plt.imshow(img)
#
# plt.show()

# 最近邻插值

"""图像插值"""


def nearest():
    desImage = np.zeros((desHeight, desWidth, mode), np.uint8)
    for des_x in range(0, desHeight):
        for des_y in range(0, desWidth):
            src_x = int(des_x * (height / desHeight))
            src_y = int(des_y * (width / desWidth))
            desImage[des_x, des_y] = img[src_x, src_y]
    print(desImage.shape)
    plt.imsave("pic/nearest.jpg", desImage)


# 线性插值
def linear():
    new_img = np.zeros((desHeight, desWidth, mode), dtype=np.uint8)
    for des_y in range(0, desHeight):
        for des_x in range(0, desWidth):
            src_x = (des_x + 0.5) * width / desWidth - 0.5  # 为了使目的点在中间，所有的点都能用上
            src_y = int((des_y + 0.5) * height / desHeight - 0.5)

            src_x_1 = int(np.floor(src_x))
            src_x_2 = min(src_x_1 + 1, width - 1)

            if src_x_2 == src_x_1:  # 防止warning 分母为0
                value = 0
            else:
                value = (img[src_y, src_x_1] * (src_x_2 - src_x) / (src_x_2 - src_x_1) + img[
                    src_y, src_x_2] * (src_x - src_x_1) / (src_x_2 - src_x_1))
            new_img[des_y, des_x] = value

    plt.imsave("pic/linear.jpg", new_img)


# 双线性插值
def linear_2():
    new_img = np.zeros((desHeight, desWidth, mode), dtype=np.uint8)
    for des_y in range(0, desHeight):
        for des_x in range(0, desWidth):
            src_x = (des_x + 0.5) * width / desWidth - 0.5  # 为了使目的点在中间，所有的点都能用上
            src_y = int((des_y + 0.5) * height / desHeight - 0.5)

            src_x_1 = int(np.floor(src_x))
            src_y_1 = int(np.floor(src_y))
            src_x_2 = min(src_x_1 + 1, width - 1)
            src_y_2 = min(src_y_1 + 1, height - 1)
            if src_x_2 == src_x_1 or src_y_2 == src_y_1:  # 防止warning 分母为0
                value = 0
            else:
                value_1 = (src_x_2 - src_x) * img[src_y_1, src_x_1] + (src_x - src_x_1) * img[
                    src_y_1, src_x_2]  # x轴插值
                value_2 = (src_x_2 - src_x) * img[src_y_2, src_x_1] + (src_x - src_x_1) * img[
                    src_y_2, src_x_2]
                value = (src_y_2 - src_y) * value_1 + (src_y - src_y_1) * value_2  # y轴插值

            new_img[des_y, des_x] = value

    plt.imsave("pic/doubleLinear.jpg", new_img)


def get_weight(x):
    X = np.floor(x)
    stemp_x = [1 + (x - X), x - X, 1 - (x - X), 2 - (x - X)]
    a = -0.5

    w_x = [0, 0, 0, 0]
    w_x[0] = a * abs(stemp_x[0] * stemp_x[0] * stemp_x[0]) - 5 * a * stemp_x[0] * stemp_x[0] + 8 * a * abs(
        stemp_x[0]) - 4 * a
    w_x[1] = (a + 2) * abs(stemp_x[1] * stemp_x[1] * stemp_x[1]) - (a + 3) * stemp_x[1] * stemp_x[1] + 1
    w_x[2] = (a + 2) * abs(stemp_x[2] * stemp_x[2] * stemp_x[2]) - (a + 3) * stemp_x[2] * stemp_x[2] + 1
    w_x[3] = a * abs(stemp_x[3] * stemp_x[3] * stemp_x[3]) - 5 * a * stemp_x[3] * stemp_x[3] + 8 * a * abs(
        stemp_x[3]) - 4 * a
    return w_x


# 双三次插值
def biCubic():
    new_img = np.zeros((desHeight, desWidth, mode), dtype=np.uint8)

    for des_y in range(2, desHeight - 4):
        for des_x in range(2, desWidth - 4):
            src_x = des_x * width / desWidth
            src_y = des_y * height / desHeight
            w_x = get_weight(src_x)
            w_y = get_weight(src_y)
            value = np.array([0, 0, 0])
            for i in range(4):  # 1 2 3 4 5 6 7 8 9 10
                for j in range(4):
                    value = value + img[int(src_y) + i - 1, int(src_x) + j - 1] * w_x[j] * w_y[i]
            for n in range(mode):
                value[n] = np.clip(value[n], 0, 255)
            new_img[des_y, des_x] = value
    plt.imsave("pic/biCubic.jpg", new_img)


# # 显示
# picture=[plt.imread("pic/jean.jpg"),plt.imread("pic/nearest.jpg"),plt.imread("pic/linear.jpg"),plt.imread(
#     "pic/doubleLinear.jpg"),plt.imread("pic/biCubic.jpg")]
# plt.rcParams['font.sans-serif']=['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
#
# plt.subplot(4,6,2)
# plt.imshow(picture[0])
# plt.title("原图",)
# plt.subplot(2,3,2)
# plt.imshow(picture[1])
# plt.title("最近邻插值")
# plt.subplot(2,3,3)
# plt.imshow(picture[2])
# plt.title("单线性插值")
# plt.subplot(2,3,5)
# plt.imshow(picture[3])
# plt.title("双线性插值")
# plt.subplot(2,3,6)
# plt.imshow(picture[4])
# plt.title("双三次插值")
# pylab.show()


# 灰度图

"""直方图"""
b = np.array([0.299, 0.587, 0.114])
x = np.round(np.dot(img, b))
plt.imsave("pic/new.jpg", x, cmap="gray")

gray_points = np.clip(x, 0, 255)
gray_points = gray_points.reshape(-1)  # 灰度一维数组


# 直方图统计
def hist(graypoints):
    hist_map = np.zeros(256, dtype=np.uint32)

    for pix in graypoints:
        hist_map[int(pix)] += 1

    plt.figure()
    x_axis = np.linspace(0, 255, 256)
    plt.hist(x_axis, bins=256, weights=hist_map)

    ave = sum(hist_map) / 256
    # print(ave)
    return hist_map


# 直方图均衡化
def histBalance():
    img = plt.imread("pic/IMG.jpg")
    height_, width_ = img.shape[0], img.shape[1]

    b = np.array([0.299, 0.587, 0.114])
    x = np.round(np.dot(img, b))

    gpoints = (np.clip(x, 0, 255)).reshape(-1)
    hist_map = hist(gpoints)  # 直方图 频数
    nsum = height_ * width_  # 像素总数
    freq = hist_map / nsum  # 各明度值出现的频率
    freq_total = np.zeros(256)
    for i in range(0, 256):
        freq_total[i] = sum(freq[0:i + 1])

    map_new = np.zeros(256, dtype=np.uint32)  # map_new是灰度值的对应关系
    for i in range(0, 256):
        map_new[i] = np.round(freq_total[i] * 255)

    # 转换原图像
    x_new = np.zeros([height_, width_])
    for i in range(0, height_):
        for j in range(0, width_):
            value = x[i, j]
            value_n = map_new[int(value)]
            x_new[i, j] = value_n

    gpointsn = (np.clip(x_new, 0, 255)).reshape(-1)
    hist(gpointsn)
    plt.imsave("pic/balance.jpg", x_new, cmap="gray")


# 主函数

nearest()
linear()
linear_2()
biCubic()

hist(gray_points)
histBalance()

