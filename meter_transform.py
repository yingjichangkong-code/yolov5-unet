import copy
import time
import cv2
import numpy as np
from collections import Counter

METER_SHAPE = [512, 512]  # 高x宽
# 圆形表盘的中心点
CIRCLE_CENTER = [256, 256]  # 高x宽
# 圆形表盘的半径
CIRCLE_RADIUS = 250
# 圆周率
PI = 3.1415926536
# 在把圆形表盘转成矩形后矩形的高
# 当前设置值约为半径的一半，原因是：圆形表盘的中心区域除了指针根部就是背景了
# 我们只需要把外围的刻度、指针的尖部保存下来就可以定位出指针指向的刻度
RECTANGLE_HEIGHT = 140
RECTANGLE_HEIGHT_SCALE = 100
RECTANGLE_HEIGHT_POINT = 160
# 矩形表盘的宽，即圆形表盘的外周长
RECTANGLE_WIDTH = 600
#每一个刻度大小
RangeNUM = 0.2
#分割类别字典
SEG_CNAME2CLSID = {'background': 0, 'pointer': 1, 'scale': 2}

#将分割后的圆环转换成矩形
def Circle2Rectangle(CircleImg):
    x = np.arange(RECTANGLE_WIDTH)
    y = np.arange(RECTANGLE_HEIGHT)
    X, Y = np.meshgrid(x, y)
    theta = PI * 2 * (X + 1) / RECTANGLE_WIDTH
    rho = CIRCLE_RADIUS - Y - 1
    Y = (CIRCLE_CENTER[0] + rho * np.cos(theta) + 0.5).astype(int)
    X = (CIRCLE_CENTER[1] - rho * np.sin(theta) + 0.5).astype(int)
    rectangle_meter = CircleImg[Y, X]
    return rectangle_meter

#将矩阵转换成线性刻度
def Rectangle2Line(rectangle_meter):
    start = time.time()
    col_sum = list(map(sum, zip(*rectangle_meter)))  #按列求和
    # 对上述线性数组进行均值二值化
    bin_col_sum = copy.deepcopy(col_sum)
    bin_col_sum_mean_data = np.mean(col_sum)         #求上述得到的列表的均值
    # 对上述列表进行二值化
    for col in range(RECTANGLE_WIDTH):
        if col_sum[col] < bin_col_sum_mean_data:
            bin_col_sum[col] = 0
        else:
            bin_col_sum[col] = 1
    finish = time.time()
    # print(f"矩阵沿高度压缩用时：{1000 * (finish - start)}ms")
    return bin_col_sum


#找每一块刻度的中心点
def FindCenter(bin_line,IsPoint):
    start = time.time()
    find_start = False
    one_scale_start = 0
    one_scale_end = 0
    locations = list()
    for i in range(RECTANGLE_WIDTH - 1):
        if bin_line[i] > 0 and bin_line[i + 1] > 0:
            if find_start == False:
                one_scale_start = i
                find_start = True
        if find_start:
            if bin_line[i] == 0 and bin_line[i + 1] == 0:
                one_scale_end = i - 1
                one_scale_location = (one_scale_start + one_scale_end) / 2
                locations.append(int(one_scale_location))
                if IsPoint == 1:
                    break
                else:
                    one_scale_start = 0
                    one_scale_end = 0
                    find_start = False
    finish = time.time()
    # print(f"寻找中心点用时：{1000 * (finish - start)}ms")
    return locations

#读数，即找指针指向了第几个刻度线上
def Reading(scale_locations, point_locations):
    num_scales = 0
    num_PointerinRange = 0
    for i, scale in enumerate(scale_locations):
        if scale > point_locations[0]:
            num_scales = i-1
            num_PointerinRange = round(1 - (scale - point_locations[0]) / (scale - scale_locations[i - 1]), 2)
            break
    return num_scales, num_PointerinRange

#做标准刻度线
def MakeScale(point_locations,scale_locations):
    scale_img = np.zeros((RECTANGLE_HEIGHT, RECTANGLE_WIDTH), dtype=np.uint8)
    row, col = scale_img.shape
    scale_index = 0
    draw_flag = False
    scale_flag = False
    for i in range(col - 1):
        if draw_flag == True and scale_flag == True:
            scale_img[int(RECTANGLE_HEIGHT / 2 )][i] = 1
            scale_img[int(RECTANGLE_HEIGHT / 2 - 1)][i] = 1
        if point_locations[0] == i:
            for j in range(int(RECTANGLE_HEIGHT / 2) - 50, int(RECTANGLE_HEIGHT / 2)):
                scale_img[j][i - 1] = 2
                scale_img[j][i] = 2
                scale_img[j][i + 1] = 2
        if scale_locations[scale_index] == i:
            scale_flag = True
            for j in range(int(RECTANGLE_HEIGHT / 2) - 20, int(RECTANGLE_HEIGHT / 2)):
                scale_img[j][i - 1] = 1
                scale_img[j][i] = 1
                scale_img[j][i + 1] = 1
            if scale_index < len(scale_locations) - 1:
                draw_flag = True
                scale_index += 1
            else:
                draw_flag = False
    return scale_img

#生成结果图
def MakeResultImg(resultnum):
    ResImg = np.ones((512, 512), dtype=np.uint8) * 255  #生成一张白底图
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (200, 260)  #文本的位置
    fontScale = 2  #字体缩放因子
    thickness = 3  #宽度
    color = (0, 0, 0)
    # cv2.LINE_AA 线条类型
    cv2.putText(ResImg, str(resultnum), org, font, fontScale, color, thickness, cv2.LINE_AA)
    return ResImg

def check_scale(arr, final_length):
    # 找到相邻两个数的差值的众数
    diff_counter = Counter([arr[i + 1] - arr[i] for i in range(len(arr) - 1)])
    mode_diff = diff_counter.most_common(1)[0][0]

    # 根据众数整理数组
    new_arr = arr[:2]
    for i in range(2, len(arr)):
        diff = arr[i] - arr[i - 1]
        if i == 2 and diff < mode_diff:
            new_arr.append(arr[i])
            continue
        if diff < mode_diff - 2:
            continue
        elif diff > mode_diff + 2:
            insert_num = arr[i - 1] + mode_diff
            while insert_num < arr[i]:
                new_arr.append(insert_num)
                insert_num += mode_diff
        new_arr.append(arr[i])

    # 如果整理后的数组长度小于规定的长度，需要进行补全
    while len(new_arr) < final_length:
        new_arr.append(new_arr[-1] + mode_diff)

    # 如果整理后的数组长度大于规定的长度，需要进行截断
    return new_arr[:final_length]

def check_pointer_seg(seg_img):
    # 将图像转换为灰度图
    gray = cv2.cvtColor(seg_img, cv2.COLOR_BGR2GRAY)
    # 二值化图像
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # 寻找轮廓
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # 找到最大轮廓
    max_contour = max(contours, key=cv2.contourArea)
    # 创建掩模
    mask = np.zeros_like(gray)
    # 将最大轮廓填充为白色
    cv2.drawContours(mask, [max_contour], 0, 255, -1)
    # 将其他轮廓填充为黑色
    for contour in contours:
        if contour is not max_contour:
            cv2.drawContours(mask, [contour], 0, 0, -1)
    # 将掩模应用于原始图像
    result = cv2.bitwise_and(seg_img, seg_img, mask=mask)
    return result