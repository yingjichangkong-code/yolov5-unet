import copy
from yolov5_Detect import detect_model_load,detect
from Unet_Segment import seg_model_load,segment_image,Visualimg
from meter_transform import Circle2Rectangle,Rectangle2Line,FindCenter,Reading
import cv2
import numpy as np
import time
from scipy import stats
from collections import Counter


# 输入图片
path = 'image/22222.jpg'
img = cv2.imread(path)
detect_imgs = []
segment_res = []
segment_imgs = []

# 模型加载
detect_model = detect_model_load((640, 640),False) #yolov5检测模型加载
seg_model = seg_model_load()  # unet检测模型加载

FENDUZHI = 0.5
MAX_VALUE = 16


# def check_scale(scale_list):
#     scale_num = MAX_VALUE / FENDUZHI  # 刻度线的数量
#     # 计算刻度线数据差值的中位数
#     scale_list = np.array(scale_list)
#     diff = np.diff(scale_list)
#     print(diff)
#     median = np.median(diff)
#     zhongshu = stats.mode(diff)
#     avarge = np.mean(diff)
#     print('median',median)
#     print('mean',avarge)
#     print('zhongshu',zhongshu[0][0])
#     for index in range(len(diff)):
#         if diff[index] < zhongshu-2:
#             scale_list.pop(index+1)
#         elif diff[index] > zhongshu+2:
#             scale_list.insert(index+1,scale_list[index]+zhongshu)


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






# 向目标检测网络传入一张图片
det_res = detect(detect_model,img)

# 对模型预测结果进行处理
for info in det_res:
    position = info['position']
    # xywh2xyxy
    position[2] = position[0] + position[2]
    position[3] = position[1] + position[3]
    xmin, ymin, w, h = position
    print(xmin, ymin, w, h)
    #得到检测出的图片
    temp_img = img[position[1]:position[3],position[0]:position[2]]
    detect_imgs.append(temp_img)
    cv2.imshow('temp_img',temp_img)
    cv2.waitKey(0)

# 把检测出的图片进行分割
for img in detect_imgs:
    res = segment_image(seg_model,img)  #把图像传入unet分割模型进行分割
    vis_img = Visualimg(res)            #分割结果可视化
    segment_res.append(res)
    segment_imgs.append(vis_img)
    cv2.imshow('vis_img', vis_img)
    cv2.waitKey(0)

#把分割的圆环图拉成矩形
for segimg in segment_res:
    C2R_start = time.time()
    resimg = Circle2Rectangle(segimg)    #刻度加指针的图
    scale_rect = copy.deepcopy(resimg)   #只有刻度的图
    pointer_rect = copy.deepcopy(resimg) #只有指针的图
    scale_rect[scale_rect == 1] = 0      #从原分割图中分割出来刻度图
    pointer_rect[pointer_rect == 2] = 0  #从原分割图中分割出来指针图
    C2R_finish = time.time()
    print(f"圆环拉伸成矩形用时{1000*(C2R_finish-C2R_start)}ms")
    #将矩阵转换成线性刻度
    bin_line_scale = Rectangle2Line(scale_rect)
    bin_pointer_scale = Rectangle2Line(pointer_rect)
    # 找每一块刻度的中心点
    scale_locations = FindCenter(bin_line_scale,False)
    pointer_locations = FindCenter(bin_pointer_scale,True)
    print('scale_locations',scale_locations)
    print('pointer_locations',pointer_locations)
    scale_locations = check_scale(scale_locations,32)
    print('scale_locations', scale_locations)
    num_scales,num_PointerinRange = Reading(scale_locations,pointer_locations)
    ReadingRes = round((num_scales) * 0.5 + num_PointerinRange * 0.5, 2)
    print("最终读数为：",ReadingRes)
