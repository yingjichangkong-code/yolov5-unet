from PyQt5.QtGui import QCursor, QImage, QPixmap
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QGridLayout
from matplotlib.figure import Figure
from matplotlib.ticker import MultipleLocator
from matplotlib.lines import Line2D
from meter import Ui_MainWindow
from Setting import SettingWin
import matplotlib
from PyQt5.QtCore import Qt,QThread, pyqtSignal
import sys
matplotlib.use("Qt5Agg")  # 声明使用QT5
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import copy
import time
import pygame
import numpy as np
import cv2
from yolov5_Detect import detect_model_load,detect
from Unet_Segment import seg_model_load,segment_image,Visualimg
from meter_transform import Circle2Rectangle,Rectangle2Line,FindCenter,Reading,MakeScale,MakeResultImg,check_scale,check_pointer_seg
from DiyDialog import diydialog   #自定义对话框


class Figure_Canvas(FigureCanvas):
    def __init__(self, width=2, height=1):
        self.fig = Figure(figsize=(width, height), dpi=80)
        super(Figure_Canvas, self).__init__(self.fig)
        self.ax = self.fig.add_subplot(111)  # 111表示1行1列，第一张曲线图

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
RECTANGLE_WIDTH = 1400
#展示框高度和宽度
SHOW_WIGHT = 370
SHOW_HEIGHT = 670
#每一个刻度大小
RangeNUM = 0.5
#分割类别字典
SEG_CNAME2CLSID = {'background': 0, 'pointer': 1, 'scale': 2}

KEEP_RATE = 0.05  #保压的比例
DESENCE_RATE = 0.3   #骤降的比例

#颜色
colors = [
    [255, 255, 255],  # background 白色
    [0, 0, 0],  # class 1     黑色
    [255, 0, 0],  # class 2   红色
    [0,255,0 ],  # class 3   绿色
]


resimg_list = [[] for i in range(5)]    #用来存过程图片包括，表盘检测图、分割图、拉伸图、刻度图、结果图
#当前展示的图像的index


class PicWorkThread(QThread): #图片检测工作线程
    workend = pyqtSignal()  #分析结束信号
    def __init__(self,detector,segmente,pic_1,pic_2,pic_3,pic_result,frame_show,fenduzhi,zongkedu):
        super(PicWorkThread, self).__init__()
        self.detector = detector
        self.segmente = segmente
        self.pic_1 = pic_1
        self.pic_2 = pic_2
        self.pic_3 = pic_3
        self.pic_result = pic_result
        self.frame_show = frame_show
        self.fenduzhi = fenduzhi
        self.zongkedu = zongkedu

    def run(self):
        self.pic_1.setPixmap(QPixmap(""))
        self.pic_2.setPixmap(QPixmap(""))
        self.pic_3.setPixmap(QPixmap(""))
        self.frame_show.setPixmap(QPixmap(""))
        self.pic_result.setPixmap(QPixmap(""))
        #读入数据
        self.img = cv2.imdecode(np.fromfile(self.filepath, dtype=np.uint8), -1)
        det_result = detect(self.detector,self.img) #得到目标检测结果
        PIC_INDEX = 1
        for res in det_result:
            xmin, ymin, xmax, ymax = res['position']
            w = xmax - xmin
            h = ymax - ymin  #左上角坐标、图片的长，宽
            sub_img = self.img[ymin:ymin + h, xmin:xmin + w]
            sub_img = cv2.resize(sub_img, (512, 512), cv2.INTER_LINEAR)  #裁剪原图片，并resize尺寸
            sub_img_show = self.PicResizeAsWidth(sub_img)
            resimg_list[0].append(sub_img_show)   # step1 - 目标检测结果的图片
            cv2.imwrite(f"output/Image/{PIC_INDEX}-step1.jpg",sub_img)
            seg_result = segment_image(self.segmente,sub_img)   #分割上述步骤得到的结果
            # 对分割的图形进行腐蚀操作
            erode_kernel = np.ones((3, 3), dtype=np.uint8)
            seg_result = cv2.convertScaleAbs(seg_result)
            seg_result = cv2.erode(seg_result, erode_kernel)
            # 可视化分割后的图片
            seg_result_img = Visualimg(seg_result)
            seg_result_img_show = self.PicResizeAsWidth(seg_result_img)
            resimg_list[1].append(seg_result_img_show)  # step2 - 分割结果的图片
            cv2.imwrite(f"output/Image/{PIC_INDEX}-step2.jpg", seg_result_img)
            self.showPIC(seg_result_img_show,self.pic_1)
            # 将分割的图像转换为矩形
            rectangle_meter = Circle2Rectangle(seg_result)
            scale_rect = copy.deepcopy(rectangle_meter)  # 只有刻度的图
            pointer_rect = copy.deepcopy(rectangle_meter)  # 只有指针的图
            scale_rect[scale_rect == 1] = 0  # 从原分割图中分割出来刻度图
            pointer_rect[pointer_rect == 2] = 0  # 从原分割图中分割出来指针图
            #可视化
            rectangle_meter_img = Visualimg(rectangle_meter)
            rectangle_meter_img_show = self.PicResizeAsWidth(rectangle_meter_img)
            resimg_list[2].append(rectangle_meter_img_show)  #step3 - 转换成矩阵的图片
            cv2.imwrite(f"output/Image/{PIC_INDEX}-step3.jpg", rectangle_meter_img)
            self.showPIC(rectangle_meter_img_show,self.pic_2)
            # 把上述图片沿高度方向压缩成线状格式
            bin_line_scale = Rectangle2Line(scale_rect)
            bin_line_pointer = Rectangle2Line(pointer_rect)
            #找线状图形的中心点
            scale_locations = FindCenter(bin_line_scale,False)
            scale_locations = check_scale(scale_locations,int(self.zongkedu/self.fenduzhi))
            pointer_locations = FindCenter(bin_line_pointer, True)
            #做标准横向刻度线
            scale_data = MakeScale(pointer_locations,scale_locations)
            scale_data_img = Visualimg(scale_data)
            scale_data_img_show= self.PicResizeAsWidth(scale_data_img)
            resimg_list[3].append(scale_data_img_show) #step4 - 做成的标准刻度图
            self.showPIC(scale_data_img_show,self.pic_3)
            cv2.imwrite(f"output/Image/{PIC_INDEX}-step4.jpg", scale_data_img)
            #读数
            num_scales,num_PointerinRange = Reading(scale_locations,pointer_locations)
            #计算读数
            ReadingRes = round((num_scales+1) * self.fenduzhi + num_PointerinRange * self.fenduzhi,2)
            # ReadingRes = round((pointer_locations[0] - scale_locations[0]) / (scale_locations[-1] - scale_locations[0]) * 6,2)
            # self.pic_result.setText(str(ReadingRes))
            ReadingResImg = MakeResultImg(ReadingRes)
            self.showPIC(ReadingResImg, self.pic_result)
            resimg_list[4].append(ReadingResImg)  # step5 - 读取结果图
            cv2.imwrite(f"output/Image/{PIC_INDEX}-Reading result.jpg", ReadingResImg)
            PIC_INDEX += 1
        self.workend.emit()


    def GetFilePath(self,filepath):
        self.filepath = filepath


    def PicResizeAsWidth(self,Img):
        Img_Height,Img_Width,channel = Img.shape
        if Img_Height == Img_Width:
            Img = cv2.resize(Img, (370, 370), cv2.INTER_LINEAR)  #裁剪原图片，并resize尺寸
        else:
            Img = cv2.resize(Img, (670, 58), cv2.INTER_LINEAR)  # 裁剪原图片，并resize尺寸
        return Img


    def showPIC(self,pic,label):
        label.setScaledContents(True)
        frame = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)
        height, width, channel = frame.shape
        bytesPerLine = 3 * width
        qImg = QImage(frame.data, width, height, bytesPerLine, QImage.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(qImg))
        self.frame_show.setPixmap(QPixmap.fromImage(qImg))


class VideoWorkThread(QThread): #视频工作线程
    video_workend = pyqtSignal()  #分析结束信号
    updata = pyqtSignal(list)   #更新曲线数据信号
    def __init__(self,detector,segmente,video_show,plot_show,fenduzhi,zongkedu):
        super(VideoWorkThread, self).__init__()
        self.detector = detector
        self.segmente = segmente
        self.videoshow = video_show
        self.plot_show = plot_show
        self.reslist = []
        self.timelist = []
        self.framenum = 0
        self.num_show = 10
        self.fenduzhi = fenduzhi
        self.zongkedu = zongkedu
        self._is_running = True

    def run(self):
        # 读入视频
        vc = cv2.VideoCapture(self.videopath)  # 读入视频文件
        rval, frame = vc.read()
        bin_line_scale = []
        scale_locations = []
        fps = vc.get(cv2.CAP_PROP_FPS)  # 获取视频原帧率
        print("当前视频帧率为：", fps)
        start_time = time.time()  # 获取视频开始播放时的时间
        self.length, self.mean = 0, 0
        while rval and self._is_running:  # 循环读取视频帧
            self.framenum += 1
            rval, frame = vc.read()
            if rval == False:
                break
            frame = cv2.resize(frame, (720, 404))
            if (time.time() - start_time) != 0:  # 实时显示帧数
                cv2.putText(frame, "FPS {0}".format(float('%.1f' % (self.framenum / (time.time() - start_time)))),
                            (500, 300),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                            2)
            if self.framenum % fps == 0 or self.framenum == 1:  # 一秒检测一次
                # 检测表盘
                det_result = detect(self.detector, frame)  # 得到目标检测结果
                for res in det_result:
                    xmin, ymin, w, h = res['position']  # 左上角坐标、图片的长，宽
                    sub_img = frame[ymin:ymin + h, xmin:xmin + w]
                    sub_img = cv2.resize(sub_img, (512, 512), cv2.INTER_LINEAR)  # 裁剪原图片，并resize尺寸
                    seg_result = segment_image(self.segmente, sub_img)  # 分割上述步骤得到的结果
                    # 对分割的图形进行腐蚀操作
                    seg_result = cv2.convertScaleAbs(seg_result)
                    erode_kernel = np.ones((3, 3), dtype=np.uint8)
                    seg_result = cv2.erode(seg_result, erode_kernel)
                    # 圆环转矩形
                    rectangle_meter = Circle2Rectangle(seg_result)
                    pointer_rect = copy.deepcopy(rectangle_meter)
                    pointer_rect[pointer_rect == 2] = 0  # 从原分割图中分出来指针图
                    # 只有第一帧需要分出来刻度，其他帧只需要指针的分割图即可
                    if self.framenum == 1:
                        first_scale_rect = copy.deepcopy(rectangle_meter)
                        first_scale_rect[first_scale_rect == 1] = 0
                        bin_line_scale = Rectangle2Line(first_scale_rect)
                        scale_locations = FindCenter(bin_line_scale, False)
                        # self.length, self.mean = self.get_len_span(scale_locations)
                        # scale_locations = self.check_centerpoint(scale_locations, self.length, self.mean)
                        print('scale_locations', scale_locations)
                        continue
                    pointer_rect = copy.deepcopy(rectangle_meter)  # 只有指针的图
                    pointer_rect[pointer_rect == 2] = 0  # 从原分割图中分离出来指针图
                    # 把上述图片沿高度方向压缩成线状格式
                    bin_line_pointer = Rectangle2Line(pointer_rect)
                    # 找线状图形的中心点
                    pointer_locations = FindCenter(bin_line_pointer, True)
                    print('pointer_locations', pointer_locations)
                    # 读数
                    num_scales, num_PointerinRange = Reading(scale_locations, pointer_locations)
                    # 计算读数
                    ReadingRes = round((num_scales) * RangeNUM + num_PointerinRange * RangeNUM, 2)
                    self.reslist.append(ReadingRes)
                    print(self.reslist)
                    self.timelist.append(self.framenum / fps)
                    self.updata.emit([self.timelist, self.reslist])
                    # cv2.putText(frame, f"result:{ReadingRes}", (xmin+10, ymin),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),2)
                    # 把frame放在label上
                    self.videoshow.setScaledContents(True)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    height, width, channel = frame.shape
                    bytesPerLine = 3 * width
                    qImg = QImage(frame.data, width, height, bytesPerLine, QImage.Format_RGB888)
                    self.videoshow.setPixmap(QPixmap.fromImage(qImg))
            else:
                # 把frame放在label上
                self.videoshow.setScaledContents(True)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height, width, channel = frame.shape
                bytesPerLine = 3 * width
                qImg = QImage(frame.data, width, height, bytesPerLine, QImage.Format_RGB888)
                self.videoshow.setPixmap(QPixmap.fromImage(qImg))
            time.sleep(1 / fps)  # 按原帧率播放
        vc.release()
        self.video_workend.emit()

    def stop(self):
        self._is_running = False

    def get_len_span(self,scale_locations):
        span_list = np.diff(scale_locations)
        print(len(scale_locations), np.mean(span_list))
        return len(scale_locations), np.mean(span_list)

    def check_centerpoint(self,scale_locations,length ,mean):
        if len(scale_locations) > length:
            for i in range(1,length):
                if (scale_locations[i] - scale_locations[i - 1]) < (int(mean)-3):
                    print('scale_locations.pop(i):',scale_locations[i])
                    scale_locations.pop(i)
        else:
            pass
        return scale_locations


    def GetFilePath(self,filepath):
        self.videopath = filepath

class cameraWorkThread(QThread): #摄像头工作线程
    video_workend = pyqtSignal()  #分析结束信号
    updata = pyqtSignal(list)   #更新曲线数据信号
    def __init__(self,detector,segmente,video_show,plot_show,fenduzhi,cameranum,zongkedu):
        super(cameraWorkThread, self).__init__()
        self.detector = detector
        self.segmente = segmente
        self.videoshow = video_show
        self.plot_show = plot_show
        self.reslist = []
        self.timelist = []
        self.framenum = 0
        self.num_show = 10
        self.fenduzhi = fenduzhi
        self.cameranum = cameranum
        self.zongkedu = zongkedu
        self._is_running = True

    def run(self):
        # 读入视频
        if len(self.cameranum) < 2:
            vc = cv2.VideoCapture(int(self.cameranum))  # 调用摄像头
        else:
            vc = cv2.VideoCapture(self.cameranum)  # 调用摄像头
        # vc = cv2.VideoCapture(1)  # 调用摄像头
        rval, frame = vc.read()
        bin_line_scale = []
        scale_locations = []
        fps = vc.get(cv2.CAP_PROP_FPS)  # 获取视频原帧率
        print("当前视频帧率为：",fps)
        start_time = time.time()  # 获取视频开始播放时的时间
        self.length, self.mean = 0,0
        while rval and self._is_running:  # 循环读取视频帧
            self.framenum += 1
            rval, frame = vc.read()
            if rval == False:
                break
            frame = cv2.resize(frame, (720, 404))
            if (time.time() - start_time) != 0:  # 实时显示帧数
                cv2.putText(frame, "FPS {0}".format(float('%.1f' % (self.framenum / (time.time() - start_time)))),
                            (500, 300),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                            2)
            if self.framenum % fps == 0 or self.framenum == 1: #一秒检测一次
                # 检测表盘
                det_result = detect(self.detector,frame)  # 得到目标检测结果
                if det_result:
                    for res in det_result:
                        xmin, ymin, w, h = res['position']  #左上角坐标、图片的长，宽
                        cv2.rectangle(frame, (xmin, ymin), (xmin + w, ymin + h), (0, 0, 255), 2)
                        sub_img = frame[ymin:ymin + h, xmin:xmin + w]
                        sub_img = cv2.resize(sub_img, (512, 512), cv2.INTER_LINEAR)  # 裁剪原图片，并resize尺寸
                        seg_result = segment_image(self.segmente,sub_img)  # 分割上述步骤得到的结果

                        # 对分割的图形进行腐蚀操作
                        seg_result = cv2.convertScaleAbs(seg_result)
                        erode_kernel = np.ones((3, 3), dtype=np.uint8)
                        seg_result = cv2.erode(seg_result, erode_kernel)
                        seg_result_img = Visualimg(seg_result)
                        # cv2.imshow("seg_img",seg_result_img)
                        # cv2.waitKey(1)

                        # 圆环转矩形
                        rectangle_meter = Circle2Rectangle(seg_result)
                        pointer_rect = copy.deepcopy(rectangle_meter)
                        pointer_rect[pointer_rect == 2] = 0  # 从原分割图中分出来指针图
                        first_scale_rect = copy.deepcopy(rectangle_meter)
                        first_scale_rect[first_scale_rect == 1] = 0
                        bin_line_scale = Rectangle2Line(first_scale_rect)
                        scale_locations = FindCenter(bin_line_scale, False)

                        # 整理得到的刻度值
                        scale_locations = check_scale(scale_locations,int(self.zongkedu/self.fenduzhi))
                        print('scale_locations', scale_locations)
                        pointer_rect = copy.deepcopy(rectangle_meter)  # 只有指针的图
                        pointer_rect[pointer_rect == 2] = 0  # 从原分割图中分离出来指针图
                        rectangle_meter_img_1 = Visualimg(pointer_rect)
                        pointer_rect = check_pointer_seg(rectangle_meter_img_1)
                        # rectangle_meter_img_2 = Visualimg(pointer_rect)
                        cv2.imshow("rectangle_meter_img_1",rectangle_meter_img_1)
                        cv2.imshow("rectangle_meter_img_2", pointer_rect)
                        cv2.waitKey(1)

                        # 把上述图片沿高度方向压缩成线状格式
                        bin_line_pointer = Rectangle2Line(pointer_rect)
                        # 找线状图形的中心点
                        pointer_locations = FindCenter(bin_line_pointer, True)
                        print('pointer_locations', pointer_locations)
                        # 读数
                        num_scales, num_PointerinRange = Reading(scale_locations,pointer_locations)
                        # 计算读数
                        ReadingRes = round((num_scales) * self.fenduzhi + num_PointerinRange * self.fenduzhi, 2)
                        self.reslist.append(ReadingRes)
                        print(self.reslist)
                        self.timelist.append(self.framenum/fps)
                        self.updata.emit([self.timelist,self.reslist])
                        # cv2.putText(frame, f"result:{ReadingRes}", (xmin+10, ymin),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),2)
                        # 把frame放在label上
                        self.videoshow.setScaledContents(True)
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        height, width, channel = frame.shape
                        bytesPerLine = 3 * width
                        qImg = QImage(frame.data, width, height, bytesPerLine, QImage.Format_RGB888)
                        self.videoshow.setPixmap(QPixmap.fromImage(qImg))
                else:
                    self.reslist.append(0)
                    print(self.reslist)
                    self.timelist.append(self.framenum / fps)
            else:
                # 把frame放在label上
                self.videoshow.setScaledContents(True)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height, width, channel = frame.shape
                bytesPerLine = 3 * width
                qImg = QImage(frame.data, width, height, bytesPerLine, QImage.Format_RGB888)
                self.videoshow.setPixmap(QPixmap.fromImage(qImg))
            time.sleep(1 / fps)  # 按原帧率播放
        vc.release()
        self.video_workend.emit()

    def get_len_span(self,scale_locations):
        span_list = np.diff(scale_locations)
        print(len(scale_locations), np.mean(span_list))
        return len(scale_locations), np.mean(span_list)

    def check_centerpoint(self,scale_locations,length ,mean):
        if len(scale_locations) > length:
            for i in range(1,length):
                if (scale_locations[i] - scale_locations[i - 1]) < (int(mean)-3):
                    print('scale_locations.pop(i):',scale_locations[i])
                    scale_locations.pop(i)
        else:
            pass
        return scale_locations

    def stop(self):
        self._is_running = False

    def GetFilePath(self,filepath):
        self.videopath = filepath


class Meter_Reading(QMainWindow,Ui_MainWindow):
    sendfile = pyqtSignal(str)   # 发送文件路径
    endwork = pyqtSignal()   # 工作结束信号
    sendvideo = pyqtSignal(str)
    danger_signal = pyqtSignal(str)

    def __init__(self):
        super(Meter_Reading, self).__init__()
        self.NUM = 0
        self.CurImgIndex = 0
        self.fenduzhi = 0.5 # 分度值，默认为0.5
        self.cameranum = 0  # 摄像头序号，默认为本地第一个摄像头
        self.zongkedu = 16  #表盘总刻度，默认为16
        self.setupUi(self)
        self.WorkStatus = 0  #记录工作状态
        self.num_show = 10
        self.workmode = 0    # 0表示未工作  1表示图片  2表示视频   3表示摄像头
        self.resultlist = []
        self.setAttribute(Qt.WA_TranslucentBackground) #设置窗口背景透明
        self.setWindowFlag(Qt.FramelessWindowHint) #设置窗口标志：隐藏窗口边框
        self.ModelInit()
        self.init_solt()
        self.init_plot()
        self.pic_0.mousePressEvent = lambda event: self.showImage(0)
        self.pic_1.mousePressEvent = lambda event: self.showImage(1)
        self.pic_2.mousePressEvent = lambda event: self.showImage(2)
        self.pic_3.mousePressEvent = lambda event: self.showImage(3)
        self.pic_result.mousePressEvent = lambda event: self.showImage(4)

    def showImage(self,index):
        self.CurImgIndex = index
        if len(resimg_list[index]) > 0:
            frame = cv2.cvtColor(resimg_list[self.CurImgIndex][self.NUM % len(resimg_list[self.CurImgIndex])], cv2.COLOR_BGR2RGB)
            height, width, channel = frame.shape
            bytesPerLine = 3 * width
            qImg = QImage(frame.data, width, height, bytesPerLine, QImage.Format_RGB888)
            self.frame_show.setPixmap(QPixmap.fromImage(qImg))

    def init_plot(self):
        # 处理视频的折线图
        self.LineFigure = Figure_Canvas()
        self.LineFigureLayout = QGridLayout(self.groupbox)
        self.LineFigureLayout.addWidget(self.LineFigure)
        self.LineFigure.ax.set_xlim(0,10)
        y_major_locator = MultipleLocator(1)
        self.LineFigure.ax.yaxis.set_major_locator(y_major_locator)
        self.LineFigure.ax.set_ylim(0, 5)

        # 处理摄像头的折线图
        self.LineFigure_camera = Figure_Canvas()
        self.LineFigureLayout_camera = QGridLayout(self.groupbox_camera)
        self.LineFigureLayout_camera.addWidget(self.LineFigure_camera)
        self.LineFigure_camera.ax.set_xlim(0, 10)
        y_major_locator_camera = MultipleLocator(1)
        self.LineFigure_camera.ax.yaxis.set_major_locator(y_major_locator_camera)
        self.LineFigure_camera.ax.set_ylim(0, 5)


    def init_solt(self):
        self.endwork.connect(self.WorkEnd)
        self.btn_exit.clicked.connect(self.close)
        self.btn_picwin.clicked.connect(self.choosepic)
        self.btn_videowin.clicked.connect(self.choosevideo)
        self.btn_setting.clicked.connect(self.settingwin)
        self.btn_run.clicked.connect(self.work)
        self.btn_left.clicked.connect(self.GetLeftPic)
        self.btn_right.clicked.connect(self.GetRightPic)
        self.danger_signal.connect(self.danger_dispose)
        self.btn_camera.clicked.connect(self.choosecamera)



    def settingdata(self,datalist):
        self.fenduzhi = datalist[0]
        self.zongkedu = datalist[1]
        self.cameranum = datalist[2]

        print('self.fenduzhi',self.fenduzhi,'self.zongkedu',self.zongkedu,'self.cameranum',self.cameranum)

    def settingwin(self):
        self.SetWin = SettingWin()  # 设置界面初始化
        self.SetWin.senddata.connect(self.settingdata)
        self.SetWin.show()


    def danger_dispose(self,danger_Status):
        pygame.init()
        pygame.mixer.music.load(r"meter_reading_res/warning.wav")
        if danger_Status == '1':
            if self.stackedWidget.currentIndex() == 1:
                self.reslist.setStyleSheet("border:2px solid;border-radius:0px;border-color: red;font-size:20px;font-weight:1000;font-family:微软雅黑;color:red;")
            elif self.stackedWidget.currentIndex() == 2:
                self.reslist_camera.setStyleSheet(
                    "border:2px solid;border-radius:0px;border-color: red;font-size:20px;font-weight:1000;font-family:微软雅黑;color:red;")
            pygame.mixer.music.play()
        else:
            if pygame.mixer.music.get_busy():
                pygame.mixer.music.stop()
            if self.stackedWidget.currentIndex() == 1:
                self.reslist.setStyleSheet("border:2px solid;border-radius:0px;border-color: green;font-size:20px;font-weight:1000;font-family:微软雅黑;color:black;")
            elif self.stackedWidget.currentIndex() == 2:
                self.reslist_camera.setStyleSheet(
                    "border:2px solid;border-radius:0px;border-color: green;font-size:20px;font-weight:1000;font-family:微软雅黑;color:black;")

    def GetLeftPic(self):
        if len(resimg_list[self.CurImgIndex]) == 0:
            return
        self.NUM -= 1
        if len(resimg_list[self.CurImgIndex]) > 0:
            frame = cv2.cvtColor(resimg_list[self.CurImgIndex][self.NUM % len(resimg_list[self.CurImgIndex])], cv2.COLOR_BGR2RGB)
            height, width, channel = frame.shape
            bytesPerLine = 3 * width
            qImg = QImage(frame.data, width, height, bytesPerLine, QImage.Format_RGB888)
            self.frame_show.setPixmap(QPixmap.fromImage(qImg))


    def GetRightPic(self):
        if len(resimg_list[self.CurImgIndex]) == 0:
            return
        self.NUM += 1
        if len(resimg_list[self.CurImgIndex]) > 0:
            frame = cv2.cvtColor(resimg_list[self.CurImgIndex][self.NUM % len(resimg_list[self.CurImgIndex])],
                                 cv2.COLOR_BGR2RGB)
            height, width, channel = frame.shape
            bytesPerLine = 3 * width
            qImg = QImage(frame.data, width, height, bytesPerLine, QImage.Format_RGB888)
            self.frame_show.setPixmap(QPixmap.fromImage(qImg))


    def work(self):
        if self.WorkStatus == 1:
            self.endwork.emit()
            return

        if self.stackedWidget.currentIndex() == 0:   #处理图片
            # 清空过程图片
            resimg_list= [[] for i in range(5)]
            self.picthread = PicWorkThread(self.detector, self.segmenter, self.pic_1, self.pic_2, self.pic_3,
                                           self.pic_result, self.frame_show,self.fenduzhi,self.zongkedu)  # 图片处理线程
            self.sendfile.connect(self.picthread.GetFilePath)
            self.picthread.workend.connect(self.WorkEnd)
            self.sendfile.emit(self.PicPath)
            self.picthread.start()
            self.workmode = 1

        elif self.stackedWidget.currentIndex() == 1:  # 处理视频
            self.videothread = VideoWorkThread(self.detector, self.segmenter,self.video_show,self.groupbox,self.fenduzhi,self.zongkedu)
            self.sendvideo.connect(self.videothread.GetFilePath)
            self.videothread.updata.connect(self.updata)
            self.videothread.video_workend.connect(self.WorkEnd)
            self.sendvideo.emit(self.VideoPath)
            self.videothread.start()
            self.workmode = 2

        elif self.stackedWidget.currentIndex() == 2:  # 处理摄像头
            self.camerathread = cameraWorkThread(self.detector, self.segmenter,self.video_show_camera,self.groupbox_camera,self.fenduzhi,self.cameranum,self.zongkedu)
            self.camerathread.updata.connect(self.updata)
            self.camerathread.video_workend.connect(self.WorkEnd)
            self.camerathread.start()
            self.workmode = 3

        self.btn_run.setStyleSheet("QPushButton{border-radius:10;background-color: rgb(255, 0, 0);qproperty-icon: url(meter_reading_res/pause2.png)}")
        self.WorkStatus = 1


    def updata(self,list):
        danger_state = 0 # 0代表不危险
        self.num_show = self.num_show + 1  # 每进一次循环，x轴的显示范围+1
        self.LineFigure.ax.set_xlim(0, self.num_show)  # 设置x轴的显示范围+1
        self.line = Line2D(list[0],list[1], linewidth=1, color='red')  # 画曲线图
        if self.stackedWidget.currentIndex() == 1:
            self.reslist.setText(f"上一秒读数结果为：{list[1][-1]}")
            self.resultlist = list[1]  # 得到所有结果
            self.LineFigure.ax.add_line(self.line)  # 把曲线图添加到画布上
            self.LineFigure.draw()  # 画图
        elif self.stackedWidget.currentIndex() == 2:
            self.reslist_camera.setText(f"上一秒读数结果为：{list[1][-1]}")
            self.resultlist = list[1]  # 得到所有结果
            self.LineFigure_camera.ax.add_line(self.line)  # 把曲线图添加到画布上
            self.LineFigure_camera.draw()  # 画图


        # KEEP_RATE = 0.01  # 保压的比例
        # DESENCE_RATE = 0.3  # 骤降的比例
        if len(list[1])>2:
            change_value = list[1][-2] - list[1][-1]
            if abs(change_value) <= list[1][-2]*KEEP_RATE:
                if self.stackedWidget.currentIndex() == 1:
                    self.reslist.setText(self.reslist.text()+"\n当前状态：保压")
                elif self.stackedWidget.currentIndex() == 2:
                    self.reslist_camera.setText(self.reslist_camera.text() + "\n当前状态：保压")
                danger_state = 0
            elif change_value > 0 and change_value>list[1][-2] * DESENCE_RATE:
                if self.stackedWidget.currentIndex() == 1:
                    self.reslist.setText(self.reslist.text()+"\n当前状态：骤降")
                elif self.stackedWidget.currentIndex() == 2:
                    self.reslist_camera.setText(self.reslist_camera.text() + "\n当前状态：骤降")
                danger_state = 1 # 骤降表示危险，设置为1
            elif change_value > 0 and change_value>list[1][-2] * KEEP_RATE:
                danger_state = 0
                if self.stackedWidget.currentIndex() == 1:
                    self.reslist.setText(self.reslist.text()+"\n当前状态：下降")
                elif  self.stackedWidget.currentIndex() == 2:
                    self.reslist_camera.setText(self.reslist_camera.text() + "\n当前状态：下降")
            elif change_value < 0 and abs(change_value) > list[1][-2] * KEEP_RATE:
                danger_state = 0
                if self.stackedWidget.currentIndex() == 1:
                    self.reslist.setText(self.reslist.text()+"\n当前状态：上升")
                elif self.stackedWidget.currentIndex() == 2:
                    self.reslist_camera.setText(self.reslist_camera.text() + "\n当前状态：上升")
        self.danger_signal.emit(str(danger_state))


    def WorkEnd(self):
        if self.workmode == 1:
            self.picthread.terminate()
            self.PicFinish_dialog = diydialog()
            self.PicFinish_dialog.savepath = 'output\Image'
            self.PicFinish_dialog.info_content.setText(f"识别结束!\n结果保存在{self.PicFinish_dialog.savepath}文件夹")
            self.PicFinish_dialog.show()

        elif self.workmode == 2:
            self.videothread.stop()
            self.videothread.wait()
            self.VideoFinish_dialog = diydialog()
            self.VideoFinish_dialog.savepath = 'output\RunCondition'
            self.VideoFinish_dialog.info_content.setText(f"识别结束!\n结果保存在RunCondition文件夹")
            self.VideoFinish_dialog.show()
            self.maketxt(self.resultlist)


        elif self.workmode == 3:
            self.camerathread.stop()
            self.camerathread.wait()
            self.VideoFinish_dialog = diydialog()
            self.VideoFinish_dialog.savepath = 'output\RunCondition'
            self.VideoFinish_dialog.info_content.setText(f"识别结束!\n结果保存在RunCondition文件夹")
            self.VideoFinish_dialog.show()
            self.maketxt(self.resultlist)

        self.workmode = 0
        self.WorkStatus = 0
        self.btn_run.setStyleSheet(
            "QPushButton{border-radius:10;background-color: rgb(19, 34, 122);qproperty-icon: url(meter_reading_res/pause1.png)}")

    def maketxt(self,res):
        result_index = {'increase': [], 'steady': [], 'plummet': [], 'decrease': []}
        start = 0
        last_status = 0
        currect_status = 0 #0为初始，1为上升，2为保持，3为骤降，4为下降
        for i in range(1,len(res)):                                              # res即为视频检测线程每秒检测的结果
            if res[i]-res[i-1] > 0 and res[i]-res[i-1] > res[i-1] * KEEP_RATE:   # 上升阶段
                currect_status = 1
            elif abs(res[i]-res[i-1]) < res[i] * KEEP_RATE:                      # 保压阶段
                currect_status = 2
            elif res[i]-res[i-1] < 0 and abs(res[i]-res[i-1]) > res[i-1] * DESENCE_RATE: #骤降阶段
                currect_status = 3
            elif res[i]-res[i-1] < 0 and abs(res[i]-res[i-1]) > res[i-1] * KEEP_RATE:   # 下降阶段
                currect_status = 4

            if currect_status != last_status :
                if last_status == 1 and last_status != 0:
                    result_index['increase'].append((start,i))
                elif last_status == 2 and last_status != 0:
                    result_index['steady'].append((start, i))
                elif last_status == 3 and last_status != 0:
                    result_index['plummet'].append((start, i))
                elif last_status == 4 and last_status != 0:
                    result_index['decrease'].append((start, i))
                last_status = currect_status
                start = i

            #处理视频结尾的那个状态
            if i == len(res)-1 :
                if last_status == 1:
                    result_index['increase'].append((start,i+1))
                elif last_status == 2:
                    result_index['steady'].append((start, i+1))
                elif last_status == 3:
                    result_index['plummet'].append((start, i+1))
                elif last_status == 4:
                    result_index['decrease'].append((start, i+1))

        print(result_index)
        file = open('output/RunCondition/run.txt','w+')
        file.write(f"输入的视频总时间为{self.convert(len(res))}\n\n")
        file.write("其中读数处于上升状态的时间段为:\n")
        if len(result_index['increase']) == 0:
            file.write("无")
        else:
            for index in result_index['increase']:
                file.write(f"{self.convert(index[0])} - {self.convert(index[1])},读数从{res[index[0]-1]}上升到了{res[index[1]-1]}\n")
        file.write("\n其中读数处于稳定状态的时间段为:\n")
        if len(result_index['steady']) == 0:
            file.write("无")
        else:
            for index in result_index['steady']:
                file.write(f"{self.convert(index[0])} - {self.convert(index[1])},读数在{round(sum(res[index[0]-1:index[1]-1])/len(res[index[0]-1:index[1]-1]),1)}附近稳定\n")
        file.write(f"\n其中读数处于骤降状态的时间段为:\n")
        if len(result_index['plummet']) == 0:
            file.write("无")
        else:
            for index in result_index['plummet']:
                file.write(f"{self.convert(index[0])} - {self.convert(index[1])},读数从{res[index[0]-1]}骤降到了{res[index[1]-1]}\n")
        file.write("\n其中读数处于下降状态的时间段为:\n")
        if len(result_index['decrease']) == 0:
            file.write("无")
        else:
            for index in result_index['decrease']:
                file.write(f"{self.convert(index[0])} - {self.convert(index[1])},读数从{res[index[0]-1]}下降到了{res[index[1]-1]}\n")

        file.close()

    def convert(self,seconds):
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        return "%02d:%02d:%02d" % (h, m, s)

    #选择视频
    def choosevideo(self):
        self.stackedWidget.setCurrentIndex(1)
        self.VideoPath = QtWidgets.QFileDialog.getOpenFileName(self, "选择视频", "","*.mp4;;*.avi;;All Files(*)")[0]
        if self.VideoPath != '':
            self.filepath.setText(self.VideoPath)

    #选择摄像头
    def choosecamera(self):
        self.stackedWidget.setCurrentIndex(2)

    #选择图片
    def choosepic(self):
        self.stackedWidget.setCurrentIndex(0)
        self.PicPath = QtWidgets.QFileDialog.getOpenFileName(self, "选择图片", "", "*.JPG;;*.PNG;;All Files(*)")[0]
        if self.PicPath != '':
            self.filepath.setText(self.PicPath)
            self.step_0 = cv2.imdecode(np.fromfile(self.PicPath, dtype=np.uint8), -1)
            self.pic_0.setScaledContents(True)
            # self.frame_show.setScaledContents(True) #图片适应画面
            frame = cv2.cvtColor(self.step_0, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (680, 370), cv2.INTER_LINEAR)
            height, width, channel = frame.shape
            bytesPerLine = 3 * width
            qImg = QImage(frame.data, width, height, bytesPerLine, QImage.Format_RGB888)
            self.pic_0.setPixmap(QPixmap.fromImage(qImg))
            self.frame_show.setPixmap(QPixmap.fromImage(qImg))

    def ModelInit(self):
        self.detector = detect_model_load((640, 640), False)  # yolov5检测模型加载
        self.segmenter = seg_model_load()  # unet检测模型加载

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.m_flag = True
            self.m_Position = event.globalPos() - self.pos()  # 获取鼠标相对窗口的位置
            event.accept()
            self.setCursor(QCursor(Qt.OpenHandCursor))  # 更改鼠标图标

    def mouseMoveEvent(self, QMouseEvent):
        if Qt.LeftButton and self.m_flag:
            self.move(QMouseEvent.globalPos() - self.m_Position)  # 更改窗口位置
            QMouseEvent.accept()

    def mouseReleaseEvent(self, QMouseEvent):
        self.m_flag = False
        self.setCursor(QCursor(Qt.ArrowCursor))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = Meter_Reading()
    win.show()
    sys.exit(app.exec_())