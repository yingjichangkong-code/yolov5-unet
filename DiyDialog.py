from PyQt5.QtWidgets import QMainWindow, QApplication, QDialogButtonBox
from PyQt5.QtGui import QCursor
import sys
import os
from Dialog import Ui_Dialog
from PyQt5.QtCore import Qt, pyqtSignal

class diydialog(Ui_Dialog,QMainWindow):
    def __init__(self):
        super(diydialog, self).__init__()
        self.setAttribute(Qt.WA_TranslucentBackground)  # 设置窗口背景透明
        self.setWindowFlag(Qt.FramelessWindowHint)  # 设置窗口标志：隐藏窗口边框
        self.setupUi(self)
        self.savepath = ''                          # 存放结果的文件夹路径
        self.info_content.setText("识别结束!\n结果保存在output/Image文件夹中")  # 设置对话框中显示的内容

    # 设置accept按钮功能，即打开存放结果的文件夹
    def accept(self):
        if self.savepath == '':
            self.close()
            return
        work_path = os.getcwd()
        start_directory = os.path.join(work_path,self.savepath)
        os.system("explorer.exe %s" % start_directory)    # 打开文件夹
        self.close()


    # 设置reject按钮功能
    def reject(self):
        self.close()   # 点击按钮之后关闭当前对话框

    # 重写窗口的 鼠标按压事件，实现鼠标拖动窗口
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

