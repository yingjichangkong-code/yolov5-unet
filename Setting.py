import sys
from Settingui import Ui_MainWindow
from PyQt5.QtWidgets import QMainWindow, QApplication, QDialogButtonBox
from PyQt5.QtCore import Qt,pyqtSignal
from PyQt5.QtGui import QCursor
from DiyDialog import diydialog   #自定义对话框

class SettingWin(Ui_MainWindow,QMainWindow):
    senddata = pyqtSignal(list)              # 设置完成之后发送信号
    def __init__(self):
        super(SettingWin, self).__init__()
        self.setupUi(self)
        self.setAttribute(Qt.WA_TranslucentBackground)  # 设置窗口背景透明
        self.setWindowFlag(Qt.FramelessWindowHint)  # 设置窗口标志：隐藏窗口边框
        self.btn_finishsetting.clicked.connect(self.sendinfo)

    def sendinfo(self):
        fenduzhi = float(self.fenduzhi.text())
        camera_path = self.camera_path.text()
        zongkekdu = int(self.zongkedu.text())
        self.senddata.emit([fenduzhi,zongkekdu,camera_path])   # 触发发送信号，把当前设置的数值发送给主界面
        # 调用自定义对话框
        self.setting_finish = diydialog()
        # 改变accept按钮显示的文字
        self.setting_finish.buttonbox.button(QDialogButtonBox.Ok).setText("确定")
        # 设置对话框的内容
        self.setting_finish.info_content.setText(f"设置完成!\n当前分度值为: {fenduzhi}\n当前总刻度值为: {fenduzhi}\n当前摄像头序号为: {camera_path}")
        self.setting_finish.show()                   # 显示对话框
        self.close()

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
    Settingwin = SettingWin()
    Settingwin.show()
    sys.exit(app.exec_())


