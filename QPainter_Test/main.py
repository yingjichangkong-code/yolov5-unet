import sys
from meter import Ui_MainWindow
from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow
from PyQt5.QtGui import QPainter, QColor, QPen, QFont, QRadialGradient, QPolygon
from PyQt5.QtCore import Qt, QRectF


class SettingWin(Ui_MainWindow,QMainWindow):
    def __init__(self):
        super(SettingWin, self).__init__()
        self.setupUi(self)
        self.setMinimumSize(600, 600)
        # 颜色设置
        self.pieColorStart = QColor(0, 0, 0)  # 绿色
        self.pieColorMid = QColor(0, 0, 0)  # 蓝色
        self.pieColorEnd = QColor(0, 0, 0)  # 红色
        self.pointerColor = QColor(0, 0, 0)  # 青色

        self.startAngle = 45
        self.endAngle = 45
        # 设置字符
        self.font = QFont("宋体", 8)
        self.font.setBold(True)

    def paintEvent(self, event):
        # 坐标轴变换 默认640*480
        width = self.width()
        height = self.height()

        painter = QPainter(self)  # 初始化painter
        painter.translate(width / 2, height / 2)  # 坐标轴变换，调用translate()将坐标原点平移至窗口中心

        # 坐标刻度自适应
        side = min(width, height)
        painter.scale(side / 200.0, side / 200.0)
        # 本项目中将坐标缩小为side/200倍，即画出length=10的直线，其实际长度应为10*(side/200)。

        # 启用反锯齿，使画出的曲线更平滑
        painter.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing)
        painter.begin(self)

        # 开始画图
        self.drawColorPie(painter)
        self.drawPointerIndicator(painter)


    def drawColorPie(self, painter):  # 绘制三色环
        painter.save()  # save()保存当前坐标系
        # print("drawColorPie")
        # 设置扇形部分区域
        radius = 80  # 半径
        painter.setPen(Qt.NoPen)
        rect = QRectF(-radius, -radius, radius * 2, radius * 2)  # 扇形所在圆区域

        # 计算三色圆环范围角度。green：blue：red = 1：2：1
        angleAll = 360.0 - self.startAngle - self.endAngle  # self.startAngle = 45, self.endAngle = 45
        angleStart = angleAll * 0.25
        angleMid = angleAll * 0.5
        angleEnd = angleAll * 0.25

        # 圆的中心部分填充为透明色，形成环的样式
        rg = QRadialGradient(0, 0, radius, 0, 0)  # 起始圆心坐标，半径，焦点坐标
        ratio = 0.95  # 透明：实色 = 0.8 ：1

        # 绘制绿色环
        rg.setColorAt(0, Qt.transparent)  # 透明色
        rg.setColorAt(ratio, Qt.transparent)
        rg.setColorAt(ratio + 0.01, self.pieColorStart)
        rg.setColorAt(1, self.pieColorStart)

        painter.setBrush(rg)
        painter.drawPie(rect, (270 - self.startAngle - angleStart) * 16, angleStart * 16)

        # 绘制蓝色环
        rg.setColorAt(0, Qt.transparent)
        rg.setColorAt(ratio, Qt.transparent)
        rg.setColorAt(ratio + 0.01, self.pieColorMid)
        rg.setColorAt(1, self.pieColorMid)

        painter.setBrush(rg)
        painter.drawPie(rect, (270 - self.startAngle - angleStart - angleMid) * 16, angleMid * 16)

        # 绘制红色环
        rg.setColorAt(0, Qt.transparent)
        rg.setColorAt(ratio, Qt.transparent)
        rg.setColorAt(ratio + 0.01, self.pieColorEnd)
        rg.setColorAt(1, self.pieColorEnd)

        painter.setBrush(rg)
        painter.drawPie(rect, (270 - self.startAngle - angleStart - angleMid - angleEnd) * 16, angleEnd * 16)

        painter.restore()  # restore()恢复坐标系

    def drawPointerIndicator(self, painter):
        painter.save()
        # 绘制指针
        # print("drawPointerIndicator")
        radius = 58  # 指针长度
        painter.setPen(Qt.NoPen)
        painter.setBrush(self.pointerColor)

        # (-5, 0), (0, -8), (5, 0)和（0, radius) 四个点绘出指针形状
        # 绘制多边形做指针
        pts = QPolygon()
        pts.setPoints(-5, 0, 0, -8, 5, 0, 0, radius)
        # print("radius:" + str(radius))

        # 旋转指针，使得指针起始指向为0刻度处
        painter.rotate(self.startAngle)
        degRotate = (360.0 - self.startAngle - self.endAngle) / (self.maxValue - self.minValue) \
                    * (self.currentValue - self.minValue)
        painter.rotate(degRotate)
        painter.drawConvexPolygon(pts)
        painter.restore()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    Settingwin = SettingWin()
    Settingwin.show()
    sys.exit(app.exec_())


