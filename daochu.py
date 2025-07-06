import torch
import torchvision
from yolov5_Detect import detect_model_load,detect

detect_model = detect_model_load((640, 640),False) #yolov5检测模型加载

# An instance of your model.
model = detect_model

# Switch the model to eval model
model.eval()

# An example input you would normally provide to your model's forward() method.
example = torch.rand(1, 3, 640, 640)
device = torch.device('cuda:0')
example = example.to(device)
# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
traced_script_module = torch.jit.trace(model, example)


# Save the TorchScript model
traced_script_module.save("traced_resnet_model.pt")


def circle_to_rectangle(self, seg_results):
    """将圆形表盘的预测结果label_map转换成矩形

    圆形到矩形的计算方法：
        因本案例中两种表盘的刻度起始值都在左下方，故以圆形的中心点为坐标原点，
        从-y轴开始逆时针计算极坐标到x-y坐标的对应关系：
          x = r + r * cos(theta)
          y = r - r * sin(theta)
        注意：
            1. 因为是从-y轴开始逆时针计算，所以r * sin(theta)前有负号。
            2. 还是因为从-y轴开始逆时针计算，所以矩形从上往下对应圆形从外到内，
               可以想象把圆形从-y轴切开再往左右拉平时，圆形的外围是上面，內围在下面。

    参数：
        seg_results (list[dict])：分割模型的预测结果。

    返回值：
        rectangle_meters (list[np.array])：矩形表盘的预测结果label_map。

    """
    rectangle_meters = list()
    for i, seg_result in enumerate(seg_results):
        label_map = seg_result['label_map']
        # rectangle_meter的大小已经由预先设置的全局变量RECTANGLE_HEIGHT, RECTANGLE_WIDTH决定
        rectangle_meter = np.zeros(
            (RECTANGLE_HEIGHT, RECTANGLE_WIDTH), dtype=np.uint8)
        for row in range(RECTANGLE_HEIGHT):
            for col in range(RECTANGLE_WIDTH):
                theta = PI * 2 * (col + 1) / RECTANGLE_WIDTH
                # 矩形从上往下对应圆形从外到内
                rho = CIRCLE_RADIUS - row - 1
                y = int(CIRCLE_CENTER[0] + rho * math.cos(theta) + 0.5)
                x = int(CIRCLE_CENTER[1] - rho * math.sin(theta) + 0.5)
                rectangle_meter[row, col] = label_map[y, x]
        rectangle_meters.append(rectangle_meter)
    return rectangle_meters
