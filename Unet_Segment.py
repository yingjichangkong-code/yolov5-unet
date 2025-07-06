from nets.unet import Unet as unet
import torch
import torch.nn.functional as F
from torch import nn
import time
from PIL import Image
import numpy as np

#颜色
colors = [
    [255, 255, 255],  # background 白色
    [0, 0, 0],  # class 1     黑色
    [255, 0, 0],  # class 2   红色
    [0,255,0 ],  # class 3   绿色
]


def seg_model_load():
    net = unet(num_classes=3, backbone="vgg")
    model_path = 'best_epoch_weights.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.load_state_dict(torch.load(model_path, map_location=device))
    net = net.eval()
    net = nn.DataParallel(net)
    net = net.cuda()
    print('{} model, and classes loaded.'.format(model_path))
    return net

def segment_image(net,image):
    image_data = cvtColor(image)
    image_data, nw, nh = resize_image(image, (512,512))
    image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)
    with torch.no_grad():
        images = torch.from_numpy(image_data)
        images = images.cuda()
        # 图片传入网络进行预测
        seg_start = time.time()
        pr = net(images)[0]
        seg_finish = time.time()
        # print(f"分割表盘用时：{1000 * (seg_finish - seg_start)}ms")
        pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()
        pr = pr.argmax(axis=-1)
    return pr


def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image

def resize_image(image, size):
    image = Image.fromarray(image)
    iw, ih = image.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))

    return new_image, nw, nh

def preprocess_input(image):
    image /= 255.0
    return image


def Visualimg(ImgList):
    color_image = np.zeros((ImgList.shape[0], ImgList.shape[1], 3), dtype=np.uint8)
    for i in range(ImgList.shape[0]):
        for j in range(ImgList.shape[1]):
            color_image[i, j, :] = colors[ImgList[i, j]]
    return color_image