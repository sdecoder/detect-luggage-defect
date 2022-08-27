import tritonclient.http as tritonhttpclient

import argparse
import os
import platform
import sys
from pathlib import Path

import cv2

import torch
import torch.backends.cudnn as cudnn
import onnx
import onnxruntime
import random

from utils.general import scale_coords, non_max_suppression
from utils.torch_utils import select_device


def plot_one_box(x, image, color=None, label=None, line_thickness=None):
  # Plots one bounding box on image img
  tl = line_thickness or round(0.002 * (image.shape[0] + image.shape[1]) / 2) + 1  # line/font thickness
  color = color or [random.randint(0, 255) for _ in range(3)]
  c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
  cv2.rectangle(image, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
  if label:
    tf = max(tl - 1, 1)  # font thickness
    t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
    c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
    cv2.rectangle(image, c1, c2, color, -1, cv2.LINE_AA)  # filled
    cv2.putText(image, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def render_result(image_file_path, pred):

  image = cv2.imread(image_file_path)
  try:
    height, width, channels = image.shape
  except:
    print('no shape info.')
    return 0

  imgsz = (640, 640)  # inference size (height, width)
  for i, det in enumerate(pred):  # per image

    if len(det):
      print(f'[trace] object detected!')
      det[:, :4] = scale_coords(imgsz, det[:, :4], image.shape).round()

      for *xyxy, conf, cls in reversed(det):
        c = int(cls)  # integer class
        print(f"f[trace] coordinates: {xyxy}")
        print(f'[trace] found class code: {c}')

        from utils.general import xyxy2xywh
        gn = torch.tensor(image.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh

        x_center_ratio = float(xywh[0])
        y_center_ratio = float(xywh[1])
        width_ratio = float(xywh[2])
        height_ratio = float(xywh[3])

        x_center, y_center, w, h = x_center_ratio * width, y_center_ratio * height, width_ratio * width, height_ratio * height
        x1 = round(x_center - w / 2)
        y1 = round(y_center - h / 2)
        x2 = round(x_center + w / 2)
        y2 = round(y_center + h / 2)

        # if class_idx == 0:
        #     draw_people_tangle = cv2.rectangle(image, (x1,y1),(x2,y2),(0,0,255),2)   # 画框操作  红框  宽度为1
        #     cv2.imwrite(save_file_path,draw_people_tangle)  #画框 并保存
        # elif class_idx == 1:
        #     draw_car_tangle = cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,0),2)     # 画框操作  绿框  宽度为1
        #     cv2.imwrite(save_file_path,draw_car_tangle)  #画框 并保存
        if c == 28:
          # this is a luggage
          color = [0, 255, 0]
        else:
          # this is a damage
          color = [255, 0, 0]

        plot_one_box([x1, y1, x2, y2], image, color=color, label=None, line_thickness=2)


        # cv2.imwrite(save_file_path, image)
  cv2.imshow('cv2 show image', image)
  cv2.waitKey(0)
  pass


print(f'[trace] defining the parameters')
VERBOSE = False
input_name = 'images'
input_shape = (1, 3, 640, 640)
input_dtype = 'FP32'
output_name = 'output'
model_name = 'yolov5-luggage-defect'
url = 'localhost:8000'
model_version = '1'

print(f'[trace] creating objects')
triton_client = tritonhttpclient.InferenceServerClient(url=url, verbose=VERBOSE)
model_metadata = triton_client.get_model_metadata(model_name=model_name, model_version=model_version)
model_config = triton_client.get_model_config(model_name=model_name, model_version=model_version)


import numpy as np
from PIL import Image
from torchvision import transforms

print(f'[trace] preparing input data')


def _load_img(image_path):
  from PIL import Image
  import torchvision.transforms

  im = Image.open(image_path)
  im = im.resize((640, 640))
  # im.show()
  trans_to_tensor = torchvision.transforms.ToTensor()
  _tensor = trans_to_tensor(im)
  return _tensor

image_file_path = '/home/noname/projects/deeplearning/datasets/luggage/images/train/000000000000.jpg'
image_tensor = _load_img(image_file_path)
image_numpy = image_tensor.cpu().numpy()
image_numpy = image_numpy[None]
print(image_numpy.shape)

print(f'[trace] start to interact with Triton server...')
input0 = tritonhttpclient.InferInput(input_name, input_shape, input_dtype)
input0.set_data_from_numpy(image_numpy, binary_data=False)

output = tritonhttpclient.InferRequestedOutput(output_name, binary_data=False)
response = triton_client.infer(model_name, model_version=model_version,
                               inputs=[input0], outputs=[output])


bs = 1  # batch_size
conf_thres = 0.25  # confidence threshold
iou_thres = 0.45  # NMS IOU threshold
classes = None  # filter by class: --class 0, or --class 0 2 3
agnostic_nms = False  # class-agnostic NMS
max_det = 1000  # maximum detections per image

device = select_device('')
pred = response.as_numpy(output_name)
pred = torch.from_numpy(pred).to(device)
pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
# img = cv2.imread(image_file_path, cv2.IMREAD_COLOR) / 255
# show_img = cv2.imread(image_file_path, cv2.IMREAD_ANYCOLOR)
# cv2.imshow('output image', show_img) # this is for output
# cv2.waitKey(0)

render_result(image_file_path, pred)
