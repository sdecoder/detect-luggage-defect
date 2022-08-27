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

import numpy as np
from torchvision import transforms

from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.general import print_args, non_max_suppression, scale_coords, check_img_size
from utils.plots import Annotator
from utils.torch_utils import select_device

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
  sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# common definition
image_file_path = '/home/noname/projects/deeplearning/datasets/luggage/images/train/000000000025.jpg'
bs = 1  # batch_size
conf_thres = 0.10  # confidence threshold
iou_thres = 0.25  # NMS IOU threshold
classes = None  # filter by class: --class 0, or --class 0 2 3
agnostic_nms = False  # class-agnostic NMS
max_det = 1000  # maximum detections per image
device = select_device('')

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
        # the color mode is BGR
        if c == 0:
          # this is a luggage
          color = [0, 255, 0]
        else:          # this is a damage
          color = [0, 0, 255]

        plot_one_box([x1, y1, x2, y2], image, color=color, label=None, line_thickness=2)


        # cv2.imwrite(save_file_path, image)
  cv2.imshow('composed result image', image)
  cv2.waitKey(0)
  pass


def _onnx_infer():
  image_width = 640
  image_height = 640

  # onnx_file_path = '/home/noname/projects/deeplearning/yolov5-luggage-defect-detection/runs/train/exp11/weights/best.onnx'
  #onnx_file_path = '/home/noname/projects/deeplearning/yolov5-luggage-defect-detection/model_binaries/yolov5x.onnx'
  onnx_file_path = '/home/noname/projects/deeplearning/yolov5-luggage-defect-detection/runs/train/exp13/weights/best.onnx'
  #onnx_file_path = '/home/noname/projects/deeplearning/yolov5-luggage-defect-detection/runs/train/exp13/weights/last.onnx'
  if not os.path.exists(onnx_file_path):
    print(f'[trace] target onnx file does NOT exist: {onnx_file_path}')
    exit(-1)

  dim = (image_width, image_height)  # resize image
  from PIL import Image
  im = Image.open(image_file_path)
  im = im.resize(dim)
  trans = transforms.ToTensor()
  im = trans(im)
  im = torch.unsqueeze(im, 0)

  def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

  ort_session = onnxruntime.InferenceSession(onnx_file_path)
  # compute ONNX Runtime output prediction
  ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(im)}
  ort_outs = ort_session.run(None, ort_inputs)

  # compare ONNX Runtime and PyTorch results
  pred = ort_outs[0]
  pred = torch.from_numpy(pred).to(device)
  pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
  # img = cv2.imread(image_file_path, cv2.IMREAD_COLOR) / 255
  # show_img = cv2.imread(image_file_path, cv2.IMREAD_ANYCOLOR)
  # cv2.imshow('output image', show_img) # this is for output
  # cv2.waitKey(0)

  render_result(image_file_path, pred)
  pass


def _pt_model_infer():
  '''
  im = torch.from_numpy(im).to(device)
  im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
  im /= 255  # 0 - 255 to 0.0 - 1.0
  if len(im.shape) == 3:
    im = im[None]  # expand for batch dim

  '''

  image_width = 640
  image_height = 640
  '''
  img = cv2.imread(image_file_path, cv2.IMREAD_ANYCOLOR)
  old_shape = img.shape
  old_shape = [old_shape[0], old_shape[1]] # height width
  

  resized_image = img / 255.0
  resized_image = resized_image.astype(np.float32)
  if len(resized_image) == 3:
    resized_image = resized_image[None]  # expand for batch dim
  
  '''
  augment = False  # augmented inference
  dnn = False  # use OpenCV DNN for ONNX inference
  data = f'{ROOT}/data/coco128.yaml'  # dataset.yaml path
  visualize = False
  weights = f'{ROOT}/yolov5s.pt'  # model.pt path(s)
  half = False  # use FP16 half-precision inference
  model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
  stride, names, pt = model.stride, model.names, model.pt
  imgsz = (640, 640)  # inference size (height, width)
  imgsz = check_img_size(imgsz, s=stride)  # check image size
  dataset = LoadImages(image_file_path, img_size=imgsz, stride=stride, auto=pt)
  seen = 0

  for path, im, im0s, vid_cap, s in dataset:
    im = torch.from_numpy(im).to(device)
    im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
      im = im[None]  # expand for batch dim

    pred = model(im, augment=augment, visualize=visualize)
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    for i, det in enumerate(pred):  # per image
      seen += 1
      if len(det):
        print(f'[trace] object detected!')
        for *xyxy, conf, cls in reversed(det):
          c = int(cls)  # integer class
          print(f'[trace] found class code: {c}')

  print(f'[trace] end of func')
  return

  line_thickness = 1
  from utils.plots import Annotator, colors, save_one_box
  annotator = Annotator(show_img, line_width=line_thickness, example="")

  for i, det in enumerate(pred):  # per image
    seen += 1

    if len(det):
      print(f'[trace] object detected!')
      # Write results
      det[:, :4] = scale_coords(imgsz, det[:, :4], im0.shape).round()
      for *xyxy, conf, cls in reversed(det):
        c = int(cls)  # integer class
        print(f'[trace] found class code: {c}')
        label = None
        annotator.box_label(xyxy, label, color=colors(c, True))

    #  im0 = annotator.result()
    #  cv2.imwrite('./test.jpg', im0)
    # Rescale boxes from img_size to im0 size

    pass

  print(f"[trace] done with ONNX inference testing")
  pass


def _cvt_onnx_to_trt():
  pass


class HostDeviceMem(object):
  def __init__(self, host_mem, device_mem):
    self.host = host_mem
    self.device = device_mem

  def __str__(self):
    return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

  def __repr__(self):
    return self.__str__()


def allocate_buffers_for_encoder(engine):
  import tensorrt as trt
  from PIL import Image
  import pycuda.driver as cuda
  import pycuda.autoinit
  import numpy as np
  import time
  import os
  import cv2
  import torch
  import argparse
  """Allocates host and device buffer for TRT engine inference.
  This function is similair to the one in common.py, but
  converts network outputs (which are np.float32) appropriately
  before writing them to Python buffer. This is needed, since
  TensorRT plugins doesn't support output type description, and
  in our particular case, we use NMS plugin as network output.
  Args:
      engine (trt.ICudaEngine): TensorRT engine
  Returns:
      inputs [HostDeviceMem]: engine input memory
      outputs [HostDeviceMem]: engine output memory
      bindings [int]: buffer to device bindings
      stream (cuda.Stream): cuda stream for engine inference synchronization
  """
  print('[trace] reach func@allocate_buffers')
  inputs = []
  outputs = []
  bindings = []
  stream = cuda.Stream()

  binding_to_type = {}
  binding_to_type['images'] = np.float32
  binding_to_type['output'] = np.float32

  # Current NMS implementation in TRT only supports DataType.FLOAT but
  # it may change in the future, which could brake this sample here
  # when using lower precision [e.g. NMS output would not be np.float32
  # anymore, even though this is assumed in binding_to_type]

  for binding in engine:
    print(f'[trace] current binding: {str(binding)}')
    _binding_shape = engine.get_binding_shape(binding)
    _volume = trt.volume(_binding_shape)
    size = _volume * engine.max_batch_size
    print(f'[trace] current binding size: {size}')
    dtype = binding_to_type[str(binding)]
    # Allocate host and device buffers
    host_mem = cuda.pagelocked_empty(size, dtype)
    device_mem = cuda.mem_alloc(host_mem.nbytes)
    # Append the device buffer to device bindings.
    bindings.append(int(device_mem))
    # Append to the appropriate list.
    if engine.binding_is_input(binding):
      inputs.append(HostDeviceMem(host_mem, device_mem))
    else:
      outputs.append(HostDeviceMem(host_mem, device_mem))
  return inputs, outputs, bindings, stream


def load_engine(trt_runtime, engine_path):
  with open(engine_path, 'rb') as f:
    engine_data = f.read()
  engine = trt_runtime.deserialize_cuda_engine(engine_data)
  return engine


def _load_img(image_path):
  from PIL import Image
  import torchvision.transforms

  im = Image.open(image_path)
  im = im.resize((640, 640))
  # im.show()
  trans_to_tensor = torchvision.transforms.ToTensor()
  _tensor = trans_to_tensor(im)
  return _tensor


def _trt_infer():
  import tensorrt as trt
  import pycuda.driver as cuda
  import pycuda.autoinit
  import numpy as np
  import time
  import os
  import cv2
  import torch
  import argparse

  _trt_engine_path = '/home/noname/projects/deeplearning/yolov5-luggage-defect-detection/model_binaries/yolov5x.engine'
  if not os.path.exists(_trt_engine_path):
    print(f'[trace] target TensorRT engine file does NOT exist: {_trt_engine_path}')
    exit(-1)

  TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
  trt.init_libnvinfer_plugins(TRT_LOGGER, '')
  trt_runtime = trt.Runtime(TRT_LOGGER)
  batch_size = 1
  trt_engine = load_engine(trt_runtime, _trt_engine_path)
  print("[trace] TensorRT engine loaded")
  print("[trace] allocating buffers for TensorRT engine")
  inputs, outputs, bindings, stream = allocate_buffers_for_encoder(trt_engine)
  print("[trace] allocating buffers done")

  print("[trace] TensorRT engine: creating execution context")
  context = trt_engine.create_execution_context()

  img = _load_img(image_file_path)
  np.copyto(inputs[0].host, img.ravel())
  [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
  # Run inference.
  context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
  stream.synchronize()
  [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
  stream.synchronize()
  print(f'[trace] done with trt inference')
  ret = outputs[0]
  # pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

  output = torch.from_numpy(ret.host)
  output = torch.reshape(output, (1, 25200, 85))
  pred = non_max_suppression(output, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
  render_result(image_file_path, pred)

  pass


def run(
    weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
    source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
    data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
    imgsz=(640, 640),  # inference size (height, width)
    conf_thres=0.25,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    max_det=1000,  # maximum detections per image
    device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img=False,  # show results
    save_txt=False,  # save results to *.txt
    save_conf=False,  # save confidences in --save-txt labels
    save_crop=False,  # save cropped prediction boxes
    nosave=False,  # do not save images/videos
    classes=None,  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,  # class-agnostic NMS
    augment=False,  # augmented inference
    visualize=False,  # visualize features
    update=False,  # update all models
    project=ROOT / 'runs/detect',  # save results to project/name
    name='exp',  # save results to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    line_thickness=3,  # bounding box thickness (pixels)
    hide_labels=False,  # hide labels
    hide_conf=False,  # hide confidences
    half=False,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
):
  _onnx_infer()
  #_trt_infer()

  pass


def parse_opt():
  parser = argparse.ArgumentParser()
  parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
  parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
  parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
  parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
  parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
  parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
  parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
  parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
  parser.add_argument('--view-img', action='store_true', help='show results')
  parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
  parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
  parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
  parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
  parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
  parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
  parser.add_argument('--augment', action='store_true', help='augmented inference')
  parser.add_argument('--visualize', action='store_true', help='visualize features')
  parser.add_argument('--update', action='store_true', help='update all models')
  parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
  parser.add_argument('--name', default='exp', help='save results to project/name')
  parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
  parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
  parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
  parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
  parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
  parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
  opt = parser.parse_args()
  opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
  print_args(vars(opt))
  return opt


def main(opt):
  run(**vars(opt))


if __name__ == "__main__":
  opt = parse_opt()
  main(opt)
