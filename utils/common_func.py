from utils.general import non_max_suppression
import cv2
import random
import torch
from utils.general import print_args, non_max_suppression, scale_coords, check_img_size

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


def render_result(image_file_path, pred, title='Result Image'):
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
        else:  # this is a damage
          color = [0, 0, 255]

        plot_one_box([x1, y1, x2, y2], image, color=color, label=None, line_thickness=2)

        # cv2.imwrite(save_file_path, image)
  cv2.imshow(title, image)
  cv2.waitKey(0)
  pass



def _load_img_to_tensor(im, dims=(640, 640)):
  from PIL import Image
  import torchvision.transforms
  im = im.resize(dims)
  # im.show()
  trans_to_tensor = torchvision.transforms.ToTensor()
  _tensor = trans_to_tensor(im)
  return _tensor

def load_engine(trt_runtime, engine_path):
  with open(engine_path, 'rb') as f:
    engine_data = f.read()
  engine = trt_runtime.deserialize_cuda_engine(engine_data)
  return engine

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


def _trt_infer(img, _trt_engine_path):
  import tensorrt as trt
  import pycuda.driver as cuda
  import pycuda.autoinit
  import numpy as np
  import time
  import os
  import cv2
  import torch
  import argparse

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

  #img = _load_img(image_file_path)
  print("[trace] start to copy data")
  img = _load_img_to_tensor(img)
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
  output = torch.reshape(output, (1, 25200, 7))

  conf_thres = 0.10  # confidence threshold
  iou_thres = 0.25  # NMS IOU threshold
  classes = None  # filter by class: --class 0, or --class 0 2 3
  agnostic_nms = False  # class-agnostic NMS
  max_det = 1000  # maximum detections per image
  from utils.torch_utils import select_device
  device = select_device('')

  pred = non_max_suppression(output, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
  return pred
