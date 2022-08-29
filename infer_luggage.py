import argparse
import os
import platform
import sys
from pathlib import Path

import PIL
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
from utils.common_func import load_engine, allocate_buffers_for_encoder, _load_img_to_tensor, render_result

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
  sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# common definition
image_file_path = '/home/noname/projects/deeplearning/datasets/luggage/images/train/000000000016.jpg'
bs = 1  # batch_size
conf_thres = 0.10  # confidence threshold
iou_thres = 0.25  # NMS IOU threshold
classes = None  # filter by class: --class 0, or --class 0 2 3
agnostic_nms = False  # class-agnostic NMS
max_det = 1000  # maximum detections per image
device = select_device('')


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


class InferenceEngine:
  def __int__(self):
    pass

  def infer(self):
    return None


class OnnxInferenceEngine(InferenceEngine):
  def infer(self):
    print(f'[trace] infer using onnx engine')
    self._onnx_infer()
    pass

  def _onnx_infer(self):
    image_width = 640
    image_height = 640
    # onnx_file_path = '/home/noname/projects/deeplearning/yolov5-luggage-defect-detection/runs/train/exp11/weights/best.onnx'
    # onnx_file_path = '/home/noname/projects/deeplearning/yolov5-luggage-defect-detection/model_binaries/yolov5x.onnx'
    onnx_file_path = '/home/noname/projects/deeplearning/yolov5-luggage-defect-detection/runs/train/exp13/weights/best.onnx'
    # onnx_file_path = '/home/noname/projects/deeplearning/yolov5-luggage-defect-detection/runs/train/exp13/weights/last.onnx'
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


class TensorRTInferenceEngine(InferenceEngine):
  def infer(self):
    print(f'[trace] infer using tensorrt engine')
    self._infer_with_tensorrt()
    pass

  def _infer_with_tensorrt(self):
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    import numpy as np
    import time
    import os
    import cv2
    import torch
    import argparse

    FILE = Path(__file__).resolve()
    ROOT = FILE.parents[0]  # YOLOv5 root directory
    # '/home/noname/projects/deeplearning/yolov5-luggage-defect-detection/'
    _trt_engine_path = f'{ROOT}/runs/train/exp13/weights/best.engine'
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

    img = PIL.Image.open(image_file_path)
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
    pred = non_max_suppression(output, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    render_result(image_file_path, pred)

    pass


class TritonInferenceEngine(InferenceEngine):

  def infer(self):
    print(f'[trace] infer using triton engine')
    self._infer_with_triton()
    pass

  def _infer_with_triton(self):
    import tritonclient.http as tritonhttpclient
    import torch
    from utils.general import scale_coords, non_max_suppression
    from utils.torch_utils import select_device

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
    image = PIL.Image.open(image_file_path)
    image_tensor = _load_img_to_tensor(image)
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
    conf_thres = 0.10  # confidence threshold
    iou_thres = 0.25  # NMS IOU threshold
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
    pass


class FlaskInferenceEngine(InferenceEngine):

  def infer(self):
    print(f'[trace] infer using flask engine')
    self._infer_with_flask()
    pass

  def _infer_with_flask(self):
    # curl -X POST  http://127.0.0.1:8080/infer_by_trt
    import pickle
    import requests

    image_file_path = '/home/noname/projects/deeplearning/datasets/luggage/images/train/000000000016.jpg'
    file_handle = open(image_file_path, 'rb')
    files = {'file': file_handle}
    values = {'DB': 'photcat', 'OUT': 'csv', 'SHORT': 'short'}
    url = 'http://localhost:8080/uploader'
    # curl -X POST  http://127.0.0.1:8080/infer_by_trt

    post_result = requests.post(url, files=files, data=values)
    bytes = post_result.content
    pred = pickle.loads(bytes)
    render_result(image_file_path, pred, title='Flask inference result')
    pass


engine_mapping = {
  "onnx": OnnxInferenceEngine(),
  "triton": TritonInferenceEngine(),
  "tensorrt": TensorRTInferenceEngine(),
  "flask": FlaskInferenceEngine(),
}


class InferenceEngineFactory:

  def produce_engine(engine_name):
    if engine_name not in engine_mapping:
      raise ValueError(engine_name)
    return engine_mapping[engine_name]


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
    infra=None,
):
  if not infra:
    return None

  print(f'[trace] current infra={infra}')
  inferenceEngine = InferenceEngineFactory.produce_engine(infra)
  inferenceEngine.infer()
  # _onnx_infer()
  # _infer_with_tensorrt()
  # _triton_infer()
  # _infer_with_flask()
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
  parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
  parser.add_argument('--name', default='exp', help='save results to project/name')
  parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
  parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
  parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
  parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
  parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
  parser.add_argument('--infra', type=str, help='use onnx/triton/tensorrt/flask')
  opt = parser.parse_args()
  opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
  print_args(vars(opt))
  return opt


def main(opt):
  run(**vars(opt))


if __name__ == "__main__":
  opt = parse_opt()
  main(opt)
