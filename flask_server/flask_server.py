from flask import Flask
from flask import request
from flask import Flask, render_template, request

from utils.general import non_max_suppression

app = Flask(__name__)

# image_file_path = '/home/noname/projects/deeplearning/datasets/luggage/images/train/000000000016.jpg'

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

def _load_img(im):
  from PIL import Image
  import torchvision.transforms
  im = im.resize((640, 640))
  # im.show()
  trans_to_tensor = torchvision.transforms.ToTensor()
  _tensor = trans_to_tensor(im)
  return _tensor

def _trt_infer(img):
  import tensorrt as trt
  import pycuda.driver as cuda
  import pycuda.autoinit
  import numpy as np
  import time
  import os
  import cv2
  import torch
  import argparse

  _trt_engine_path = '/home/noname/projects/deeplearning/yolov5-luggage-defect-detection/runs/train/exp13/weights/best.engine'
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
  img = _load_img(img)
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

@app.route('/infer_by_trt', methods=['GET', 'POST', 'DELETE'])
def infer_by_trt():
  # for html test
  return render_template('upload.html')

@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
  if request.method == 'POST':
    f = request.files['file']
    # f.save(secure_filename(f.filename))

    from PIL import Image
    # file_bytes is bytes[36348]
    img = Image.open(f)  # load with Pillow
    pred = _trt_infer(img)
    import pickle
    pred_string = pickle.dumps(pred)
    print(f'[trace] picke dumps string: {pred_string}')
    return pred_string


'''
  if request.method == 'GET':
    return "return the information for <user_id>"

  elif request.method == 'POST':
    return "modify/update the information for <user_id>"

  elif request.method == 'DELETE':
    return "delete user with ID <user_id>"""
  return 'Hello, World!'

'''

app.run(debug=True, port=8080)
