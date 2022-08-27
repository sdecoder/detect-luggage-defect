python train.py --data luggage.yaml --cfg yolov5x.yaml --weights '' --batch-size 4

Convert trained model to onnx files:
https://github.com/ultralytics/yolov5/issues/251
python export.py --weights yolov5s.pt --include torchscript onnx


* Tasks done so far

1. able to use the ONNX and TensorRT engine to infer the bounding box
2. annotate the dataset

* Tasks to do

1. deployment using Triton Server and Python flask
2. transfer learning using yolov5x 
