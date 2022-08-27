* Train
python train.py --data luggage.yaml --cfg yolov5x.yaml --weights '' --batch-size 4
python train.py --data data/luggage.yaml --cfg models/luggage.yaml --weights yolov5x.pt --batch-size 4

Convert trained model to onnx files:
https://github.com/ultralytics/yolov5/issues/251
python export.py --weights yolov5s.pt --include torchscript onnx

fine-tuned weights from the pretrained weighted
300 epochs completed in 1.189 hours.
Optimizer stripped from runs/train/exp13/weights/last.pt, 173.2MB
Optimizer stripped from runs/train/exp13/weights/best.pt, 173.2MB
python export.py --weights runs/train/exp13/weights/best.pt --include torchscript onnx
python export.py --weights runs/train/exp13/weights/last.pt --include torchscript onnx

Export complete (9.7s)
Results saved to /home/noname/projects/deeplearning/yolov5-luggage-defect-detection/runs/train/exp13/weights
Detect:          python detect.py --weights runs/train/exp13/weights/best.onnx
Validate:        python val.py --weights runs/train/exp13/weights/best.onnx
PyTorch Hub:     model = torch.hub.load('ultralytics/yolov5', 'custom', 'runs/train/exp13/weights/best.onnx')
Visualize:       https://netron.app


Export complete (9.3s)
Results saved to /home/noname/projects/deeplearning/yolov5-luggage-defect-detection/runs/train/exp13/weights
Detect:          python detect.py --weights runs/train/exp13/weights/last.onnx
Validate:        python val.py --weights runs/train/exp13/weights/last.onnx
PyTorch Hub:     model = torch.hub.load('ultralytics/yolov5', 'custom', 'runs/train/exp13/weights/last.onnx')
Visualize:       https://netron.app

* available result images:
1. image_file_path = '/home/noname/projects/deeplearning/datasets/luggage/images/train/000000000013.jpg'
2. image_file_path = '/home/noname/projects/deeplearning/datasets/luggage/images/train/000000000016.jpg'
3. image_file_path = '/home/noname/projects/deeplearning/datasets/luggage/images/train/000000000024.jpg'


* Tasks done so far

1. able to use the ONNX and TensorRT engine to infer the bounding box
2. annotate the dataset
3. transfer learning using yolov5x
4. deployment using Triton Server and Python flask

* Tasks to do

1. using SOTA models like: swin transform
2. train with more data

