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

* Performance data:
trtexec --loadEngine=best.engine --batch=128 --streams=32 --verbose --avgRuns=20
```shell
[08/29/2022-17:09:04] [I] Throughput: 4719.69 qps
[08/29/2022-17:09:04] [I] Latency: min = 114.873 ms, max = 1295.56 ms, mean = 847.8 ms, median = 934.057 ms, percentile(99%) = 1189.62 ms
[08/29/2022-17:09:04] [I] Enqueue Time: min = 0.366281 ms, max = 452.828 ms, mean = 99.6711 ms, median = 38.6289 ms, percentile(99%) = 438.662 ms
[08/29/2022-17:09:04] [I] H2D Latency: min = 1.66406 ms, max = 19.3125 ms, mean = 4.17684 ms, median = 1.74438 ms, percentile(99%) = 19.1172 ms
[08/29/2022-17:09:04] [I] GPU Compute Time: min = 112.816 ms, max = 1287.63 ms, mean = 843.108 ms, median = 928.094 ms, percentile(99%) = 1181.64 ms
[08/29/2022-17:09:04] [I] D2H Latency: min = 0.226562 ms, max = 1.34229 ms, mean = 0.515291 ms, median = 0.245117 ms, percentile(99%) = 1.06641 ms
[08/29/2022-17:09:04] [I] Total Host Walltime: 68.669 s

```


* Tasks done so far

1. able to use the ONNX and TensorRT engine to infer the bounding box
2. annotate the dataset
3. transfer learning using yolov5x
4. deployment using Triton Server and Python flask

* Tasks to do

1. using SOTA models like: swin transform
2. train with more data
