from flask import Flask
from flask import request
from flask import Flask, render_template, request

from utils.general import non_max_suppression
from utils.common_func import load_engine, _trt_infer

app = Flask(__name__)

# image_file_path = '/home/noname/projects/deeplearning/datasets/luggage/images/train/000000000016.jpg'
_trt_engine_path = '/home/noname/projects/deeplearning/yolov5-luggage-defect-detection/runs/train/exp13/weights/best.engine'

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
    pred = _trt_infer(img, _trt_engine_path)
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
