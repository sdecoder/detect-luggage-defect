


from flask import Flask
from flask import request

app = Flask(__name__)


@app.route('/infer_by_trt', methods=['GET', 'POST', 'DELETE'])
def infer_by_trt():
  if request.method == 'GET':
    return "return the information for <user_id>"

  elif request.method == 'POST':
    return "modify/update the information for <user_id>"

  elif request.method == 'DELETE':
    return "delete user with ID <user_id>"""
  return 'Hello, World!'


app.run(debug=True, port=8080)
