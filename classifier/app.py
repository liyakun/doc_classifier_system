from classifier.lib.classifier import Classifier
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests

app = Flask(__name__)
CORS(app)
CAFFE_MODEL = 'models/caffe_alexnet_train_iter_95000.caffemodel'
DEPLOY_FILE = 'models/deploy.prototxt'
LABELS_FILE = 'models/labels.txt'
OCR_API = 'http://0.0.0.0:9996/'


@app.route('/', methods=['POST'])
def classifier():
    classifications = 'Not classified'
    text = 'No OCR'
    if request.method == 'POST':
        try:
            image_file = request.files['media']
            text = requests.post(url=OCR_API, files={'media': image_file})
            classifications = Classifier.classify(
                caffemodel=CAFFE_MODEL,
                deploy_file=DEPLOY_FILE,
                image_file=image_file,
                labels_file=LABELS_FILE,
                use_gpu=True
            )
        except KeyError as e:
            print(e)

    return jsonify('Classifications: \n' + classifications + '\n\n Content: ' + ' '.join(text.text))


if __name__ == "__main__":
    app.debug = True
    app.run(host="0.0.0.0", port=9998)



