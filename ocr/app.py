from ocr.lib.ocr import OCR
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.route('/', methods=['POST'])
def ocr_doc():
    text = ''
    if request.method == 'POST':
        try:
            image_file = request.files['media']
            text = OCR.ocr_doc(image_file)
        except KeyError as e:
            print(e)
    return jsonify(text)


if __name__ == "__main__":
    app.debug = True
    app.run(host="0.0.0.0", port=9996)
