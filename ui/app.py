from flask import Flask, render_template, request
from flask_cors import CORS
import simplejson as json
from sys import path
path.append('../')
from lib.forms import ImageForm
from lib.send_image import UIImageSender

app = Flask(__name__)
app.secret_key = 'lkasjdfi3jeofiwjfowflmksfdsoifjsi#@$%@%#$%'
CORS(app)
CLASSIFIER_API = 'http://0.0.0.0:9998/'
RABBIT_HOST = 'localhost'


@app.route('/', methods=['GET', 'POST'])
def home():
    form = ImageForm()
    if request.method == 'POST':
        image_file = form.image.data
        result = UIImageSender(host=RABBIT_HOST, queue_name='img_classifier').call(image_f=image_file)
        return render_template('show.html', classifications=json.loads(result))
    else:
        return render_template('home.html')


if __name__ == "__main__":
    app.debug = True
    app.run(host="0.0.0.0", port=9999)



