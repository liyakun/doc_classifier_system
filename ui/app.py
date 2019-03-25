import requests
from ui.lib.forms import ImageForm
from flask import Flask, render_template, request
from flask_cors import CORS

app = Flask(__name__)
app.secret_key = 'lkasjdfi3jeofiwjfowflmksfdsoifjsi#@$%@%#$%'
CORS(app)
CLASSIFIER_API = 'http://0.0.0.0:9998/'


@app.route('/', methods=['GET', 'POST'])
def home():
    form = ImageForm()
    if request.method == 'POST':
        image_file = form.image.data
        classifications = requests.post(url=CLASSIFIER_API, files={'media': image_file})
        return render_template('show.html', classifications=classifications.content)
    else:
        return render_template('home.html')


if __name__ == "__main__":
    app.debug = True
    app.run(host="0.0.0.0", port=9999)



