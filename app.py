import os
from flask import Flask, render_template, request, make_response
import classifier.classifier


INDEX_PATH = 'index.html'
RESPONSE_PATH = 'hotdog_response.html'

app = Flask(__name__)

detector = None
try:
    detector = classifier.classifier.Core()
    detector.load()
except NotImplementedError:
    print("core is not implemented")
    exit(-1)
except FileNotFoundError:
    detector.train()


@app.route('/', methods=['GET', 'POST'])
def handle_single_page():
    if request.method == "POST":
        if request.files:
            image = request.files["image"]
            img_name = "image.{}".format(image.filename.split(".")[-1])
            image.save(os.path.join("./static/uploads", img_name))
            print("img saved !")
            print("detecting img...")
            result = detector.classify(f"../static/uploads/{img_name}")

            return make_response(render_template(RESPONSE_PATH, img_name=img_name, hotdog_text=f"This is {result}"))

    return render_template(INDEX_PATH)


@app.route('/index.html', methods=['GET', 'POST'])
def index():
    return render_template(INDEX_PATH)


@app.route('/index', methods=['GET', 'POST'])
def index_no_extension():
    return render_template(INDEX_PATH)


if __name__ == '__main__':
    app.run()
