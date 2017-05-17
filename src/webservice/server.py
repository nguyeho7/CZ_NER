from flask import Flask
from flask import request, render_template, jsonify
from src.common.NER_utils import transform_dataset_web 
from src.CRF_NER.CRF_NER import parse_commands
import pycrfsuite

model = "1_nbr"

def init(filename = "model.txt"):
    with open(filename) as f:
        line = f.read()
        tokens = line.strip().split(' ')
        label = tokens[0]
        params = tokens[1:]
        return label, params

_, params = init()
tagger = pycrfsuite.Tagger()
tagger.open(model+".crfmodel")

app = Flask(__name__)
def wrap_text(tag, token):
    if tag == "O":
        return token
    return "<{} {}>".format(tag, token)

@app.route("/")
def my_form():
    return render_template("my-form.html")

@app.route("/annotate", methods=['POST', 'GET'])
def my_for_post():
    text = request.args.get('sentence', 0, type=str)
    features, tokens = transform_dataset_web(text, params, merge = "supertype")
    predictions = tagger.tag(features)
    output = " ".join(wrap_text(tag, token) for tag, token in zip(predictions, tokens))
    return jsonify(result=output)
if __name__ == "__main__":
    app.run()
 