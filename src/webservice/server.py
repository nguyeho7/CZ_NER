from flask import Flask
from flask import request, render_template, jsonify
from src.CRF_NER.CRF_NER import parse_commands
import pycrfsuite
from src.common.feature_extractor import *
model = "conll2003nopos"

def init(filename = "model.txt"):
    with open(filename) as f:
        line = f.read()
        tokens = line.strip().split(' ')
        label = tokens[0]
        params = tokens[1:]
        return label, params

_, params = init()
ft = feature_extractor(["get_type", "is_capitalised", "contains_at", "contains_digit",
        "suffix_2", "suffix_3", "prefix_2", "prefix_3", "get_type", "per_gzttr", "eng_PER", "loc_gzttr", "eng_LOC",
        "org_gzttr", "eng_ORG", "misc_gzttr", "eng_MISC"])
app = Flask(__name__)
tagger = pycrfsuite.Tagger()
tagger.open(model+".crfmodel")
def wrap_text(tag, token):
    if tag == "O":
        return token
    return "<{} {}>".format(tag, token)

@app.route("/")
def my_form():
    return render_template("my-form.html")

@app.route("/annotate", methods=['POST', 'GET'])
def my_for_post():
    external_data = defaultdict(set)
    external_entities = request.args.get('entities', "", type=str)
    print(external_entities)
    if external_entities != "":
        for line in external_entities.split('\n'):
            tag, token = line.split(' ')
            external_data[tag].add(token)
    print(external_data)
    text = request.args.get('sentence', "", type=str)
    sentence = []
    tokens = text.split(" ")
    for tok in tokens:
        sentence.append({"label": tok, "pos":"none", "dep": "none"})
    features = ft.extract_features_sentence_conll(sentence, external_data, string_format=False)
    print(features)
    predictions = tagger.tag(features)
    output = " ".join(wrap_text(tag, token) for tag, token in zip(predictions, tokens))
    return jsonify(result=output)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)
 
