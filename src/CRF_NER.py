#!/usr/bin/env python3
from NER_utils import transform_dataset 
from NER_utils import load_dataset
import pycrfsuite
import sys 
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import f1_score, classification_report

"""
Usage: python CRF_NER.py named_ent_dtest.txt named_ent_etest.txt model.txt 
"""
def global_eval(ypred, ytrue):
    """
    measures macro f1 score for the classification
    """
    merged_ypred = [item for sublist in ypred for item in sublist]
    merged_ytrue = [item for sublist in ytrue for item in sublist]
    binarizer = LabelBinarizer() # makes one vs all matrices for labels
    y_true_bin = binarizer.fit_transform(merged_ytrue)
    y_pred_bin = binarizer.transform(merged_ypred)
    tags = [tag for tag in binarizer.classes_]
    cl_indices = {cls: idx for idx, cls in enumerate(binarizer.classes_)}
    return f1_score(y_true_bin, y_pred_bin, labels=[cl_indices[tag] for tag in tags],average='macro')

def per_token_eval(ypred, ytrue):
    """
    measures per-token precision, ignoring the not_named_entity tag
    """
    merged_ypred = [item for sublist in ypred for item in sublist]
    merged_ytrue = [item for sublist in ytrue for item in sublist]
    binarizer = LabelBinarizer() 
    y_true_bin = binarizer.fit_transform(merged_ytrue)
    y_pred_bin = binarizer.transform(merged_ypred)
    tags = [tag for tag in binarizer.classes_ if tag != 'O']
    cl_indices = {cls: idx for idx, cls in enumerate(binarizer.classes_)}
    return classification_report(y_true_bin,
                                 y_pred_bin,
                                 labels=[cl_indices[tag] for tag in tags],
                                 target_names=tags)

def output_evalutation(classification_report, f1_score, model_name):
    with open(model_name + '.log', 'w') as f:
        f.write('F1 macro score: {}'.format(f1_score))
        f.write('\n')
        f.write(classification_report)
        f.close()

def parse_commands(filename):
    """
    Reads the configuration file with the model name and features, separated by spaces
    model_name feat1 feat2 feat3 ....
    """
    models = []
    with open(filename) as f:
        for line in f.read().split('\n'):
            tokens = line.split(' ')
            if len(tokens[1:]) == 0:
                continue
            models.append((tokens[0], tokens[1:]))
    return models


def main():
    train_filename = sys.argv[1]
    test_filename = sys.argv[2]
    models = parse_commands(sys.argv[3]) 
    tr_raw = load_dataset(train_filename)
    te_raw = load_dataset(test_filename)
    for model, params in models:
        trainer = pycrfsuite.Trainer(verbose=False)
        tr_label, tr_feature = transform_dataset(tr_raw, params)
        te_label, te_feature = transform_dataset(te_raw, params) 
        for lab, feat in zip(tr_label, tr_feature):
            trainer.append(feat, lab)
        trainer.train(model+'.crfmodel')
        tagger = pycrfsuite.Tagger()
        tagger.open(model+'.crfmodel')
        predictions = [tagger.tag(sentence) for sentence in te_feature]
        per_class_prec = per_token_eval(predictions, te_label)
        f1_score = global_eval(predictions, te_label)
        output_evalutation(per_class_prec, f1_score, model)


if __name__ == '__main__':
    main()
