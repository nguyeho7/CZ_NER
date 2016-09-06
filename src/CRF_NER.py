#!/usr/bin/env python3
from NER_utils import transform_dataset 
from NER_utils import dump_POS_tags
from NER_utils import load_dataset
import pycrfsuite
import sys 
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import f1_score, classification_report
from collections import Counter

"""
Usage: python CRF_NER.py named_ent_dtest.txt named_ent_etest.txt model.txt 
"""
def global_eval(ypred, ytrue):
    """
    Measures micro averaged precision, recall, f1 and per-tag precision, recall, f1
    returns precision, recall, f1 (floats), per_tag_prec, per_tag_rec, per_tag_f1 (dictionaries)
    """
    merged_ypred = [item for sublist in ypred for item in sublist]
    merged_ytrue = [item for sublist in ytrue for item in sublist]
    tags = set(merged_ytrue)
    true_positives = Counter()
    false_positives = Counter()
    false_negatives = Counter()
    for yp, yt in zip(merged_ypred, merged_ytrue):
        if yp == yt:
            true_positives[yt] += 1
        else:
            false_negatives[yt] += 1
            false_positives[yp] += 1
    total_tp = 0
    total_fn = 0
    total_fp = 0
    for tag in tags:
        total_tp += true_positives[tag]
        total_fn += false_negatives[tag]
        total_fp += false_positives[tag]
    #micro measure
    precision = total_tp/(total_tp + total_fp)
    recall = total_tp/(total_tp + total_fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    # per-tag measure
    per_tag_pr = {}
    per_tag_rec = {}
    per_tag_f1 = {}
    for tag in tags:
        if true_positives[tag] + false_positives[tag] > 0:
            per_tag_pr.update({tag: true_positives[tag]/(true_positives[tag] +
                false_positives[tag])})
        else:
            per_tag_pr.update({tag: 0})
        if (true_positives[tag] + false_negatives[tag]) > 0:
            per_tag_rec.update({tag: true_positives[tag]/(true_positives[tag] +
                false_negatives[tag])})
        else:
            per_tag_rec.update({tag:0})
        if(per_tag_pr[tag] + per_tag_rec[tag] > 0):
            per_tag_f1.update({tag: 2 * (per_tag_pr[tag] * per_tag_rec[tag]) /
            (per_tag_pr[tag]+per_tag_rec[tag])})
        else:
            per_tag_f1.update({tag: 0})
    return precision, recall, f1, per_tag_pr, per_tag_rec, per_tag_f1

def output_evaluation(precision, recall, f1, per_tag_pr, per_tag_rec, per_tag_f1, model_name):
    with open(model_name + '.log', 'w') as f:
        f.write('precision(micro): {}\n'.format(precision))
        f.write('recall(micro): {}\n'.format(recall))
        f.write('F1(micro): {}\n'.format(f1))
        f.write('======per-tag-stats=====\n')
        for pr,rec,f1 in zip(per_tag_pr.items(), per_tag_rec.items(), per_tag_f1.items()):
                f.write('tag:{} \t\t precision: {} \t\t recall: {} \t\t f1: {}\n'.format(pr[0],pr[1],rec[1],f1[1]))
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
        precision, recall, f1, per_tag_pr, per_tag_rec, per_tag_f1 = global_eval(predictions,
                te_label) 
        output_evaluation(precision, recall, f1, per_tag_pr, per_tag_rec, per_tag_f1, model)

if __name__ == '__main__':
    main()
