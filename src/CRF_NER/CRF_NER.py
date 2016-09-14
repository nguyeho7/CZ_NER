#!/usr/bin/env python3
from .NER_utils import transform_dataset 
from .NER_utils import dump_POS_tags
from .NER_utils import load_dataset
import pycrfsuite
import sys 
from collections import Counter
import random

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
    tp = Counter()
    fp = Counter()
    fn = Counter()
    tag_count_pr = Counter()
    tag_count_tr = Counter()
    for yp, yt in zip(merged_ypred, merged_ytrue):
        tag_count_tr[yt] += 1
        tag_count_pr[yp] += 1
        if yp == yt:
            tp[yt] += 1
        else:
            fn[yt] += 1
            fp[yp] += 1
    total_tp = 0
    total_fn = 0
    total_fp = 0
    for tag in tags:
        total_tp += tp[tag]
        total_fn += fn[tag]
        total_fp += fp[tag]
    #micro measure
    precision = total_tp/(total_tp + total_fp)
    recall = total_tp/(total_tp + total_fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1, tp, fn, fp, tag_count_pr, tag_count_tr, tags

def output_evaluation(precision, recall, f1, tp,fn,fp, count_pr, count_tr, tags,model_name):
    with open(model_name + '.log', 'w') as f:
        f.write('precision(micro): {}\n'.format(precision))
        f.write('recall(micro): {}\n'.format(recall))
        f.write('F1(micro): {}\n'.format(f1))
        f.write('======per-tag-stats=====\n')
        for tag in tags:
            if (tp[tag] + fp[tag]>0):
                precision = tp[tag] / (tp[tag] + fp[tag])
            else:
                precision = 0.
            if (tp[tag] + fn[tag]>0):
                recall = tp[tag] / (tp[tag] + fn[tag])
            else:
                recall = 0.
            if (precision + recall) > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0.
            f.write("{}:\t\t".format(tag))
            f.write("precision: {:.3}\t".format(precision))
            f.write("recall: {:.3}\t".format(recall))
            f.write("f1: {:.3}\t".format(f1))
            f.write("predicted: {}\t".format(count_pr[tag]))
            f.write("dataset: {}\t".format(count_tr[tag]))
            f.write("\n")
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
    global_raw = load_dataset("named_ent.txt")
    merge = "BIO"
    dump_POS_tags(global_raw, "POS.json")
    for model, params in models:
        trainer = pycrfsuite.Trainer(verbose=True)
        tr_label, tr_feature = transform_dataset(tr_raw, params, merge)
        te_label, te_feature = transform_dataset(te_raw, params, merge) 
        for lab, feat in zip(tr_label, tr_feature):
            trainer.append(feat, lab)
        trainer.train(model+'.crfmodel')
        tagger = pycrfsuite.Tagger()
        tagger.open(model+'.crfmodel')
        text = [[w[0][5:] for w in sentence] for sentence in te_feature]
        predictions = [tagger.tag(sentence) for sentence in te_feature]
        evaluations = global_eval(predictions, te_label)
        output_evaluation(*evaluations, model)
        for i in range(40):
            num = random.randint(0, len(text))
            curr_sent = "\t".join(text[num])
            curr_pred = "\t ".join(predictions[num])
            curr_gold = "\t ".join(te_label[num])
            print("sent:\t",curr_sent)
            print("pred:", curr_pred)
            print("gold:", curr_gold)
if __name__ == '__main__':
    main()
