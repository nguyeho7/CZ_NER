#!/usr/bin/env python3
import random
from collections import Counter
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import classification_report
import numpy as np


def global_eval(ypred, ytrue):
    """
    Measures micro averaged precision, recall, f1 and per-tag precision, recall, f1
    returns precision, recall, f1 (floats), per_tag_prec, per_tag_rec, per_tag_f1 (dictionaries)
    """
    merged_ypred = [item for sublist in ypred for item in sublist]
    merged_ytrue = [item for sublist in ytrue for item in sublist]
    tags = set(merged_ytrue)
    precision, recall, f1, support = score(merged_ytrue, merged_ypred, average='micro')
    report = classification_report(merged_ytrue, merged_ypred, digits=3)
    return precision, recall, f1, support, report, tags

def output_evaluation(precision, recall, f1, support, report, tags,model_name):
    with open(model_name + '.log', 'w') as f:
        f.write('precision(micro): {}\n'.format(precision))
        f.write('recall(micro): {}\n'.format(recall))
        f.write('F1(micro): {}\n'.format(f1))
        f.write('support: {}\n'.format(support))
        f.write('======per-tag-stats=====\n')
        f.write(report)
        f.close()

def random_sample(filename, sentences, predictions, gold_standard, num):
    with open(filename, 'w') as f:
        for x in range(num):
            num = random.randint(0,int(len(sentences)/2))
            curr_sent = "sent:" + "\t".join(sentences[num])
            curr_pred = "pred:" + "\t".join(predictions[num])
            curr_gold = "gold:" + "\t".join(gold_standard[num])
            f.write(curr_sent)
            f.write('\n')
            f.write(curr_pred)
            f.write('\n')
            f.write(curr_gold)
            f.write('\n')
