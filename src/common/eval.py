#!/usr/bin/env python3
import random
from collections import Counter

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
