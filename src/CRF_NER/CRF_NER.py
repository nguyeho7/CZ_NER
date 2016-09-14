#!/usr/bin/env python3
from src.common.NER_utils import transform_dataset 
from src.common.NER_utils import dump_POS_tags
from src.common.NER_utils import load_dataset
from src.common.eval import global_eval, output_evaluation
import pycrfsuite
import sys 
from collections import Counter
import random

"""
Usage: python CRF_NER.py named_ent_dtest.txt named_ent_etest.txt model.txt 
"""
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
