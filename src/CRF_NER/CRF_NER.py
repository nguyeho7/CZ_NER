#!/usr/bin/env python3
from src.common.NER_utils import transform_dataset 
from src.common.NER_utils import dump_POS_tags
from src.common.NER_utils import load_dataset
from src.common.NER_utils import load_transform_dataset_json
from src.common.eval import global_eval, output_evaluation
import argparse
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

def parse_args():
    parser = argparse.ArgumentParser(description='Train or eval CRF_NER')
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-t", "--train", action="store_true")
    group.add_argument("-p", "--predict", action="store_true")
    parser.add_argument('--json', help="load eval set from json instead", action="store_true")
    parser.add_argument('train_set', help='train set filename')
    parser.add_argument('test_set', help='test set filename')
    parser.add_argument('models', help='models filename')
    parser.add_argument('merge', help='BIO|supertype|none')
    return parser.parse_args()


def predict_and_eval(models, filename, merge, json=False):
    if not json:
        te_raw = load_dataset(filename)
    for model, params in models:
        tagger = pycrfsuite.Tagger()
        tagger.open(model+".crfmodel")
        if json:
            labels, features = load_transform_dataset_json(filename, params)
        else:
            labels, features = transform_dataset(te_raw, params, merge)
        text = [[w[0][5:] for w in sentence] for sentence in features]
        predictions = [tagger.tag(sentence) for sentence in features]
        evaluations = global_eval(predictions, labels)
        output_evaluation(*evaluations, model_name=model)

def train_and_eval(models, train_set, test_set, merge):
    for model, params in models:
        trainer = pycrfsuite.Trainer(verbose=True)
        tr_label, tr_feature = transform_dataset(train_set, params, merge)
        te_label, te_feature = transform_dataset(test_set, params, merge) 
        for lab, feat in zip(tr_label, tr_feature):
            trainer.append(feat, lab)
        trainer.train(model+'.crfmodel')
        tagger = pycrfsuite.Tagger()
        tagger.open(model+'.crfmodel')
        text = [[w[0][5:] for w in sentence] for sentence in te_feature]
        predictions = [tagger.tag(sentence) for sentence in te_feature]
        evaluations = global_eval(predictions, te_label)
        output_evaluation(*evaluations, model_name=model)

def main():
    args = parse_args()
    models = parse_commands(args.models) 
    merge = args.merge
    if args.train:
        tr_raw = load_dataset(args.train_set)
        te_raw = load_dataset(args.test_set)
        train_and_eval(models, tr_raw, te_raw, merge)
    elif args.predict:
        predict_and_eval(models, args.test_set, merge, args.json)

if __name__ == '__main__':
    main()
