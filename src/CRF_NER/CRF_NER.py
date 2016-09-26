#!/usr/bin/env python3
from src.common.NER_utils import transform_dataset 
from src.common.NER_utils import dump_POS_tags
from src.common.NER_utils import load_transform_dataset
from src.common.eval import global_eval, output_evaluation, random_sample
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
    parser.add_argument('train_set', help='train set filename')
    parser.add_argument('test_set', help='test set filename')
    parser.add_argument('models', help='models filename')
    parser.add_argument('merge', help='BIO|supertype|none')
    return parser.parse_args()

def predict_and_eval(models, filename, merge):
    for model, params in models:
        tagger = pycrfsuite.Tagger()
        tagger.open(model+".crfmodel")
        labels, features, text = load_transform_dataset(filename, params, merge)
        predictions = [tagger.tag(sentence) for sentence in features]
        evaluations = global_eval(predictions, labels)
        output_evaluation(*evaluations, model_name=model)
        random_sample("sentences_50_predict_cnec", text, predictions, labels, 50)

def train_and_eval(models, train_set, test_set, merge):
    for model, params in models:
        trainer = pycrfsuite.Trainer(verbose=True)
        tr_label, tr_feature, _ = load_transform_dataset(train_set, params, merge)
        te_label, te_feature, text = load_transform_dataset(test_set, params, merge) 
    #    json_label, json_feature = load_transform_dataset('named_ent_train.txt', params, merge)
        for lab, feat in zip(tr_label, tr_feature):
            trainer.append(feat, lab)
     #   for lab, feat in zip(json_label, json_feature):
      #      trainer.append(feat, lab)
        trainer.train(model+'.crfmodel')
        tagger = pycrfsuite.Tagger()
        tagger.open(model+'.crfmodel')
        predictions = [tagger.tag(sentence) for sentence in te_feature]
        evaluations = global_eval(predictions, te_label)
        output_evaluation(*evaluations, model_name=model)
        random_sample("sentences_50_train_eval", text, predictions, te_labes, 50)

def main():
    args = parse_args()
    models = parse_commands(args.models) 
    merge = args.merge
    if args.train:
        train_and_eval(models, args.train_set, args.test_set, merge)
    elif args.predict:
        predict_and_eval(models, args.test_set, merge)

if __name__ == '__main__':
    main()
