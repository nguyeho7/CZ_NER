#!/usr/bin/env python3
from src.common.NER_utils import transform_dataset_conll
from src.common.eval import global_eval, output_evaluation, random_sample
import argparse
import pycrfsuite
import sys 
from collections import Counter
import random
import json

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
    group.add_argument("-c", "--conll", action="store_true")
    parser.add_argument('train_set', help='train set filename')
    parser.add_argument('test_set', help='test set filename')
    parser.add_argument('models', help='models filename')
    parser.add_argument('merge', help='BIO|supertype|none')
    return parser.parse_args()

def get_filenames(train_set):
    return train_set.split(' ')


def train_and_eval_conll(train_set, test_set, filename="conllout.txt"):
    model="conll2003basic"
    trainer = pycrfsuite.Trainer(verbose=True)
    te_label, te_feature, text = transform_dataset_conll(test_set) 
    tr_label, tr_feature, _ = transform_dataset_conll(train_set)
    for lab, feat in zip(tr_label, tr_feature):
        trainer.append(feat, lab)
    #trainer.train(model+'.crfmodel')
    tagger = pycrfsuite.Tagger()
    tagger.open(model+'.crfmodel')
    predictions = [tagger.tag(sentence) for sentence in te_feature]
    result_text = ""
    for sent, pred, label in zip(te_feature, predictions, te_label):
        for ft, tag, gold in zip(sent, pred, label):
            word = ft['w[0]']
            pos = ft['pos[0]']
            result_text += word +" "+pos+" "+gold+" "+tag+"\n"
        result_text += "\n"
    with open(filename, "w") as f:
        f.write(result_text)

def main():
    args = parse_args()
    models = parse_commands(args.models) 
    merge = args.merge
    if args.train:
        train_and_eval(models, args.train_set, args.test_set, merge)
    elif args.predict:
        predict_and_eval(models, args.test_set, merge)
    elif args.conll:
        train_and_eval_conll(args.train_set, args.test_set)

if __name__ == '__main__':
    main()
