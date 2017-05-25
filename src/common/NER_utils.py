#!/usr/bin/env python3
from src.common.feature_extractor import *
import string
import json

"""
Set of utils to transform the CNEC2.0 dataset into a format usable by pythoncrfsuite or keras
"""

def load_dataset(filename="named_ent_dtest.txt"):
    '''
    Returns the dataset where each line is a item in a list
    '''
    with open(filename) as f:
        return f.read().split('\n')

def transform_dataset_conll(dataset_name):
    dataset = open(dataset_name)
    sentences_text = []
    features = []
    tags = []
    current_sentence = []
    current_sentence_tags = []
    current_sentence_features = []
    ft = feature_extractor(["get_type", "is_capitalised", "contains_at", "contains_digit",
        "suffix_2", "suffix_3", "prefix_2", "prefix_3", "get_type", "per_gzttr", "eng_PER", "loc_gzttr", "eng_LOC",
        "org_gzttr", "eng_ORG", "misc_gzttr", "eng_MISC")
    for line in dataset:
        line = line.strip()
        if line.startswith("-DOCSTART-"):
            continue
        if len(line) == 0:
            if len(current_sentence) > 0:
                sentences_text.append(current_sentence)
                current_sentence = []
                extracted_fts = ft.extract_features_sentence_conll(current_sentence_features,
                        string_format=False)
                current_sentence_features = []
                features.append(extracted_fts)
                tags.append(current_sentence_tags)
                current_sentence_tags = []
            continue
        data = line.split(' ')
        token = data[0]
        pos = data[1]
        dep = data[2]
        tag = data[3]
        current_sentence.append(token)
        current_sentence_features.append({"label":token, "pos":pos, "dep":dep})
        current_sentence_tags.append(tag)
    return tags, features, sentences_text
    

def read_dataset_conll(dataset_name):
    dataset = open(dataset_name)
    sentences_text = []
    features = []
    current_sentence = []
    current_sentence_features = []
    ft = feature_extractor(["get_type", "is_capitalised", "contains_at", "contains_digit",
        "suffix_2", "suffix_3", "prefix_2", "prefix_3"])
    for line in dataset:
        line = line.strip()
        if line.startswith("-DOCSTART-"):
            continue
        if len(line) == 0:
            if len(current_sentence) > 0:
                sentences_text.append(current_sentence)
                current_sentence = []
                extracted_fts = ft.extract_features_sentence_conll(current_sentence_features,
                        string_format=False)
                current_sentence_features = []
                features.append(extracted_fts)
            continue
        data = line.split(' ')
        token = data[0]
        pos = data[1]
        dep = data[2]
        current_sentence.append(token)
        current_sentence_features.append({"label":token, "pos":pos, "dep":dep})
    return  features, sentences_text

def save_indices(indices, filename):
    """
    Save a dictionary of word/tag to num index
    """
    with open(filename, 'w') as f:
        f.write(json.dumps(indices, ensure_ascii=False))

def create_indices(dataset_filename, tag_filename, token_filename):
    dataset = load_dataset(dataset_filename)
    indices = {}
    tag_indices = {}
    i = 1 
    tag_i = 0 
    max_l = 0
    for line in dataset:
        labels, tags = get_labels_tags(line_split(line), merge="supertype")
        if len(labels) > max_l:
            max_l = len(labels)
        for tag in tags:
            if tag in tag_indices:
                continue
            else:
                tag_indices.update({tag: tag_i})
                tag_i += 1
        for token in labels:
            if token in indices:
                continue
            else:
                indices.update({token: i})
                i += 1
    save_indices(tag_indices, tag_filename)
    save_indices(indices, token_filename)
    print('{} total words'.format(i))
    print('{} max sentence length'.format(max_l))

def create_indices_conll(dataset_filename, tag_filename, token_filename, pos_filename):
    tags, features, sentences_text = transform_dataset_conll(dataset_filename)
    indices = {}
    tag_indices = {}
    pos_indices = {}
    pos_i = 1
    i = 1
    tag_i = 0
    max_l = 0
    for tag_seq, sentence in zip(tags, features):
        max_l = max(max_l, len(sentence))
        for ft in sentence:
            word = ft['w[0]']
            pos = ft['pos[0]']
            if not pos in pos_indices:
                pos_indices.update({pos : pos_i})
                pos_i += 1
            if not (word in indices):
                indices.update({word: i})
                i += 1
        for tag in tag_seq:
            if tag in tag_indices:
                continue
            else:
                tag_indices.update({tag: tag_i})
                tag_i += 1
    save_indices(tag_indices, tag_filename)
    save_indices(indices, token_filename)
    save_indices(pos_indices, pos_filename)
    print('{} total words'.format(i))
    print('{} max sentence length'.format(max_l))

