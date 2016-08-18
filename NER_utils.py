#!/usr/bin/env python3
from feature_extractor import *
"""
Set of utils to transform the CNEC2.0 dataset into a format usable by CRFSuite
extract_features should be extendable by further features
"""
import string
exclude = set('!"#$%&\'()*+,-./:;=?@[\\]^_`{|}~')

def load_dataset(filename="named_ent_dtest.txt"):
    '''
    Returns the dataset where each line is a item in a list
    '''
    with open(filename) as f:
        return f.read().split('\n')

def line_split(line):
    '''
    a split according to space characters that considers tags in <> brackets as one word
    also works with embedded tags i.e. <s <ps kote>> will be one word
    '''
    in_tag = False
    embedded = False
    current = -1
    for i, ch in enumerate(line):
        if ch == '<':
            if in_tag:
                embedded = True
            else:
                in_tag = True
        elif ch == '>':
            if embedded:
                embedded = False
            else:
                in_tag = False
        if ch.isspace() and not in_tag:
            yield line[current+1:i]
            current = i

def get_tags(tokens):
    return [get_tag(token) for token in tokens]

def get_tag(token):
    if is_NE(token):
        return get_NE_tag(token)
    else:
        return "O"

def get_NE_tag(token):
    start = 1
    end = 1
    while token[end] != ' ' and token[end] != '<':
        end+=1
    return token[start:end]

def transform_dataset(dataset, params):
    '''
    Transforms the cnec2.0 dataset into a format usable by pythoncrfsuite
    '''
    features_list = []
    labels = []
    ft = feature_extractor(params)
    for line in dataset:
        tokens = list(line_split(line))
        labels.append(get_tags(tokens))
        features_list.append(ft.extract_features(tokens))
    return labels, features_list

def dump_dataset(labels, features, filename):
    """
    NOTE: NOT WORKING YET. NESTED SENTENCES
    """
    with open(filename, 'w') as out:
        for label, feature in zip(labels, features):
            if len(feature) == 0:
                continue
            s += ["\t".join(feature) for feature in features]
            out.write(label[0])
            out.write('\t')
            out.write(s)
            out.write('\n')
"""
def main():
    import sys
    input_filename = sys.argv[1]
    output_filename = sys.argv[2]
    data_raw = load_dataset(input_filename)
    data_ft = transform_dataset(data_raw)
    dump_dataset(data_ft[0], data_ft[1], output_filename)

if __name__ == '__main__':
    main()
"""
