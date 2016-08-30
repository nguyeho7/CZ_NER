#!/usr/bin/env python3
from feature_extractor import *
import string
import json

"""
Set of utils to transform the CNEC2.0 dataset into a format usable by pythoncrfsuite
"""

def load_dataset(filename="named_ent_dtest.txt"):
    '''
    Returns the dataset where each line is a item in a list
    '''
    with open(filename) as f:
        return f.read().split('\n')

def line_split_old(line):
    '''
    a split according to space characters that considers tags in <> brackets as one word
    also works with embedded tags i.e. <s <ps kote>> will be one word
    '''
    in_tag = True
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
        if i==len(line)-1:
            yield line[current+1:i+1]
            current = i


def line_split(line):
    '''
    a split according to space characters that considers tags in <> brackets as one word
    also works with embedded tags i.e. <s <ps kote>> will be one word
    '''
    in_tag = 0
    current = -1
    for i, ch in enumerate(line):
        if ch == '<':
            in_tag += 1
        elif ch == '>':
            in_tag -= 1
        if ch.isspace() and in_tag==0:
            yield line[current+1:i]
            current = i
        if i==len(line)-1:
            yield line[current+1:i+1]
            current = i

def expand_NE_tokens(tokens):
    '''
    Goes through all tokens from line_split and either adds them outright
    or expands them, removing the inner tags and instead appending
    outertag_b for first, outertag_i for inner and outertag_b for last word 
    <P<pf Václavu> <ps Klausovi>> turns into:
    ['<P_b Václavu>', '<P_e Klausovi>']
    here we use the tag P instead of pf/ps
    will be hard to compare performance like this
    returns list of tokens with NE tokens expanded
    '''
    output = []
    for token in tokens:
        if not is_NE(token):
            output.append(token)
        else:
            tag = get_NE_tag(token)
            labels = []
            for label in expand_NE(get_label(token)):
                labels.extend(label.split(' '))
            for i,label in enumerate(labels):
                if i == 0:
                    output.append('<{}_b {}>'.format(tag, label))
                elif i<len(labels)-1:
                    output.append('<{}_i {}>'.format(tag, label))
                elif len(labels) != 1:
                    output.append('<{}_e {}>'.format(tag, label))
    return output


def expand_NE(token):
    '''
    returns a flattened list of NE labels, i.e.
    you have <gc Velké Británii>, you call get_label and get
    Velké Británii
    then you call expand_NE and you get:
    ['Velke', 'Britanii']
    for 'TOPIC , s . r . o . it returns:
    ['TOPIC', 's.r.o.']
    returns: list containing the parts of NE
    '''
    output = []
    token_list = token.strip().split(' ')
    for i, x in enumerate(token_list):
        if x == '':
            continue
        if x != '.':
            if i > 1:
                if token[i-1] == '.' and not x[0].isupper():
                    output[-1]+=x
                    continue       
            output.append(x)  
        elif x == '.':
            output[-1] += x
    return output

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

def dump_POS_tags(dataset,filename):
    with open(filename, 'w') as f:
        f.write('{')
        ft = feature_extractor(['label', 'POS_curr_json'])
        for i, line in enumerate(dataset):
            tokens = expand_NE_tokens(line_split(line))
            POS_tags = ft.extract_features(tokens)
            tag_dict = {p[0][5:]: p[1][7:] for p in POS_tags}
            tokens_no_tags = [get_label(x) for x in tokens]
            sentence = " ".join(tokens_no_tags)
            f.write(json.dumps(sentence, ensure_ascii=False))
            f.write(":")
            f.write(json.dumps(tag_dict, ensure_ascii=False))
            if i < len(dataset)-1:
                f.write(',\n')
        f.write("}")

def merge_POS_tags(filename1, filename2, output_filename):
    with open(filename1) as f:
        merged_dict = json.load(f.read())
    with open(filename2) as g:
        merged_dict.update(json.load(g.read()))
    with open(output_filename, 'w') as out:
        out.write(json.dumps(merged_dict, ensure_ascii=False))


def transform_dataset(dataset, params):
    '''
    Transforms the cnec2.0 dataset into a format usable by pythoncrfsuite
    '''
    features_list = []
    labels = []
    ft = feature_extractor(params)
    for line in dataset:
        tokens = expand_NE_tokens(line_split(line))
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
