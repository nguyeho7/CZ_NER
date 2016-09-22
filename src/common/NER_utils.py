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

def load_dataset_json(filename, params):
    j = json.load(open(filename))
    sentences = []
    y_gold = []
    ft = feature_extractor(params)
    for question in j:
        features = ft.extract_features(question['tokens'])
        sentences.append(features)
        y_gold.append(question['entity-labels'])
    return y_gold, sentences

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

def get_NE_tag(token):
    start = 1
    end = 1
    while token[end] != ' ' and token[end] != '<':
        end+=1
    return token[start:end]

def is_NE(token):
    if len(token) < 1:
        return False
    return '<' in token and '>' in token 

def is_special(tag):
    """
    special non NER tags
    f = foreign word
    segm = wrongly segmented word (start of next sentence)
    cap = capitalised word
    lower = word wrongly capitalised
    upper = word wrongly written in lowercase
    s = shortcut
    ? = unspecified
    ! = not annotated
    A,C,P,T are containers for embedded annotations
    """
    return tag in {"A", "C", "P", "T","s", "f", "segm", "cap", "lower", "upper", "?", "!"}


def get_NE_label(token):
    '''
    returns a list of the NE labels, i.e. <gt Asii> gives ['Asii']
    <P<pf Dubenka> <ps Kralova>> returns ['Dubenka', 'Kralova']
    works recursively on embedded NE labels as well
    '''
    result = []
    for ch in token.split(' '):
        if '<' in ch:
            continue
        if '>' in ch:
            result.append(ch.split('>')[0])
        else:
            result.append(ch)
    return [r.strip() for r in result]

def get_label(token):
    '''
    returns a label as string instead of list
    '''
    return " ".join(get_NE_label(token))

def get_labels_tags(tokens, merge="none"):
    '''
    Goes through all tokens from line_split and either adds them outright
    or expands them, removing the inner tags and instead appending
    outertag_b for first, outertag_i for inner and outertag_b for last word 
    <P<pf Václavu> <ps Klausovi>> turns into:
    ['<P_b Václavu>', '<P_e Klausovi>']
    here we use the tag P instead of pf/ps
    will be hard to compare performance like this
    currently returns the supertypes, with merge=True it returns only B, I, O
    returns list of tokens with NE tokens expanded
    i
    '''
    tags = []
    words= []
    for token in tokens:
        if token == "":
            continue
        if not is_NE(token):
            words.append(token)
            tags.append("O")
        elif is_special(get_NE_tag(token)):
            tag = get_NE_tag(token)
            token = token[1 + len(tag):-1].strip()    
            w_sub, t_sub = get_labels_tags(line_split(token))
            tags.extend(t_sub)
            words.extend(w_sub)
        else:
            if token[0] == "(":
                token = token[1:]
            if token[-1] == ")":
                token = token[:-1]
            tag = get_NE_tag(token)
            labels = []
            for label in expand_NE(get_label(token)):
                labels.extend(label.split(' '))
            for i,label in enumerate(labels):
                words.append(label)
                if i == 0:
                    tags.append('{}_b'.format(tag))
                else:
                    tags.append('{}_i'.format(tag))
    if merge == "supertype":
        tags = [supertype_tag(tag) for tag in tags]
    elif merge == "BIO":
        tags = [merge_tag(tag) for tag in tags]
    return words, tags 

def merge_tag(tag):
    if tag == "O":
        return tag
    else:
        return tag[-1]

def supertype_tag(tag):
    if tag == "O":
        return tag
    else:
        return tag[0] + "_" + tag[-1]


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

def get_NE_tag(token):
    start = 1
    end = 1
    while token[end] != ' ' and token[end] != '<':
        end+=1
    return token[start:end]

def dump_POS_tags(dataset,filename):
    downloaded = 0
    with open(filename, 'r') as g:
        for line in g:
            downloaded += 1
    with open(filename, 'a') as f:
        #f.write('{')
        f.write('\n')
        ft = feature_extractor(['label', 'POS_curr_json'])
        for i, line in enumerate(dataset):
            if i < downloaded:
                continue
            tokens, tags = get_labels_tags(line_split(line))
            POS_tags = ft.extract_features(tokens, string_format=False)
            print(POS_tags)
            tag_dict = {p['w[0]']: p['POS[0]'] for p in POS_tags}
            sentence = " ".join(tokens)
            f.write(json.dumps(sentence, ensure_ascii=False))
            f.write(":")
            f.write(json.dumps(tag_dict, ensure_ascii=False))
            if i < len(dataset)-1:
                f.write(',\n')
        f.write("}")

def dump_POS_tags_2(dataset, filename):
    sentences = []
    ft = feature_extractor()
    for i, line in enumerate(dataset):
        tokens, tags = get_labels_tags(line_split(line))
        sentences.append(' '.join(tokens))
    text = "*#!*".join(sentences)
    url='http://cloud.ailao.eu:4571/czech_parser' 
    r = requests.post(url, data=text.encode('utf-8'))
    tags = list(filter(lambda y: len(y) > 1, (x.split('\t') for x in r.text.strip().split('\n'))))
    start = 0
    end = 0
    tags_per_sentence = {}
    for i, tag in enumerate(tags):
        if i == 0:
            continue
        if tag[0] == '1':
            end = i
            sentence = " ".join(x[1] for x in tags[start:end])
            tags_per_sentence.update({sentence: {x[1] : x[3] for x in tags[start:end]}})
            start = end
    with open(filename, "w") as f:
        f.write(json.dumps(tags_per_sentence))

def merge_POS_tags(filename1, filename2, output_filename):
    with open(filename1) as f:
        merged_dict = json.load(f.read())
    with open(filename2) as g:
        merged_dict.update(json.load(g.read()))
    with open(output_filename, 'w') as out:
        out.write(json.dumps(merged_dict, ensure_ascii=False))


def transform_dataset(dataset, params, merge="supertype"):
    '''
    Transforms the cnec2.0 dataset into a format usable by pythoncrfsuite
    '''
    features_list = []
    labels = []
    ft = feature_extractor(params)
    for line in dataset:
        tokens, tags = get_labels_tags(line_split(line), merge)
        labels.append(tags)
        features_list.append(ft.extract_features(tokens))
    return labels, features_list

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
