#!/usr/bin/env python3
import json
from czech_stemmer import cz_stem

def load_dataset(filename="named_ent_dtest.txt"):
    '''
    Returns the dataset where each line is a item in a list
    '''
    with open(filename) as f:
        return f.read().split('\n')

def is_NE(token):
    if len(token) < 1:
        return False
    return '<' in token and '>' in token 

def get_NE_tag(token):
    start = 1
    end = 1
    while token[end] != ' ' and token[end] != '<':
        end+=1
    return token[start:end]


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
        if tag.startswith("O"):
            print(tag)
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

def save_indices(indices, filename):
    with open(filename, 'w') as f:
        f.write(json.dumps(indices, ensure_ascii=False))

def main():
    dataset = load_dataset('named_ent_train.txt')
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
    #        stemmed_token = cz_stem(token)
            stemmed_token = token
            if stemmed_token in indices:
                continue
            else:
                indices.update({stemmed_token: i})
                i += 1
    #save_indices(tag_indices, 'tag_indices.json')
    #save_indices(indices, 'token_indices.json')
    print('{} total words'.format(i))
    print('{} max sentence length'.format(max_l))
    print(len(tag_indices))
    print(tag_indices)
if __name__ == '__main__':
    main()
