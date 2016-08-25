#!/usr/bin/env python3
import requests
from collections import defaultdict

def is_NE(token):
    if len(token) < 1:
        return False
    return token[0]=='<' and token[-1] == '>'

def is_embedded(token):
    return token.count('>') > 1

def merge_NE_labels(token):
    '''
    Returns a list consisting of the labels of embedded Named Entities

    Deals with embedded <P<pf Friedrich> <ps Nietzsche>> and returns
    ['Friedrich', 'Nietsche'] 
    '''
    curr_begin = 0
    curr_end = 0
    flag = False
    first = True
    result = []
    in_tag = False
    for i, ch in enumerate(token):
        if ch.isspace() and first: #the first string before space is the tag
            first = False
        # sometimes there are words between embedded NE 
        if ch.isspace() and not in_tag and not flag and not first:
            curr_begin = i
            flag = True
            continue
        if ch == '<':
            if flag and curr_begin+1!=i:
                result.append(get_NE_label(token[curr_begin+1:i]))
                flag = False
            in_tag = True
            curr_begin = i
        if ch == '>' and in_tag:
            flag = False
            in_tag = False
            curr_end = i
            result.append(get_label(token[curr_begin:curr_end+1])) 
    return result
    

def get_NE_label(token):
    '''
    returns a list of the NE labels, i.e. <gt Asii> gives ['Asii']
    <P<pf Dubenka> <ps Kralova>> returns ['Dubenka', 'Kralova']
    works recursively on embedded NE labels as well
    '''
    if not is_NE(token):
        return token
    if is_embedded(token):
        return merge_NE_labels(token[1:-1])
    return get_NE_label(token[1:-1].split(' ')[1:])

def get_label(token):
    '''
    returns a label as string instead of list
    '''
    if not is_NE(token):
        return token
    return " ".join(get_NE_label(token))


class feature_extractor:
    '''
    Extracts the features from the given sentence, can be extended by appending more features to the result list
    Right now it only appends the two neighbours of the word
    ex:
    need: POS tags, gazetteer, lemmatizer, stemmer
    [[w[0]=Vede', 'w[1]=ji'],
    ['w[0]=Izraelem', '[w-1]=mezi','w[1]=a']]

    usage ft_extractor = feature_extractor(['ft_get_label, ft_get_neighbours_1, ft_is_capitalised,
    ft_lower'])
    ft_extractor.extract_features(tokens)
    returns list of features
    '''

    def __init__(self, params):
        self.function_dict = {'label' : self.ft_get_label, 
                     'to_lower': self.ft_to_lower,
                     'is_capitalised': self.ft_is_capitalised,
                     'next' :self.ft_get_next,
                     'prev' :self.ft_get_prev,
                     'POS_curr_json': self.ft_POS_curr,
                     'POS_prev': self.ft_POS_prev,
                     'POS_next': self.ft_POS_next,
                     'POS_cond': self.ft_POS_cond,
                     'suffix_2': self.ft_get_suffix_2,
                     'suffix_3': self.ft_get_suffix_3,
                     'conditional_prev_1': self.ft_conditional_prev_1,
                     'get_type': self.ft_get_type,
                     'addr_gzttr': self.ft_addr_gzttr,
                     'name_gzttr': self.ft_name_gzttr}
        external_functs = {'addr_gzttr', 'name_gzttr'}
        self.functions = []
        self.POS_tags = {}
        function = True
        for param in params:
            if function:
                self.functions.append(self.function_dict[param])
            else:
                self.functions[-1](param,init=True)
                function = True
            if param in external_functs:
                function = False
        print("i will use following feature functions:", self.functions)

    def extract_features(self, tokens):
        result = []
        tokens_no_tags = [get_label(x) for x in tokens]
        for i, token in enumerate(tokens_no_tags):
            features = []
            for ft_func in self.functions:
                features.append(ft_func(token, i, tokens_no_tags))
            result.append(features)
        self.POS_tags={}
        return result 

    def ft_get_label(self, *params):
        token = params[0]
        return "w[0]="+token

    def ft_to_lower(self, *params):
        token = params[0]
        return "lower="+token.lower()

    def ft_get_prev(self, *params):
        token,i,tokens = params
        end = len(tokens)-1
        if i > 0:
            result=("w[-1]="+get_label(tokens[i-1]))
        else:
            result="w[-1]=START"
        return result

    def ft_get_next(self, *params):
        token,i,tokens = params
        end = len(tokens)-1
        if i < end:
            result="w[1]="+get_label(tokens[i+1])
        else:
            result="w[1]=END" 
        return result

    def ft_get_next_2(self, *params):
        token,i,tokens = params
        end = len(tokens)-1
        if i < end-1:
            result+=("w[2]="+get_label(tokens[i+2]))
        else:
            result="w[2]=END"
        return result

    def ft_get_prev_2(self, *params):
        token,i,tokens = params
        end = len(tokens)-1
        if i > 1:
            result=("w[-2]="+get_label(tokens[i-2]))
        else:
            result="w[-2]=START"
        return result

    def ft_is_capitalised(self, *params):
        token = params[0]
        print(token)
        return "is_upper="+str(token[:1].isupper())

    def ft_get_suffix_2(self, *params):
        token = params[0]
        return "suffix_2="+token[:-2]

    def ft_get_suffix_3(self, *params):
        token = params[0]
        return "suffix_3="+token[:-3]

    def ft_conditional_prev_1(self, *params):
        token, i, tokens = params
        if i==0:
            return token+"|START"
        else:
            return token+"|"+tokens[i-1]

    def ft_get_type(self, *params):
        token = params[0]
        output = ""
        for ch in token:
            if ch.isalpha():
                if ch.isupper():
                    k="A"
                else:
                    k="a"
            elif ch.isdigit():
                k="N"
            elif not ch.isalnum():
                k="."
            if not output.endswith(k):
                output += k
        return "type="+output

    def _get_POS(self, *params):
        """
        for use with production server
        Note this function needs the whole sentence
        """
        token, i, tokens = params
        url='http://cloud.ailao.eu:13880/czech_parser' 
        sentence = " ".join(tokens)
        print(sentence)
        r = requests.post(url, data=sentence.encode('utf-8'))
        tags = [x.split('\t') for x in r.text.strip().split('\n')]
        self.POS_tags = defaultdict(lambda: "none")
        self.POS_tags.update({tag[1] : tag[3] for tag in tags})
                
    def ft_POS_curr(self, *params, init=False):
        if not self.POS_tags:
            self._get_POS(params)
        token = params[0]
        if token not in self.POS_tags:
            print(token)
        return "POS[0]="+self.POS_tags[token]

    def ft_POS_prev(self, *params,init=False):
        if not self.POS_tags:
            self._get_POS(params)
        token,i,tokens = params
        if i > 0:
            return "POS[-1]="+self.POS_tags[tokens[i-1]]
        else:
            return "POS[-1]=START"

    def ft_POS_next(self, *params, init=False):
        if not self.POS_tags:
            self._get_POS(params)
        token,i,tokens = params
        end = len(tokens)-1
        if i < end:
            return "POS[1]="+self.POS_tags[tokens[i+1]]
        else:
            return "POS[1]=END"

    def ft_POS_cond(self, *param, init=False):
        if not self.POS_tags:
            self._get_POS(params)
        token,i,tokens = params
        if i > 0:
            return "POS[-1]|POS[0]="+self.POS_tags[tokens[i-1]]+"|"+self.POS_tags[token]
        else:
            return "POS[-1]|POS[0]="+self.POS_tags[tokens[i-1]] + "|START" 

    def ft_name_gzttr(self, *params, init=False):
        if init:
            self._load_name_gzttr(params[0])
            return
        token = params[0]
        flag = token in self.name_gzttr
        return "name="+str(flag)

    def _load_name_gzttr(self, filename):
        self.name_gzttr =  set()
        with open(filename) as f:
            for l in f:
                self.name_gzttr.add(l)

    def ft_addr_gzttr(self, *params,init=False):
        if init:
            self._load_addr_gzttr(params[0])
            return
        token = params[0]
        flag = token in self.addr_gzttr
        return "address="+str(flag)

    def _load_addr_gzttr(self, filename):
        self.addr_gzttr =  set()
        with open(filename) as f:
            for l in f:
                self.addr_gzttr.add(l)


