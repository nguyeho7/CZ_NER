#!/usr/bin/env python3
import requests
import json
from collections import defaultdict

def is_NE(token):
    if len(token) < 1:
        return False
    return '<' in token and '>' in token 

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
                     'next_2':self.ft_get_next_2,
                     'prev_2':self.ft_get_prev_2,
                     'POS_curr': self.ft_POS_curr,
                     'POS_curr_json': self.ft_POS_curr,
                     'POS_prev': self.ft_POS_prev,
                     'POS_next': self.ft_POS_next,
                     'POS_prev_2': self.ft_POS_prev_2,
                     'POS_next_2': self.ft_POS_next_2,
                     'POS_cond': self.ft_POS_cond,
                     'suffix_2': self.ft_get_suffix_2,
                     'suffix_3': self.ft_get_suffix_3,
                     'conditional_prev_1': self.ft_conditional_prev_1,
                     'clusters_8': self.ft_bclusters_8,
                     'clusters_12': self.ft_bclusters_12,
                     'clusters_16': self.ft_bclusters_16,
                     'get_type': self.ft_get_type,
                     'addr_gzttr': self.ft_addr_gzttr,
                     'name_gzttr': self.ft_name_gzttr}
        external_functs = {'addr_gzttr', 'name_gzttr', 'POS_curr', 'clusters_8'}
        self.functions = []
        self.POS_dict = {}
        self.POS_tags = {}
        self.clusters = defaultdict(lambda:'-1')
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
        for i, token in enumerate(tokens):
            features = []
            for ft_func in self.functions:
                features.append(ft_func(token, i, tokens))
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
            return "w[-1]="+get_label(tokens[i-1])
        else:
            return "w[-1]=START"

    def ft_get_next(self, *params):
        token,i,tokens = params
        end = len(tokens)-1
        if i < end:
            return "w[1]="+get_label(tokens[i+1])
        else:
            return"w[1]=END" 

    def ft_get_next_2(self, *params):
        token,i,tokens = params
        end = len(tokens)-1
        if i < end-1:
            return "w[2]="+get_label(tokens[i+2])
        else:
            return "w[2]=END"

    def ft_get_prev_2(self, *params):
        token,i,tokens = params
        end = len(tokens)-1
        if i > 1:
            return "w[-2]="+get_label(tokens[i-2])
        else:
            return "w[-2]=START"

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


    def _get_POS(self, params):
        """
        get POS tag for given sentence, either from the loaded json
        or online
        """
        token, i, tokens = params
        sentence = " ".join(tokens)
        if not self.POS_dict:
            self._get_POS_online(params)
        elif sentence not in self.POS_dict:
            print('should not happen')
            self._get_POS_online(params)
        else:
            self.POS_tags = defaultdict(lambda: "none")
            self.POS_tags.update(self.POS_dict[sentence])

    def _get_POS_online(self, params):
        """
        for use with production server
        Note this function needs the whole sentence
        """
        token, i, tokens = params
        url='http://cloud.ailao.eu:4070/czech_parser' 
        sentence = " ".join(tokens)
        r = requests.post(url, data=sentence.encode('utf-8'))
        tags = [x.split('\t') for x in r.text.strip().split('\n')]
        self.POS_tags = defaultdict(lambda: "none")
        self.POS_tags.update({tag[1] : tag[3] for tag in tags})

    def _load_POS(self, filename):
        with open(filename) as f:
            if not self.POS_dict:
                self.POS_dict = json.loads(f.read())
            else:
                self.POS_dict.update(json.loads(f.read()))

    def ft_POS_curr(self, *params, init=False):
        if init:
            self._load_POS(params[0])
            return
        if not self.POS_tags:
            self._get_POS(params)
        token = params[0]
        if token not in self.POS_tags:
            print(token)
        return "POS[0]="+self.POS_tags[token]

    def ft_POS_prev(self, *params):
        if not self.POS_tags:
            self._get_POS(params)
        token,i,tokens = params
        if i > 0:
            return "POS[-1]="+self.POS_tags[tokens[i-1]]
        else:
            return "POS[-1]=START"

    def ft_POS_next(self, *params):
        if not self.POS_tags:
            self._get_POS(params)
        token,i,tokens = params
        end = len(tokens)-1
        if i < end:
            return "POS[1]="+self.POS_tags[tokens[i+1]]
        else:
            return "POS[1]=END"

    def ft_POS_prev_2(self, *params):
        if not self.POS_tags:
            self._get_POS(params)
        token,i,tokens = params
        if i > 1:
            return "POS[-2]="+self.POS_tags[tokens[i-2]]
        else:
            return "POS[-2]=START"

    def ft_POS_next_2(self, *params):
        if not self.POS_tags:
            self._get_POS(params)
        token,i,tokens = params
        end = len(tokens)-1
        if i < end-1:
            return "POS[2]="+self.POS_tags[tokens[i+1]]
        else:
            return "POS[2]=END"

    def ft_POS_cond(self, *params):
        if not self.POS_tags:
            self._get_POS(params)
        token,i,tokens = params
        if i > 0:
            return "POS[-1]|POS[0]="+self.POS_tags[tokens[i-1]]+"|"+self.POS_tags[token]
        else:
            return "POS[-1]|POS[0]="+self.POS_tags[tokens[i-1]] + "|START" 

    def ft_bclusters_8(self, *params, init=False):
        if init:
            self._load_clusters(params[0])
        token = params[0]
        return "BC_8="+self.clusters[token.lower()][:8]

    def ft_bclusters_12(self, *params):
        token = params[0]
        return "BC_12="+self.clusters[token.lower()][:12]

    def ft_bclusters_16(self, *params):
        token = params[0]
        return "BC_16="+self.clusters[token.lower()][:16]

    def _load_clusters(self, filename):
        with open(filename) as f:
            for l in f:
                l_split = l.split('\t')
                self.clusters.update({l_split[1] : l_split[0]})

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


