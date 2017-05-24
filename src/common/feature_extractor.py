#!/usr/bin/env python3
import requests
import json
from collections import defaultdict
from czech_stemmer import cz_stem
from ufal.morphodita import *


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
                     'lemma':self.ft_lemma,
                     'POS_curr': self.ft_POS_curr,
                     'POS_curr_json': self.ft_POS_curr,
                     'POS_prev': self.ft_POS_prev,
                     'POS_next': self.ft_POS_next,
                     'POS_prev_2': self.ft_POS_prev_2,
                     'POS_next_2': self.ft_POS_next_2,
                     'POS_cond': self.ft_POS_cond,
                     'suffix_2': self.ft_get_suffix_2,
                     'suffix_3': self.ft_get_suffix_3,
                     'prefix_2': self.ft_get_prefix_2,
                     'prefix_3': self.ft_get_prefix_3,
                     'conditional_prev_1': self.ft_conditional_prev_1,
                     'clusters_4': self.ft_bclusters_4,
                     'clusters_8': self.ft_bclusters_8,
                     'clusters_12': self.ft_bclusters_12,
                     'clusters_16': self.ft_bclusters_16,
                     'clusters_20': self.ft_bclusters_20,
                     'get_type': self.ft_get_type,
                     'addr_gzttr': self.ft_addr_gzttr,
                     'name_gzttr': self.ft_name_gzttr,
                     'lname_gzttr': self.ft_lname_gzttr,
                     'contains_at': self.ft_contains_at,
                     'contains_digit': self.ft_contains_digit}

        external_functs = {'addr_gzttr', 'name_gzttr', 'POS_curr_json', 'clusters_8', 'lname_gzttr'}
        self.functions = []
        self.POS_dict = {}
        self.POS_tags = {}
        self.clusters = defaultdict(lambda:'-1')
        self.morpho = Morpho.load("czech-morfflex-161115.dict")
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

    def lemmatize(self, token):
        lemmas = TaggedLemmas()
        result = self.morpho.analyze(token, self.morpho.GUESSER, lemmas)
        return {"lemma": lemmas[0].lemma, "raw": self.morpho.rawLemma(lemmas[0].lemma),
                "tag":lemmas[0].tag}

    def extract_features(self, tokens, string_format=True):
        """
        Given a tokenized sentence, returns either a list of features for each word
        [["w[0]=first", "POS[0]=VB", "cap=True"],["w[0]=second", "POS[0]=ADJ"], ..] if string_format is true
        or a list of dictionaries of features for each word
        [{"w[0]": "first", "POS[0]": "VB", "cap": True}, {"w[0]": "second", "POS[0]": "ADJ"}, ..]
        Both use the same feature functions defined in init
        """
        result = []
        for i, token in enumerate(tokens):
            features = {}
            for ft_func in self.functions:
                key, value = ft_func(token,i,tokens)
                features.update({key: value})
            if string_format:
                ft_list = [key+"="+str(value) for key, value in features.items()]
                result.append(ft_list)
            else:
                result.append(features)
        self.POS_tags={}
        return result 

    def extract_features_sentence_conll(self, features, string_format=False):
        result = []
        for i, fts in enumerate(features):
            token = fts['label']
            curr_ft = {}
            # prev and next label
            curr_ft.update({"w[0]" : fts['label']})
            curr_ft.update(self.ft_conll_get_prev(features,"label", "w[-1]", i))
            curr_ft.update(self.ft_conll_get_next(features,"label", "w[1]", i))
            # prev and next POS tag
            curr_ft.update({"pos[0]" : fts['pos']})
            curr_ft.update(self.ft_conll_get_prev(features,"pos", "pos[-1]", i))
            curr_ft.update(self.ft_conll_get_next(features,"pos", "pos[1]", i))

            curr_ft.update({"dep[0]" : fts['dep']})
            curr_ft.update(self.ft_conll_get_prev(features,"dep", "dep[-1]", i))
            curr_ft.update(self.ft_conll_get_next(features,"dep", "dep[1]", i))
            for ft_func in self.functions:
                key, value = ft_func(token, i)
                curr_ft.update({key: str(value)})
            if string_format:
                ft_list = [key+"="+str(value) for key, value in curr_ft.items()]
                result.append(ft_list)
            else:
                result.append(curr_ft)
        return result

    def ft_conll_get_prev(self, features, feature, label, i):
        end = len(features)-1
        if i > 0:
            return {label: features[i-1][feature]}
        else:
            return {label: "START"}

    def ft_conll_get_next(self, features, feature, label, i):
        end = len(features)-1
        if i < end:
            return {label: features[i+1][feature]}
        else:
            return {label: "END"}

    def ft_get_label(self, *params):
        token = params[0]
        lemmas = self.lemmatize(token)
        return "w[0]", token

    def ft_lemma(self, *params):
        token = params[0]
        lemmas = self.lemmatize(token)
        return "lemma[0]", lemmas['raw']

    def ft_to_lower(self, *params):
        token = params[0]
        return "lower", token.lower()

    def ft_get_prev(self, *params):
        token,i,tokens = params
        end = len(tokens)-1
        if i > 0:
            lemmas = self.lemmatize(tokens[i-1])
            return "w[-1]", lemmas['raw']
        else:
            return "w[-1]", "START"

    def ft_get_next(self, *params):
        token,i,tokens = params
        end = len(tokens)-1
        if i < end:
            lemmas = self.lemmatize(tokens[i+1])
            return "w[1]", lemmas['raw']
        else:
            return"w[1]"," END" 

    def ft_get_next_2(self, *params):
        token,i,tokens = params
        end = len(tokens)-1
        if i < end-1:
            lemmas = self.lemmatize(tokens[i+1])
            return "w[2]", lemmas['raw']
        else:
            return "w[2]", "END"

    def ft_get_prev_2(self, *params):
        token,i,tokens = params
        end = len(tokens)-1
        if i > 1:
            lemmas = self.lemmatize(tokens[i-2])
            return "w[-2]", lemmas['raw']
        else:
            return "w[-2]", "START"

    def ft_is_capitalised(self, *params):
        token = params[0]
        return "is_upper", token[:1].isupper()

    def ft_get_suffix_2(self, *params):
        token = params[0]
        return "suffix_2", token[-2:]

    def ft_get_suffix_3(self, *params):
        token = params[0]
        return "suffix_3",token[-3:]

    def ft_get_prefix_2(self, *params):
        token = params[0]
        return "prefix_2",token[:2]

    def ft_get_prefix_3(self, *params):
        token = params[0]
        return "prefix_3",token[:3]

    def ft_conditional_prev_1(self, *params):
        token, i, tokens = params
        if i==0:
            return token, "|START"
        else:
            return token, "|"+tokens[i-1]

    def ft_contains_at(self, *params):
        token = params[0]
        flag = "@" in token
        return "at", flag

    def ft_contains_digit(self, *params):
        token = params[0]
        flag = any(char.isdigit() for char in token)
        return "contains_digit", flag

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
        return "type", output


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
        url='http://cloud.ailao.eu:4570/czech_parser' 
        sentence = " ".join(tokens)
        r = requests.post(url, data=sentence.encode('utf-8'))
        tags = [x.split('\t') for x in r.text.strip().split('\n')]
        self.POS_tags = defaultdict(lambda: "none")
        print(sentence)
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
        return "POS[0]", self.POS_tags[token]

    def ft_POS_prev(self, *params):
        if not self.POS_tags:
            self._get_POS(params)
        token,i,tokens = params
        if i > 0:
            return "POS[-1]", self.POS_tags[tokens[i-1]]
        else:
            return "POS[-1]","START"

    def ft_POS_next(self, *params):
        if not self.POS_tags:
            self._get_POS(params)
        token,i,tokens = params
        end = len(tokens)-1
        if i < end:
            return "POS[1]",self.POS_tags[tokens[i+1]]
        else:
            return "POS[1]","END"

    def ft_POS_prev_2(self, *params):
        if not self.POS_tags:
            self._get_POS(params)
        token,i,tokens = params
        if i > 1:
            return "POS[-2]", self.POS_tags[tokens[i-2]]
        else:
            return "POS[-2]", "START"

    def ft_POS_next_2(self, *params):
        if not self.POS_tags:
            self._get_POS(params)
        token,i,tokens = params
        end = len(tokens)-1
        if i < end-1:
            return "POS[2]", self.POS_tags[tokens[i+1]]
        else:
            return "POS[2]", "END"

    def ft_POS_cond(self, *params):
        if not self.POS_tags:
            self._get_POS(params)
        token,i,tokens = params
        if i > 0:
            return "POS[-1]|POS[0]", self.POS_tags[tokens[i-1]]+"|"+self.POS_tags[token]
        else:
            return "POS[-1]|POS[0]", self.POS_tags[tokens[i-1]] + "|START" 

    def ft_bclusters_4(self, *params):
        token = params[0]
        return "BC_4", self.clusters[token.lower()][:4]

    def ft_bclusters_6(self, *params):
        token = params[0]
        return "BC_4", self.clusters[token.lower()][:4]

    def ft_bclusters_8(self, *params, init=False):
        if init:
            self._load_clusters(params[0])
        token = params[0]
        return "BC_8", self.clusters[token.lower()][:8]

    def ft_bclusters_12(self, *params):
        token = params[0]
        return "BC_12", self.clusters[token.lower()][:12]

    def ft_bclusters_16(self, *params):
        token = params[0]
        return "BC_16", self.clusters[token.lower()][:16]

    def ft_bclusters_20(self, *params):
        token = params[0]
        return "BC_20", self.clusters[token.lower()][:20]

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
        flag = self.lemmatize(token)['raw'] in self.name_gzttr
        return "name", flag

    def _load_name_gzttr(self, filename):
        self.name_gzttr =  set()
        with open(filename) as f:
            for l in f:
                self.name_gzttr.add(l.strip())

    def ft_addr_gzttr(self, *params,init=False):
        if init:
            self._load_addr_gzttr(params[0])
            return
        token,i,tokens = params
        end = len(tokens)
        token_lemma = self.lemmatize(token)['raw']
        flag = False
        if token_lemma in self.addr_gzttr:
            flag = True
        elif token in self.addr_long_gzttr:
            if i+1 < end:
                next_token = tokens[i+1].title()
                next_token_lemma = self.lemmatize(next_token)['raw'].title()
                if next_token in self.addr_long_gzttr[token] or next_token_lemma in self.addr_long_gzttr[token]:
                    flag = True
        return "address", flag

    def _load_addr_gzttr(self, filename):
        self.addr_gzttr =  set()
        self.addr_long_gzttr = defaultdict(set)
        with open(filename) as f:
            for l in f:
                token = l.title().strip()
                if token.isdigit():
                    continue
                if " " in token:
                    subtokens = token.split(" ")
                    self.addr_long_gzttr[subtokens[0]].add(" ".join(subtokens[1:]))
                else:
                    self.addr_gzttr.add(token)

    def ft_lname_gzttr(self, *params, init=False):
        if init:
            self._load_lname_gzttr(params[0])
            return
        token = params[0]
        flag = self.lemmatize(token)['raw'] in self.lname_gzttr
        return "last_name", flag

    def _load_lname_gzttr(self, filename):
        self.lname_gzttr =  set()
        with open(filename) as f:
            for l in f:
                self.lname_gzttr.add(l.strip())
