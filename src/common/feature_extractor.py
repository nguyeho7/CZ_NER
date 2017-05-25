#!/usr/bin/env python3
import requests
import json
from collections import defaultdict


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
        self.function_dict = { 
                     'is_capitalised': self.ft_is_capitalised,
                     'suffix_2': self.ft_get_suffix_2,
                     'suffix_3': self.ft_get_suffix_3,
                     'prefix_2': self.ft_get_prefix_2,
                     'prefix_3': self.ft_get_prefix_3,
                     'get_type': self.ft_get_type,
                     'per_gzttr' : self.ft_per_gzttr,
                     'loc_gzttr' : self.ft_loc_gzttr,
                     'misc_gzttr' : self.ft_misc_gzttr,
                     'org_gzttr' : self.ft_org_gzttr,
                     'contains_at': self.ft_contains_at,
                     'contains_digit': self.ft_contains_digit}

        external_functs = {'per_gzttr', 'loc_gzttr', 'misc_gzttr', 'org_gzttr'}
        self.functions = []
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


    def extract_features_sentence_conll(self, features, string_format=False):
        result = []
        for i, fts in enumerate(features):
            token = fts['label']
            curr_ft = {}
            # prev and next label
            curr_ft.update({"w[0]" : fts['label']})
            curr_ft.update(self.ft_conll_get_prev(features,"label", "w[-1]", i))
            curr_ft.update(self.ft_conll_get_next(features,"label", "w[1]", i))
            curr_ft.update(self.ft_conll_get_prev_2(features,"label", "w[-2]", i))
            curr_ft.update(self.ft_conll_get_next_2(features,"label", "w[2]", i))
            curr_ft.update(self.ft_conll_conditional_prev_1(features, 'label', 'label', i))
            # prev and next POS tag
            """
            curr_ft.update({"pos[0]" : fts['pos']})
            curr_ft.update(self.ft_conll_get_prev(features,"pos", "pos[-1]", i))
            curr_ft.update(self.ft_conll_get_next(features,"pos", "pos[1]", i))
            curr_ft.update(self.ft_conll_get_prev_2(features,"pos", "pos[-2]", i))
            curr_ft.update(self.ft_conll_get_next_2(features,"pos", "pos[2]", i))
            curr_ft.update(self.ft_conll_conditional_prev_1(features, 'pos', 'pos', i))

            curr_ft.update({"chunk[0]" : fts['dep']})
            curr_ft.update(self.ft_conll_get_prev(features,"dep", "chunk[-1]", i))
            curr_ft.update(self.ft_conll_get_next(features,"dep", "chunk[1]", i))
"""
            for ft_func in self.functions:
                key, value = ft_func(features, "ft", token, i)
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
        if i < end - 1:
            return {label: features[i+1][feature]}
        else:
            return {label: "END"}

    def ft_conll_get_prev_2(self, features, feature, label, i):
        end = len(features)-1
        if i > 1:
            return {label: features[i-2][feature]}
        else:
            return {label: "START"}

    def ft_conll_get_next_2(self, features, feature, label, i):
        end = len(features)-1
        if i < end-1:
            return {label: features[i+2][feature]}
        else:
            return {label: "END"}

    def ft_conll_conditional_prev_1(self, features, feature, label, i):
        if i==0:
            return {"cond_"+feature : features[i][feature]+ "|START"}
        else:
            return{"cond_"+feature : features[i][feature]+ "|"+features[i-1][feature]}

    def ft_is_capitalised(self, *params):
        features, feature, token, i = params
        return "is_upper", token[:1].isupper()

    def ft_get_suffix_2(self, *params):
        features, feature, token, i = params
        return "suffix_2", token[-2:]

    def ft_get_suffix_3(self, *params):
        features, feature, token, i = params
        return "suffix_3",token[-3:]

    def ft_get_prefix_2(self, *params):
        features, feature, token, i = params
        return "prefix_2",token[:2]

    def ft_get_prefix_3(self, *params):
        features, feature, token, i = params
        return "prefix_3",token[:3]


    def ft_contains_at(self, *params):
        features, feature, token, i = params
        flag = "@" in token
        return "at", flag

    def ft_contains_digit(self, *params):
        features, feature, token, i = params
        flag = any(char.isdigit() for char in token)
        return "contains_digit", flag

    def ft_get_type(self, *params):
        features, feature, token, i = params
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

    def ft_loc_gzttr(self, *params, init=False):
        if init:
            self._load_loc_gzttr(params[0])
            return
        features, feature, label, i = params
        token = features[i]['label']
        end = len(features)
        flag = False
        if token in self.loc_gzttr_1word:
            flag = True
        elif token in self.loc_gzttr_after:
            if i+1 < end:
                next_token = features[i+1]['label']
                if next_token in self.loc_gzttr_after[token]:
                    flag = True
        elif token in self.loc_gzttr_before:
            if i>0:
                next_token = features[i-1]['label']
                if next_token in self.loc_gzttr_before[token]:
                    flag = True
        return "loc", flag

    def _load_loc_gzttr(self, filename):
        self.loc_gzttr_1word =  set()
        self.loc_gzttr_after =  defaultdict(set)
        self.loc_gzttr_before =  defaultdict(set)
        with open(filename) as f:
            for l in f:
                l = l.strip()
                tokens = l.split(' ')
                if len(tokens) == 2:
                    self.loc_gzttr_1word.add(tokens[1])
                else:
                    for i in range(3, len(tokens)):
                        subtoken = tokens[i]
                        subtoken_prev = tokens[i-1]
                        self.loc_gzttr_after[subtoken_prev].add(subtoken)
                        self.loc_gzttr_before[subtoken].add(subtoken_prev)

    def ft_misc_gzttr(self, *params, init=False):
        if init:
            self._load_misc_gzttr(params[0])
            return
        features, feature, label, i = params
        token = features[i]['label']
        end = len(features)
        flag = False
        if token in self.misc_gzttr_1word:
            flag = True
        elif token in self.misc_gzttr_after:
            if i+1 < end:
                next_token = features[i+1]['label']
                if next_token in self.misc_gzttr_after[token]:
                    flag = True
        elif token in self.misc_gzttr_before:
            if i>0:
                next_token = features[i-1]['label']
                if next_token in self.misc_gzttr_before[token]:
                    flag = True
        return "misc", flag

    def _load_misc_gzttr(self, filename):
        self.misc_gzttr_1word =  set()
        self.misc_gzttr_after =  defaultdict(set)
        self.misc_gzttr_before =  defaultdict(set)
        with open(filename) as f:
            for l in f:
                l = l.strip()
                tokens = l.split(' ')
                if len(tokens) == 2:
                    self.misc_gzttr_1word.add(tokens[1])
                else:
                    for i in range(3, len(tokens)):
                        subtoken = tokens[i]
                        subtoken_prev = tokens[i-1]
                        self.misc_gzttr_after[subtoken_prev].add(subtoken)
                        self.misc_gzttr_before[subtoken].add(subtoken_prev)


    def ft_per_gzttr(self, *params, init=False):
        if init:
            self._load_per_gzttr(params[0])
            return
        features, feature, label, i = params
        token = features[i]['label']
        end = len(features)
        flag = False
        if token in self.per_gzttr_1word:
            flag = True
        elif token in self.per_gzttr_after:
            if i+1 < end:
                next_token = features[i+1]['label']
                if next_token in self.per_gzttr_after[token]:
                    flag = True
        elif token in self.per_gzttr_before:
            if i>0:
                next_token = features[i-1]['label']
                if next_token in self.per_gzttr_before[token]:
                    flag = True
        return "per", flag

    def _load_per_gzttr(self, filename):
        self.per_gzttr_1word =  set()
        self.per_gzttr_after =  defaultdict(set)
        self.per_gzttr_before =  defaultdict(set)
        with open(filename) as f:
            for l in f:
                l = l.strip()
                tokens = l.split(' ')
                if len(tokens) == 2:
                    self.per_gzttr_1word.add(tokens[1])
                else:
                    for i in range(3, len(tokens)):
                        subtoken = tokens[i]
                        subtoken_prev = tokens[i-1]
                        self.per_gzttr_after[subtoken_prev].add(subtoken)
                        self.per_gzttr_before[subtoken].add(subtoken_prev)


    def ft_org_gzttr(self, *params, init=False):
        if init:
            self._load_org_gzttr(params[0])
            return
        features, feature, label, i = params
        token = features[i]['label']
        end = len(features)
        flag = False
        if token in self.org_gzttr_1word:
            flag = True
        elif token in self.org_gzttr_after:
            if i+1 < end:
                next_token = features[i+1]['label']
                if next_token in self.org_gzttr_after[token]:
                    flag = True
        elif token in self.org_gzttr_before:
            if i>0:
                next_token = features[i-1]['label']
                if next_token in self.org_gzttr_before[token]:
                    flag = True
        return "org", flag

    def _load_org_gzttr(self, filename):
        self.org_gzttr_1word =  set()
        self.org_gzttr_after =  defaultdict(set)
        self.org_gzttr_before =  defaultdict(set)
        with open(filename) as f:
            for l in f:
                l = l.strip()
                tokens = l.split(' ')
                if len(tokens) == 2:
                    self.org_gzttr_1word.add(tokens[1])
                else:
                    for i in range(3, len(tokens)):
                        subtoken = tokens[i]
                        subtoken_prev = tokens[i-1]
                        self.org_gzttr_after[subtoken_prev].add(subtoken)
                        self.org_gzttr_before[subtoken].add(subtoken_prev)

