#!/usr/bin/env python3

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
        function_dict = {'label' : self.ft_get_label, 
                     'to_lower': self.ft_to_lower,
                     'neighbours_1': self.ft_get_neighbours_1,
                     'neighbours_2': self.ft_get_neighbours_2,
                     'is_capitalised': self.ft_is_capitalised,
                     '2d': self.ft_get_2d,
                     '4d': self.ft_get_4d,
                     'suffix_2': self.ft_get_suffix_2,
                     'suffix_3': self.ft_get_suffix_3}
        self.functions = [function_dict[param] for param in params] # functions to use

    def extract_features(self, tokens):
        result = []
        tokens_no_tags = [get_label(x) for x in tokens]
        for i, token in enumerate(tokens_no_tags):
            features = []
            for ft_func in self.functions:
                features.append(ft_func(token, i, tokens_no_tags))
            result.append(features)
        return result 

    def ft_get_label(self, *params):
        token = params[0]
        return "w[0]="+token

    def ft_to_lower(self, *params):
        token = params[0]
        return "lower="+token.tolower()

    def ft_get_neighbours_1(self, *params):
        assert len(params) == 3
        token,i,tokens = params
        end = len(tokens)-1
        result = ""
        if i > 0:
            result+=("w[-1]="+get_label(tokens[i-1]))
        if i < end:
            result+=("w[1]="+get_label(tokens[i+1]))
        return result

    def ft_get_neighbours_2(self, *params):
        token,i,tokens = params
        end = len(tokens)-1
        result = ""
        if i > 1:
            result+=("w[-2]="+get_label(tokens[i-2]))
        if i < end-1:
            result+=("w[2]="+get_label(tokens[i+2]))
        return result

    def ft_is_capitalised(self, *params):
        token = params[0]
        return "is_upper="+token[0].isupper()

    def ft_get_2d(self, *params):
        token = params[0]
        is_digit =  len(token) == 2 and token.isdigit()
        return "2_number="+ is_digit

    def ft_get_4d(self, *params):
        token = params[0]
        is_digit =  len(token) == 4 and token.isdigit()
        return "4_number="+ is_digit

    def ft_get_suffix_2(self, *params):
        token = params[0]
        return "suffix_2="+token[:-2]

    def ft_get_suffix_3(self, *params):
        token = params[0]
        return "suffix_3="+token[:-3]
