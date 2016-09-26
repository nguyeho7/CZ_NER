#!/usr/bin/env python3
import src.common.NER_utils as t
from src.common.eval import global_eval, output_evaluation, random_sample
from keras.models import Sequential, load_model, model_from_json, Model
from keras.layers import Input, Embedding, Bidirectional, Merge, TimeDistributed, Dense, LSTM, merge
from keras.preprocessing.sequence import pad_sequences
from gensim.models.word2vec import Word2Vec
from string import punctuation
import numpy as np
import json

# stopwords are chosen by taking the top20 most common words that were not found in w2v
stopwords = {'je', 've', 'z', 'v', 'se', 'o', 's', 'si', 'by', 'k', 'i', 'od', 'a', 'to', 'na',
    'po', 'do', 'u', 'za'}
pos_index = {"ADJ": 1, "ADP": 2, "ADV": 3, "AUX": 4, "CONJ": 5, "DET": 6, "INTJ": 7, "NOUN": 8, 
        "NUM": 9, "PART": 10, "PRON": 11, "PROPN": 12, "PUNCT": 13, "SCONJ": 14, "SYM": 15, "VERB":15, "X": 16, 'none' : 0}
feature_functs = ["is_upper", "name", "address"]

def load_embedding_matrix(filename='testmodel-139m_p3___.bin'):
    w = Word2Vec.load(filename)
    word2index = {x : w.vocab[x].index + 5 for x in w.vocab}
    sorted_list = sorted(word2index.items(), key=lambda x: x[1])
    embeddings = []
    embeddings.append(np.zeros(100)) # padding
    embeddings.append([np.random.normal() for x in range(100)]) #is_digit 
    embeddings.append([np.random.normal() for x in range(100)]) #is_punctuation
    embeddings.append([np.random.normal() for x in range(100)]) #is_stopword 
    embeddings.append([np.random.normal() for x in range(100)]) #OOV 
    embeddings.extend(w[x[0]] for x in sorted_list)
    embeddings_np = np.array(embeddings)
    return word2index, embeddings_np

def get_data(filename, indices, tag_indices, validation=False, merge="none"):
    raw_data = t.load_dataset(filename)
    print("opening ", filename)
    vocab_size = len(indices)
    X_train = []
    Y_train = []
    text = []
    ft_params = ['POS_curr', 'POS_final.json', "is_capitalised", 'addr_gzttr', 'adresy.txt',
            'name_gzttr','czech_names']
    ft = t.feature_extractor(ft_params)
    feature_list = []
    POS = []
    for line in raw_data:
        if len(line) == 0:
            continue
        tokens, tags = t.get_labels_tags(t.line_split(line), merge)
        features = ft.extract_features(tokens, string_format=False)
        POS.append(vectorize_POS(features))
        feature_list.append(vectorize_features(features))
        text.append(tokens)
        tokens_vector= vectorize_sentence(tokens, indices)
        X_train.append(tokens_vector)
        if validation:
            Y_train.append(tags[:60])
        else:
            Y_train.append(vectorize_tags(tags, tag_indices))
    x_train = pad_sequences(np.array(X_train), maxlen=60)
    if validation:
        y_train = np.array(Y_train)
    else:
        y_train = pad_sequences(np.array(Y_train), maxlen=60)
    POS_np = pad_sequences(np.array(POS), maxlen=60)
    feature_list_np = pad_sequences(np.array(feature_list), maxlen=60)
    return x_train, y_train, POS_np, feature_list_np, text, vocab_size

def vectorize_POS(features):
    result = [[0 for x in range(len(pos_index))] for y in range(len(features))]
    for i, feature in enumerate(features):
        result[i][pos_index[feature['POS[0]']]] = 1
    return np.array(result)

def vectorize_features(features):
    #for every feature apart from POS tags
    # list of features:
    # is_capitalised, addr_gzttr, name_gzttr
    result = [[0 for x in range(len(feature_functs))] for y in range(len(features))]
    for i, word_ft in enumerate(features):
        for j, ft in enumerate(feature_functs):
            if word_ft[ft]:
                result[i][j] = 1
    return np.array(result)
            
def vectorize_tags(tags, idx, merge=False):
    res = np.zeros((len(tags), len(idx)))
    for i,tag in enumerate(tags):
        res[i][idx[tag]] = 1
    return res

def vectorize_sentence(tokens,word_idx):
    result = []
    for token in tokens:
        tok = token
        #tok = token.lower()
        if tok in word_idx:
            result.append(word_idx[tok])
        else:
            print(tokens)
            print("this should not happen")
            result.append(0)
    #    elif tok.isdigit():
     #       result.append(1)
      #  elif tok in punctuation:
       #     result.append(2)
        #elif tok in stopwords:
         #   result.append(3)
        #else:
         #   result.append(4)
    return result

def load_indices(filename):
    with open(filename) as f:
        return json.loads(f.read())

def define_model_w2v(vocab_size, tags, embeddings):
    model = Sequential()
    model.add(Embedding(input_dim = embeddings.shape[0], output_dim=embeddings.shape[1],
        weights=[embeddings], input_length=60, mask_zero=True))
    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(TimeDistributed(Dense(tags, activation='softmax')))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    return model

def define_model_concat(vocab_size, tags, embeddings,  POS_vectors, feature_vectors):
    sentence_input = Input(shape=(60,), dtype='int32')
    embedding_layer = Embedding(input_dim = embeddings.shape[0], output_dim=embeddings.shape[1],
        weights=[embeddings], input_length=60, mask_zero=True)(sentence_input)

    POS_input = Input(shape=(60, len(pos_index)))
 #   POS_layer = Dense(len(pos_index))(POS_input)

    feature_input = Input(shape=(60, len(feature_functs)))
#    feature_layer = Dense(len(feature_functs))(feature_input)
    
    merged = merge([embedding_layer, POS_input, feature_input], mode='concat')
    bidir = Bidirectional(LSTM(128, return_sequences=True))(merged)
    bidir_merged = merge([merged, bidir], mode='concat')
    bidir_2 = Bidirectional(LSTM(128, return_sequences=True))(bidir_merged)
    time_dist_dense = TimeDistributed(Dense(tags, activation='softmax'))(bidir)
    model = Model(input=[sentence_input, POS_input, feature_input], output=time_dist_dense)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    return model

def define_model_baseline(vocab_size, tags):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size+1, output_dim=100, input_length=60, mask_zero=True))
    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(TimeDistributed(Dense(tags, activation='softmax')))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    return model

def train_model(model, x_train, y_train, x_val, y_val, filename):
    model.fit(x_train,y_train, nb_epoch=7,  batch_size=40, validation_data=(x_val, y_val))
    model.save_weights(filename+".h5") 
    with open(filename+".json", "w") as f:
        f.write(model.to_json())

def load_model(filename):
    with open(filename+".json") as f:
        model = model_from_json(f.read())
    model.load_weights(filename+".h5")
    return model

def make_predictions(model, x_test, y_test, inverted_indices):
    predictions = np.argmax(model.predict(x_test), axis=2)
    y_pred = []
    for k, sentence in enumerate(predictions):
        sentence_list = []
        length = len(y_test[k])
        for word in sentence[-length:]: 
            sentence_list.append(inverted_indices[word])
        print(len(sentence_list), len(y_test[k]))
        assert len(sentence_list) == len(y_test[k])
        y_pred.append(sentence_list)
    return y_pred

def main():
    model_filename = "BIO_baseline"
    #tag_indices_filename = "tag_indices.json"
    tag_indices_filename = "tag_indices_merged.json"
    merge_type = "BIO" # BIO, none, supertype
    #w2index, embeddings = load_embedding_matrix()
    w2index = load_indices('token_indices.json')
    tag_indices = load_indices(tag_indices_filename)
    inverted_indices = {v: k for k, v in tag_indices.items()}

    x_train, y_train, POS_train, ft_train ,_,  vocab_size = get_data('named_ent_train.txt', w2index, tag_indices, merge=merge_type)
    x_val, y_val, POS_val, ft_val, _, _ = get_data('named_ent_dtest.txt', w2index, tag_indices, merge=merge_type)
    x_test, y_test, POS_test, ft_test, test_text, _ = get_data('named_ent_etest.txt', w2index, tag_indices, validation=True, merge=merge_type)

    #model = define_model_concat(vocab_size, len(tag_indices), embeddings, POS_train, ft_train)
    #train_model(model, [x_train, POS_train, ft_train], y_train, x_val, y_val, model_filename)

    model = define_model_baseline(vocab_size, len(tag_indices))
    train_model(model, x_train, y_train, x_val, y_val, model_filename)    
    y_pred = make_predictions(model, x_test, y_test, inverted_indices)
    evaluations = global_eval(y_pred, y_test)
    output_evaluation(*evaluations, model_name=model_filename)
    random_sample("sentences_50", test_text, y_pred, y_test, 50)

if __name__ == '__main__':
    main()
