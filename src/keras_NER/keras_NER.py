#!/usr/bin/env python3
import src.common.NER_utils as t
from src.common.eval import global_eval, output_evaluation, random_sample
from keras.models import Sequential, load_model, model_from_json
from keras.layers import Masking, Embedding, Bidirectional, Merge, TimeDistributed, Dense, LSTM
from keras.preprocessing.sequence import pad_sequences
from gensim.models.word2vec import Word2Vec
import numpy as np
from collections import Counter
import json
import random

def load_embedding_matrix(filename='testmodel-139m_p3___.bin'):
    w = Word2Vec.load(filename)
    word2index = {x : w.vocab[x].index + 2 for x in w.vocab}
    sorted_list = sorted(word2index.items(), key=lambda x: x[1])
    embeddings = []
    embeddings.append(np.zeros(100)) # padding
    embeddings.append([np.random.normal() for x in range(100)]) #OOV 
    embeddings.extend(w[x[0]] for x in sorted_list)
    embeddings_np = np.array(embeddings)
    return word2index, embeddings_np

def get_data(filename, indices, tag_indices, validation=False, merge="none"):
    raw_data = t.load_dataset(filename)
    vocab_size = len(indices)
    X_train = []
    Y_train = []
    text = []
    ft_params = ['POS_curr', 'POS.json']
    #ft = feature_extractor(ft_params)
    for line in raw_data:
        if len(line) == 0:
            continue
        tokens, tags = t.get_labels_tags(t.line_split(line), merge)
     #   features = ft.extract_features(tokens)
        text.append(tokens)
        tokens_vector= vectorize_sentence(tokens, indices)
        X_train.append(tokens_vector)
        if validation:
            Y_train.append(tags)
        else:
            Y_train.append(vectorize_tags(tags, tag_indices))
    x_train = pad_sequences(np.array(X_train), maxlen=60)
    if validation:
        y_train = np.array(Y_train)
    else:
        y_train = pad_sequences(np.array(Y_train), maxlen=60)
    return x_train, y_train, text, vocab_size

def vectorize_POS(features, POS_indices):
    result = [[0 for x in range(10)] for y in range(len(POS_list))]
    for feature in features:
        result[POS_indices[feature['POS[0]']] = 1
    return np.array(result)

def vectorize_tags(tags, idx, merge=False):
    res = np.zeros((len(tags), len(idx)))
    for i,tag in enumerate(tags):
        res[i][idx[tag]] = 1
    return res

def generate_features(dataset):
    ft = feature_extractor([""])
    for line in dataset:
        tokens, tags = t.get_labels_tags(t.line_split(line), "none")
        features_list.append(ft.extract_features(tokens, string_format=False))
    return la
def vectorize_sentence(tokens,word_idx):
    result = []
    for token in tokens:
        if token in word_idx:
            result.append(word_idx[token])
        else:
            result.append(1)
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

def define_model(vocab_size, tags):
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

def load_model(model, filename):
    with open(filename+".json") as f:
        model = model_from_json(f.read())
    model.load_weights(filename+".h5")

def make_predictions(model, x_test, y_test, inverted_indices):
    predictions = model.predict(x_test)

    #predictions = np.argmax(model.predict(x_test), axis=2)
    y_pred = []
    for k, sentence in enumerate(predictions):
        sentence_list = []
        for i in range(min(len(y_test[k]), 60)):
            sentence_list.append(sentence[i])
            #sentence_list.append(inverted_indices[sentence[i]])
        y_pred.append(sentence_list)
    return y_pred

def main():
    model_filename = "BIO"
    #tag_indices_filename = "tag_indices.json"
    tag_indices_filename = "tag_indices_merged.json"
    merge_type = "BIO" # BIO, none, supertype
    w2index, embeddings = load_embedding_matrix()
    tag_indices = load_indices(tag_indices_filename)
    inverted_indices = {v: k for k, v in tag_indices.items()}
    x_train, y_train,_ , vocab_size = get_data('named_ent_train.txt', w2index, tag_indices, merge=merge_type)
    x_val, y_val,_,  _ = get_data('named_ent_dtest.txt', w2index, tag_indices, merge=merge_type)
    x_test, y_test, test_text, _ = get_data('named_ent_etest.txt', w2index, tag_indices, validation=True, merge=merge_type)
    model = define_model_w2v(vocab_size, len(tag_indices), embeddings)
    train_model(model, x_train, y_train, x_val, y_val, model_filename)
    #load_model(model, model_filename)
    y_pred = make_predictions(model, x_test, y_test, inverted_indices)
    evaluations = global_eval(y_pred, y_test)
    output_evaluation(*evaluations, model_name=model_filename)
    random_sample("sentences_50", test_text, y_pred, y_test, 50)

if __name__ == '__main__':
    main()
