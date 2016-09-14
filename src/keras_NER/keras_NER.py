#!/usr/bin/env python3
import text2embed as t
from keras.models import Sequential, load_model, model_from_json
from keras.layers import Masking, Embedding, Bidirectional, Merge, TimeDistributed, Dense, LSTM
from keras.preprocessing.sequence import pad_sequences
from gensim.models.word2vec import Word2Vec
import numpy as np
from collections import Counter
import json
import random


def global_eval(ypred, ytrue):
    """
    Measures micro averaged precision, recall, f1 and per-tag precision, recall, f1
    returns precision, recall, f1 (floats), per_tag_prec, per_tag_rec, per_tag_f1 (dictionaries)
    """
    merged_ypred = [item for sublist in ypred for item in sublist]
    merged_ytrue = [item for sublist in ytrue for item in sublist]
    tags = set(merged_ytrue)
    tp = Counter()
    fp = Counter()
    fn = Counter()
    tag_count_pr = Counter()
    tag_count_tr = Counter()
    for yp, yt in zip(merged_ypred, merged_ytrue):
        tag_count_tr[yt] += 1
        tag_count_pr[yp] += 1
        if yp == yt:
            tp[yt] += 1
        else:
            fn[yt] += 1
            fp[yp] += 1
    total_tp = 0
    total_fn = 0
    total_fp = 0
    for tag in tags:
        total_tp += tp[tag]
        total_fn += fn[tag]
        total_fp += fp[tag]
    #micro measure
    precision = total_tp/(total_tp + total_fp)
    recall = total_tp/(total_tp + total_fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1, tp, fn, fp, tag_count_pr, tag_count_tr, tags

def output_evaluation(precision, recall, f1, tp,fn,fp, count_pr, count_tr, tags,model_name):
    with open(model_name + '.log', 'w') as f:
        f.write('precision(micro): {}\n'.format(precision))
        f.write('recall(micro): {}\n'.format(recall))
        f.write('F1(micro): {}\n'.format(f1))
        f.write('======per-tag-stats=====\n')
        for tag in tags:
            if (tp[tag] + fp[tag]>0):
                precision = tp[tag] / (tp[tag] + fp[tag])
            else:
                precision = 0.
            if (tp[tag] + fn[tag]>0):
                recall = tp[tag] / (tp[tag] + fn[tag])
            else:
                recall = 0.
            if (precision + recall) > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0.
            f.write("{}:\t\t".format(tag))
            f.write("precision: {:.3}\t".format(precision))
            f.write("recall: {:.3}\t".format(recall))
            f.write("f1: {:.3}\t".format(f1))
            f.write("predicted: {}\t".format(count_pr[tag]))
            f.write("dataset: {}\t".format(count_tr[tag]))
            f.write("\n")
        f.close()


def load_embedding_matrix(filename='testmodel-139m_p3__.bin'):
    w = Word2Vec.load_word2vec_format(filename)
    print("creating word2index")
    word2index = {x : w.vocab[x].index + 2 for x in w.vocab}
    print('done, sorting')
    sorted_list = sorted(word2index.items(), key=lambda x: x[1])
    print('done sorting, creating embedding matrix')
    embeddings = []
    embeddings.append(np.zeros(100)) # padding
    embeddings.append([np.random.normal() for x in range(100)]) #OOV 
    embeddings.extend(w[x[0]] for x in sorted_list)
    embeddings_np = np.array(embeddings)
    print(embeddings_np.shape)
    return word2index, embeddings_np


def get_data(filename, indices, tag_indices, validation=False, merge="none"):
    raw_data = t.load_dataset(filename)
    indices = load_indices('token_indices.json')
    vocab_size = len(indices)
    tag_indices = load_indices(tag_indices)
    X_train = []
    Y_train = []
    text = []
    for line in raw_data:
        if len(line) == 0:
            continue
        tokens, tags = t.get_labels_tags(t.line_split(line), merge)
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

def vectorize_tags(tags, idx, merge=False):
    res = np.zeros((len(tags), len(idx)))
    for i,tag in enumerate(tags):
        res[i][idx[tag]] = 1
    return res

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
    predictions = np.argmax(model.predict(x_test), axis=2)
    y_pred = []
    for k, sentence in enumerate(predictions):
        sentence_list = []
        for i in range(min(len(y_test[k]), 60)):
            sentence_list.append(inverted_indices[sentence[i]])
        y_pred.append(sentence_list)
    return y_pred

def main():
    model_filename = "supertypes_v2"
    tag_indices_filename = "tag_indices.json"
    #tag_indices_filename = "tag_indices_merged.json"
    merge_type = "supertype" # BIO, none, supertype
    w2index, embeddings = load_embedding_matrix()
    print('loading embeddings done')
    x_train, y_train,_ , vocab_size = get_data('named_ent_train.txt', w2index, tag_indices_filename, merge=merge_type)
    x_val, y_val,_,  _ = get_data('named_ent_dtest.txt', w2index, tag_indices_filename, merge=merge_type)
    x_test, y_test, test_text, _ = get_data('named_ent_etest.txt', w2index, tag_indices_filename, validation=True, merge=merge_type)
    tag_indices = load_indices(tag_indices_filename)
    inverted_indices = {v: k for k, v in tag_indices.items()}
    model = define_model_w2v(vocab_size, len(tag_indices), embeddings)
    train_model(model, x_train, y_train, x_val, y_val, model_filename)
    #load_model(model, model_filename)
    y_pred = make_predictions(model, x_test, y_test, inverted_indices)
    evaluations = global_eval(y_pred, y_test)
    output_evaluation(*evaluations, model_name=model_filename)
    with open('sentences_40.txt', 'w') as f:
        for x in range(40):
            num = random.randint(0,int(len(test_text)/2))
            curr_sent = "sent:" + "\t".join(test_text[num])
            curr_pred = "pred:" + "\t".join(y_pred[num])
            curr_gold = "gold:" + "\t".join(y_test[num])
            f.write(curr_sent)
            f.write('\n')
            f.write(curr_pred)
            f.write('\n')
            f.write(curr_gold)
            f.write('\n')

if __name__ == '__main__':
    main()
