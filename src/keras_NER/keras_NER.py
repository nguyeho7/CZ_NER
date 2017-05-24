#!/usr/bin/env python3
import src.common.NER_utils as t
from src.common.eval import global_eval, output_evaluation, random_sample
from keras.models import Sequential, load_model, model_from_json, Model
from keras.layers.core import Reshape, Permute
from keras.layers import Input, Embedding, Bidirectional, Merge, TimeDistributed, Dense, LSTM,merge,Dropout, Convolution1D, Masking, GRU, Convolution2D, MaxPooling2D, Convolution1D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.sequence import pad_sequences
from gensim.models.word2vec import Word2Vec
from string import punctuation
import numpy as np
import json

pos_index = {"ADJ": 1, "ADP": 2, "ADV": 3, "AUX": 4, "CONJ": 5, "DET": 6, "INTJ": 7, "NOUN": 8, 
        "NUM": 9, "PART": 10, "PRON": 11, "PROPN": 12, "PUNCT": 13, "SCONJ": 14, "SYM": 15, "VERB":15, "X": 16, 'none' : 0}
feature_functs = ["is_upper", "name", "address", "last_name", "at", "contains_digit"]

def load_embedding_matrix(filename='testmodel-139m_p3___.bin', word_list_filename='none'):
    w = Word2Vec.load(filename)
    dim = w.layer1_size
    embeddings = []
    embeddings.append(np.zeros(dim)) # padding
    embeddings.append([np.random.normal() for x in range(dim)]) #OOV 
    not_found = 0
    if word_list_filename != 'none':
        word2index = {}
        raw_data = t.load_dataset(word_list_filename)
        word_set = set()
        for line in raw_data:
            tokens, tags = t.get_labels_tags(t.line_split(line), "BIO")
            for token in tokens:
                word_set.add(token.lower())
        i = 2
        for token in word_set:
            if token in w:
                embeddings.append(w[token])
                word2index.update({token: i})
                i += 1
            else:
                not_found += 1
    else:
        word2index = {x : w.vocab[x].index + 2 for x in w.vocab}
        sorted_list = sorted(word2index.items(), key=lambda x: x[1])
        embeddings.extend(w[x[0]] for x in sorted_list)
    embeddings_np = np.array(embeddings)
    return word2index, embeddings_np

def load_embedding_subset(subset_filename, tags_filename):
    embeddings = np.load(subset_filename)
    indices = load_indices(tags_filename)
    return indices, embeddings

def get_data(filename, indices, tag_indices, validation=False, merge="none"):
    vocab_size = len(indices)
    X_train = []
    Y_train = []
    ft_params = ['label','POS_curr_json', 'POS_final.json', "is_capitalised", 'addr_gzttr', 'adresy.txt',
            'name_gzttr','czech_names', 'lname_gzttr', "czech_last_names",'contains_at',
            'contains_digit']
    feature_list = []
    POS = []
    y_gold, sentences_features, sentences_text = t.load_transform_dataset(filename, ft_params, merge, str_format = False)
    for sentence in sentences_text:
        X_train.append(vectorize_sentence(sentence, indices))
    for tags in y_gold:
        if validation:
            Y_train.append(tags[:60])
        else:
            Y_train.append(vectorize_tags(tags, tag_indices))
    for features in sentences_features:
        POS.append(vectorize_POS(features))
        feature_list.append(vectorize_features(features))  
    x_train = pad_sequences(np.array(X_train), maxlen=60)
    if validation:
        y_train = np.array(Y_train)
    else:
        y_train = pad_sequences(np.array(Y_train), maxlen=60)
    POS_np = pad_sequences(np.array(POS), maxlen=60)
    feature_list_np = pad_sequences(np.array(feature_list), maxlen=60)
    return x_train, y_train, POS_np, feature_list_np, sentences_text, vocab_size

def get_data_conll(filename, indices, tag_indices, pos_indices, validation=False):
    vocab_size = len(indices)
    X_train = []
    Y_train = []
    feature_list = []
    POS = []
    gtags, features, text = t.transform_dataset_conll(filename)
    for sentence in text:
        X_train.append(vectorize_sentence(sentence, indices))
    for fts in features:
        POS.append(vectorize_POS_conll(fts, pos_indices))
    for tags in gtags:
        if validation:
            Y_train.append(tags[:60])
        else:
            Y_train.append(vectorize_tags(tags, tag_indices))
    POS_np = pad_sequences(np.array(POS), maxlen=60)
    x_train = pad_sequences(np.array(X_train), maxlen=60)
    if validation:
        y_train = Y_train
    else:
        y_train = pad_sequences(np.array(Y_train), maxlen=60)
    return x_train, y_train, POS_np, text, vocab_size
    

def vectorize_POS_conll(fts, pos_indices):
    result = [[0 for x in range(len(pos_indices)+1)] for y in range(len(fts))]
    for i, ft in enumerate(fts):
        result[i][pos_indices[ft['pos[0]']]] = 1
    return np.array(result)

def vectorize_POS(features):
    result = [[0 for x in range(len(pos_index))] for y in range(len(features))]
    for i, feature in enumerate(features):
        result[i][pos_index[feature['POS[0]']]] = 1
    return np.array(result)

def text_to_char(text, char_index):
    result = []
    for sentence in text:
        curr_sentence = [[0 for x in range(60)]]
        for word in sentence:
            vector = [char_index[c] for c in word]
            curr_sentence.append(vector)
        curr_sentence = pad_sequences(curr_sentence, maxlen=15)
        result.append(curr_sentence)
    result = pad_sequences(result, maxlen=60)
    print(result.shape)
    result = np.reshape(result, (len(text),60*15))
    print(result.shape)
    return result

def vectorize_features(features):
    #for every feature apart from POS tags
    # list of features:
    # is_capitalised, addr_gzttr, name_gzttr, contains_at, contains_digit
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
        tok = token.lower()
        if tok in word_idx:
            result.append(word_idx[tok])
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
    model.add(Bidirectional(GRU(256, return_sequences=True)))
    model.add(BatchNormalization())
    model.add(TimeDistributed(Dense(tags, activation='softmax')))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    return model

def define_model_w2v_2layer(vocab_size, tags, embeddings):
    model = Sequential()
    model.add(Embedding(input_dim = embeddings.shape[0], output_dim=embeddings.shape[1],
        weights=[embeddings], input_length=60, mask_zero=True))
    model.add(Bidirectional(GRU(256, return_sequences=True)))
    model.add(BatchNormalization())
    model.add(Bidirectional(GRU(256, return_sequences=True)))
    model.add(BatchNormalization())
    model.add(TimeDistributed(Dense(tags, activation='softmax')))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    return model

def define_model_baseline(vocab_size, tags):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size+2, output_dim=300, input_length=60, mask_zero=True))
    model.add(Bidirectional(GRU(256, return_sequences=True)))
    model.add(BatchNormalization())
    model.add(TimeDistributed(Dense(tags, activation='softmax')))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    return model

def define_model_conll(vocab_size, tags, pos_vectors):
    sentence_input = Input(shape=(60,), dtype='int32')
    embedding_layer = Embedding(input_dim=vocab_size+2, output_dim=100, input_length=60, mask_zero=True)(sentence_input)

    POS_input = Input(shape=(60, len(pos_vectors)+1))

    merged = merge([embedding_layer, POS_input], mode='concat')
    bidir = Bidirectional(GRU(256, return_sequences=True))(merged)
    bidir_bnorm = BatchNormalization()(bidir)
    time_dist_dense = TimeDistributed(Dense(tags, activation='softmax'))(bidir_bnorm)
    model = Model(input=[sentence_input,  POS_input], output=time_dist_dense)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    return model

def define_model_cnn(vocab_size, tags, embeddings, pos_vectors, feature_vectors, char_count):
    sentence_input = Input(shape=(60,), dtype='int32')
    embedding_layer = Embedding(input_dim = embeddings.shape[0], output_dim=embeddings.shape[1],
        weights=[embeddings], input_length=60, mask_zero=True)(sentence_input)

    POS_input = Input(shape=(60, len(pos_index)))
    feature_input = Input(shape=(60, len(feature_functs)))

    char_input = Input(shape=(60*15, ), dtype='int32')
    char_embedding = Embedding(input_dim = char_count, output_dim=32, input_length = 60*15)(char_input)
    reshape = Reshape((60, 15, 32))(char_embedding)
    permute = Permute((3,1,2))(reshape) 
    charcnn = Convolution2D(10, 1,2,border_mode='same')(permute)
    charcnn_bnorm = BatchNormalization()(charcnn)
    charcnn2 = Convolution2D(10, 1,2,border_mode='same')(charcnn_bnorm)
    charcnn_bnorm2 = BatchNormalization()(charcnn2)
    charcnn3 = Convolution2D(10, 1,2,border_mode='same')(charcnn_bnorm2)
    charcnn_bnorm3 = BatchNormalization()(charcnn3)
    permute2 = Permute((2,1,3))(charcnn_bnorm3)
    pooling = MaxPooling2D((1, 2))(permute2)
    print(pooling.shape)
    reshape2 = Reshape((60,16*10)) (pooling)

    merged = merge([embedding_layer, POS_input, feature_input, reshape2], mode='concat')
    bidir = Bidirectional(GRU(256, return_sequences=True))(merged)
    bidir_bnorm = BatchNormalization()(bidir)
    time_dist_dense = TimeDistributed(Dense(tags, activation='softmax'))(bidir_bnorm)
    model = Model(input=[sentence_input,  POS_input, feature_input, char_input], output=time_dist_dense)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    return model

def define_model_charrnn(vocab_size, tags, embeddings, pos_vectors, feature_vectors, char_count):
    sentence_input = Input(shape=(60,), dtype='int32')
    embedding_layer = Embedding(input_dim = embeddings.shape[0], output_dim=embeddings.shape[1],
        weights=[embeddings], input_length=60, mask_zero=True)(sentence_input)

    POS_input = Input(shape=(60, len(pos_index)))
    feature_input = Input(shape=(60, len(feature_functs)))

    char_input = Input(shape=(60*15, ), dtype='int32')
    char_embedding = Embedding(input_dim = char_count, output_dim=64, input_length = 60*15)(char_input)
    reshape = Reshape((60,15,64))(char_embedding)
    chrnn = TimeDistributed(GRU(64, return_sequences=False))(reshape)
    chrnn_bnorm = BatchNormalization()(chrnn)

    merged = merge([POS_input, feature_input, chrnn_bnorm], mode='concat')
    bidir = Bidirectional(GRU(256, return_sequences=True))(merged)
    bidir_bnorm = BatchNormalization()(bidir)
    time_dist_dense = TimeDistributed(Dense(tags, activation='softmax'))(bidir_bnorm)
    model = Model(input=[ POS_input, feature_input, char_input], output=time_dist_dense)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    return model
def define_model_baseline_2layer(vocab_size, tags):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size+2, output_dim=300, input_length=60, mask_zero=True))
    model.add(Bidirectional(LSTM(256, return_sequences=True)))
    model.add(Bidirectional(LSTM(256, return_sequences=True)))
    model.add(TimeDistributed(Dense(tags, activation='softmax')))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    return model

def define_model_POS(vocab_size, tags, embeddings, pos_vectors):
    sentence_input = Input(shape=(60,), dtype='int32')
    embedding_layer = Embedding(input_dim = embeddings.shape[0], output_dim=embeddings.shape[1],
        weights=[embeddings], input_length=60, mask_zero=True)(sentence_input)
    POS_input = Input(shape=(60, len(pos_index)))
    merged = merge([embedding_layer, POS_input], mode='concat')
    bidir = Bidirectional(GRU(256, return_sequences=True))(merged)
    bidir_bnorm = BatchNormalization()(bidir)
    time_dist_dense = TimeDistributed(Dense(tags, activation='softmax'))(bidir_bnorm)
    model = Model(input=[sentence_input, POS_input], output=time_dist_dense)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    return model

def define_model_concat(vocab_size, tags, embeddings,  pos_vectors, feature_vectors):
    sentence_input = Input(shape=(60,), dtype='int32')
    embedding_layer = Embedding(input_dim = embeddings.shape[0], output_dim=embeddings.shape[1],
        weights=[embeddings], input_length=60, mask_zero=True)(sentence_input)
    POS_input = Input(shape=(60, len(pos_index)))
    feature_input = Input(shape=(60, len(feature_functs)))
    merged = merge([embedding_layer, POS_input, feature_input], mode='concat')
    bidir = Bidirectional(GRU(256, return_sequences=True))(merged)
    bidir_bnorm = BatchNormalization()(bidir)
    time_dist_dense = TimeDistributed(Dense(tags, activation='softmax'))(bidir_bnorm)
    model = Model(input=[sentence_input, POS_input, feature_input], output=time_dist_dense)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    return model

def define_model_convolutional(vocab_size, tags, embeddings,  POS_vectors, feature_vectors):
    sentence_input = Input(shape=(60,), dtype='int32')
    embedding_layer = Embedding(input_dim = embeddings.shape[0], output_dim=embeddings.shape[1],
        weights=[embeddings], input_length=60)(sentence_input)
    POS_input = Input(shape=(60, len(pos_index)))
    feature_input = Input(shape=(60, len(feature_functs)))
    cnns = [BatchNormalization()(Convolution1D(filter_length=flt, nb_filter=20, activation="tanh",
        border_mode='same')(embedding_layer)) for flt in
            [2,3]]
    masked_embedding =Masking(mask_value=0.0)(embedding_layer) 
    cnns_merged = merge(cnns, mode='concat')
    merged = merge([masked_embedding, POS_input, feature_input, cnns_merged], mode='concat')
    bidir = Bidirectional(GRU(256, return_sequences=True))(merged)
    bidir_bnorm = BatchNormalization()(bidir)
    time_dist_dense = TimeDistributed(Dense(tags, activation='softmax'))(bidir_bnorm)
    model = Model(input=[sentence_input, POS_input, feature_input], output=time_dist_dense)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    return model

def define_model_baseline_concat(vocab_size, tags, POS_vectors, feature_vectors):
    sentence_input = Input(shape=(60,), dtype='int32')
    embedding_layer = Embedding(input_dim = vocab_size+2, output_dim=300, input_length=60, mask_zero=True)(sentence_input)
    POS_input = Input(shape=(60, len(pos_index)))
    feature_input = Input(shape=(60, len(feature_functs)))
    merged = merge([embedding_layer, POS_input, feature_input], mode='concat')
    bidir = Bidirectional(LSTM(128, return_sequences=True))(merged)
    time_dist_dense = TimeDistributed(Dense(tags, activation='softmax'))(bidir)
    model = Model(input=[sentence_input, POS_input, feature_input], output=time_dist_dense)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    return model

def define_model_2_layer(vocab_size, tags, embeddings,  POS_vectors, feature_vectors):
    sentence_input = Input(shape=(60,), dtype='int32')
    embedding_layer = Embedding(input_dim = embeddings.shape[0], output_dim=embeddings.shape[1],
        weights=[embeddings], input_length=60, mask_zero=True)(sentence_input)
    POS_input = Input(shape=(60, len(pos_index)))
    feature_input = Input(shape=(60, len(feature_functs)))
    merged = merge([embedding_layer, POS_input, feature_input], mode='concat')
    bidir = Bidirectional(LSTM(128, return_sequences=True))(merged)
    bidir_merged = merge([merged, bidir], mode='concat')
    bidir_2 = Bidirectional(LSTM(128, return_sequences=True))(bidir_merged)
    time_dist_dense = TimeDistributed(Dense(tags, activation='softmax'))(bidir_2)
    model = Model(input=[sentence_input, POS_input, feature_input], output=time_dist_dense)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    return model

def train_model(model, x_train, y_train, x_val, y_val, filename):
    model.fit(x_train,y_train, nb_epoch=25,  batch_size=512, validation_data=(x_val, y_val))
    model.save_weights(filename+".h5") 
    with open(filename+".json", "w") as f:
        f.write(model.to_json())

def load_model(filename):
    with open(filename+".json") as f:
        model = model_from_json(f.read())
    model.load_weights(filename+".h5")
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    return model

def make_predictions(model, x_test, y_test, inverted_indices):
    predictions = np.argmax(model.predict(x_test), axis=2)
    y_pred = []
    for k, sentence in enumerate(predictions):
        sentence_list = []
        length = len(y_test[k])
        for word in sentence[-length:]: 
            sentence_list.append(inverted_indices[word])
        assert len(sentence_list) == len(y_test[k])
        y_pred.append(sentence_list)
    return y_pred

def conll_test():
    tag_indices = load_indices("conll_tags.json")
    w2index = load_indices("conll_tokens.json")
    pos_indices = load_indices("conll_pos.json")
    model_filename = "conll_NN"
    x_train, y_train, POS_train, train_text, vocab_size = get_data_conll("eng.train", w2index, tag_indices,
            pos_indices)
    x_val, y_val, POS_val, val_text, _ = get_data_conll("eng.testa", w2index, tag_indices,
            pos_indices)
    x_test, y_test, POS_test,test_text, _ = get_data_conll("eng.testb", w2index, tag_indices,
            pos_indices, validation=True)

    inverted_indices = {v: k for k, v in tag_indices.items()}
    model = define_model_conll(vocab_size, len(tag_indices), pos_indices)
    train_model(model, [x_train, POS_train], y_train, [x_val, POS_val], y_val, model_filename)
    y_pred = make_predictions(model, [x_test, POS_test], y_test, inverted_indices)
    result_text = ""
    for sent, pred, label in zip(test_text, y_pred, y_test):
        for ft, tag, gold in zip(sent, pred, label):
            word = ft
            pos = 'pos[0]'
            result_text += word +" "+pos+" "+gold+" "+tag+"\n"
        result_text += "\n"
    with open(model_filename, "w") as f:
        f.write(result_text)


def main():
    conll_test()
    #tag_indices_filename = "tag_indices_merged.json"
    tag_indices_fieename = "tag_indices_bilou.json"
    #tag_indices_filename = "tag_indices.json"
    merge_type = "BILOU" # BIO, none, supertype, BILOU
    model_filename = "convolution"
    #w2index, embeddings = load_embedding_matrix("d300w5_10p_ft_skipgram", "named_ent.txt")
 #   np.save(open('d300w5_skipgram_subset.np', 'bw'), embeddings)
  #  json.dump(w2index, open('d300w5_skipgram_indices.json', 'w'))
    w2index, embeddings = load_embedding_subset("d300w5_skipgram_subset.np",
           "d300w5_skipgram_indices.json")
    #w2index = load_indices('token_indices.json')
    tag_indices = load_indices(tag_indices_filename)
    inverted_indices = {v: k for k, v in tag_indices.items()}
    char_indices = load_indices("character_index.json")

#    x_train, y_train, POS_train, ft_train ,train_text,  vocab_size = get_data('named_ent_train.txt', w2index,
 #           tag_indices,merge=merge_type)
  #  x_val, y_val, POS_val, ft_val, val_text, _ = get_data('named_ent_dtest.txt', w2index, tag_indices, merge=merge_type)
   # x_test, y_test, POS_test, ft_test, test_text, _ = get_data('named_ent_etest.txt', w2index, tag_indices, validation=True, merge=merge_type)
    #x_train_cnn = text_to_char(train_text, char_indices)
    #x_val_cnn = text_to_char(val_text, char_indices)
    #x_test_cnn = text_to_char(test_text, char_indices)
    #model = load_model(model_filename)
    #model = define_model_convolutional(vocab_size, len(tag_indices), embeddings, POS_train, ft_train)
    #model = define_model_concat(vocab_size, len(tag_indices), embeddings, POS_train, ft_train)
    #train_model(model, [x_train, POS_train, ft_train], y_train, [x_val, POS_val, ft_val], y_val, model_filename)
    #y_pred = make_predictions(model, [x_test, POS_test, ft_test], y_test, inverted_indices)

    #model = define_model_w2v_2layer(vocab_size, len(tag_indices), embeddings)
    #model = define_model_baseline(vocab_size, len(tag_indices))
    #train_model(model, x_train, y_train, x_val, y_val, model_filename)    
    #y_pred = make_predictions(model, x_test, y_test, inverted_indices)
    
    #model = define_model_POS(vocab_size, len(tag_indices), embeddings, POS_train)
    #train_model(model, [x_train, POS_train], y_train, [x_val, POS_val], y_val, model_filename)
    #y_pred = make_predictions(model, [x_test, POS_test], y_test, inverted_indices)
    #model = define_model_charrnn(vocab_size, len(tag_indices), embeddings, POS_train, ft_train, len(char_indices))
    #train_model(model, [POS_train, ft_train, x_train_cnn], y_train, [POS_val, ft_val, x_val_cnn], y_val, model_filename)
    #y_pred = make_predictions(model, [POS_test, ft_test, x_test_cnn], y_test, inverted_indices)

    evaluations = global_eval(y_pred, y_test)
    #train_pred = make_predictions(model, x_train, y_train, inverted_indices)
    test_result = [[sent,pred] for sent, pred in zip(test_text, y_pred)]
    output_evaluation(*evaluations, model_name=model_filename)
    json.dump(test_result, open(model_filename + "_textoutput.json", "w"))

if __name__ == '__main__':
    main()
