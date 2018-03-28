# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 17:14:48 2018

@author: zrssch
"""
from utils.mynltk import tokenize_sentences
from utils.embedding import read_embedding_list, clear_embedding_list, convert_tokens_to_ids
from utils.model import model_fn

import argparse
import numpy as np
import os
import pandas as pd
import tensorflow as tf

UNKNOWN_WORD = "_UNK_"
END_WORD = "_END_"
NAN_WORD = "_NAN_"

CLASSES = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

def _train_model(params, batch_size, train_x, train_y, val_x, val_y):
    best_loss = -1
    best_weights = None
    best_epoch = 0

    current_epoch = 0
    model = tf.contrib.learn.Estimator(model_fn=model_fn, params=params)
    while True:           
        print("begin fit...")
        model.fit(x=train_x, y=train_y, steps=1000, batch_size=batch_size)
        current_epoch += 1
        print("begin evaluate...")
        ev = model.evaluate(x=val_x,y=val_y,steps=100, batch_size=batch_size)
        loss = ev["loss"]
        print("Epoch {0} loss {1} best_loss {2}".format(current_epoch, loss, best_loss))
        if loss < best_loss or best_loss == -1:
            best_loss = loss
            best_epoch = current_epoch
            best_weights = model.get_params()
        else:
            if current_epoch - best_epoch == 3:
                break
    
    model.set_params(**best_weights)
    return model


def train(X, y, batch_size, params, result_path, X_test):
    l = X.shape[0]
    ind = range(l)
    np.random.shuffle(ind)
    train_x = X[ind[:int(0.9*l)]]
    train_y = y[ind[:int(0.9*l)]]
    val_x = X[ind[int(0.9*l):]]
    val_y = y[ind[int(0.9*l):]]
    print("Starting to train models...")
    model = _train_model(params, batch_size, train_x, train_y, val_x, val_y)
    print("Predicting results...")
    model_path = os.path.join(result_path, "model_weights.npy")
    np.save(model_path, model.get_params())
    test_predicts_path = os.path.join(result_path, "test_predicts.npy")
    test_predicts = model.predict(x=X_test, batch_size=batch_size, as_iterable=False)
    np.save(test_predicts_path, test_predicts)
    return test_predicts
    
    
def main():
    parser = argparse.ArgumentParser(
        description="Recurrent neural network for identifying and classifying toxic online comments")

    parser.add_argument("train_file_path")
    parser.add_argument("test_file_path")
    parser.add_argument("embedding_path")
    parser.add_argument("--result-path", default="toxic_results")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--sentences-length", type=int, default=500)
    parser.add_argument("--recurrent-units", type=int, default=64)
    parser.add_argument("--dropout-rate", type=float, default=0.3)
    parser.add_argument("--dense-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    args = parser.parse_args()

    print("read data from csv and get list...")
    train_data = pd.read_csv(args.train_file_path)
    test_data = pd.read_csv(args.test_file_path)
    list_sentences_train = train_data["comment_text"].fillna(NAN_WORD).values
    list_sentences_test = test_data["comment_text"].fillna(NAN_WORD).values
    y_train = train_data[CLASSES].values
    
    print("Tokenizing sentences in train set...")
    # tokenized_sentences_train is a double list, combines with indexes. and words_dict is dictionsary 
    tokenized_sentences_train_ori, words_dict_ori, words_cnt = tokenize_sentences(list_sentences_train, {}, {})

    print("Tokenizing sentences in test set...")
    tokenized_sentences_test_ori, words_dict_ori, words_cnt = tokenize_sentences(list_sentences_test, words_dict_ori, words_cnt)
    
    words_dict_ori[UNKNOWN_WORD] = len(words_dict_ori)
    unkind = len(words_dict_ori)
    
    id_to_word = dict((id, word) for word, id in words_dict_ori.items())
    tokenized_sentences_train = []
    words_dict = {}
    for sentence in tokenized_sentences_train_ori:
        temp = []
        for i in sentence:
            if id_to_word[i] in words_dict:
                temp.append(i)
            else:
                if words_cnt[id_to_word[i]]>=3:
                    temp.append(i)
                    words_dict[id_to_word[i]] = i
        tokenized_sentences_train.append(temp)
    tokenized_sentences_test = []
    for sentence in tokenized_sentences_test_ori:
        temp = []
        for i in sentence:
            if id_to_word[i] in words_dict:
                temp.append(i)
            else:
                if words_cnt[id_to_word[i]]>= 3:
                    temp.append(i)
                    words_dict[id_to_word[i]] = i
        tokenized_sentences_test.append(temp)

    print("Loading embeddings...")
    embedding_list, embedding_word_dict = read_embedding_list(args.embedding_path)
    embedding_size = len(embedding_list[0])

    # take words and their embeddings from words_dict which has embeddings.    
    print("Preparing data...")
    embedding_list, embedding_word_dict = clear_embedding_list(embedding_list, embedding_word_dict, words_dict)
    
    # add unknown words flag and end flag
    embedding_word_dict[UNKNOWN_WORD] = len(embedding_word_dict)
    embedding_list.append([0.] * embedding_size)
    embedding_word_dict[END_WORD] = len(embedding_word_dict)
    embedding_list.append([-1.] * embedding_size)
    
    embedding_matrix = np.array(embedding_list)
    
    
    id_to_word = dict((id, word) for word, id in words_dict.items())
    
    # get the words id of train, the id inccordance with embedding matrix
    train_list_of_token_ids = convert_tokens_to_ids(
        tokenized_sentences_train,
        id_to_word,
        embedding_word_dict,
        args.sentences_length)
    test_list_of_token_ids = convert_tokens_to_ids(
        tokenized_sentences_test,
        id_to_word,
        embedding_word_dict,
        args.sentences_length)
    X_train = np.array(train_list_of_token_ids)
    X_test = np.array(test_list_of_token_ids)
    
    params = {"embedding_matrix": embedding_matrix, 
              "sentences_length": args.sentences_length,
              "recurrent_units": args.recurrent_units,
              "dropout_rate": args.dropout_rate,
              "dense_size": args.dense_size,
              "learning_rate": args.learning_rate}             
    
    if not os.path.exists(args.result_path):
        os.mkdir(args.result_path)
    test_predicts = train(X_train, y_train, args.batch_size, params, args.result_path, X_test)
    
    test_ids = test_data["id"].values
    test_ids = test_ids.reshape((len(test_ids), 1))

    test_predicts = pd.DataFrame(data=test_predicts, columns=CLASSES)
    test_predicts["id"] = test_ids
    test_predicts = test_predicts[["id"] + CLASSES]
    submit_path = os.path.join(args.result_path, "submit.csv")
    test_predicts.to_csv(submit_path, index=False)
    
    
if __name__ == "__main__":
    main()
    
