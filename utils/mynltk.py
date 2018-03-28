# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 18:04:22 2018

@author: zrssch
"""

import nltk
import tqdm


def tokenize_sentences(sentences, words_dict, words_cnt):
    tokenized_sentences = []
    for sentence in tqdm.tqdm(sentences):
        if hasattr(sentence, "decode"):
            sentence = sentence.decode("utf-8")
        tokens = nltk.tokenize.word_tokenize(sentence)
        result = []
        for word in tokens:
            word = word.lower()
            if word not in words_dict:
                words_dict[word] = len(words_dict)
                words_cnt[word] = 0
            word_index = words_dict[word]
            words_cnt[word] += 1
            result.append(word_index)
        tokenized_sentences.append(result)
    return tokenized_sentences, words_dict, words_cnt
