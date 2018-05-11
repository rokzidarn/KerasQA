import os
import collections
import nltk
from xml.etree import ElementTree as Et
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from keras.layers import Input
from keras.layers.core import Activation, Dense, Dropout, Permute
from keras.layers.embeddings import Embedding
from keras.layers.merge import add, concatenate, dot
from keras.layers.recurrent import LSTM
from keras.models import Model
import matplotlib.pyplot as plt
from keras.utils import to_categorical

def build_vocababulary(train_data):
    counter = collections.Counter()
    for story in train_data:
        for word in nltk.word_tokenize(story):
            #print(word)
            counter[word.lower()] += 1
    word2idx = {w: (i+1) for i, (w, _) in enumerate(counter.most_common())}
    word2idx["PAD"] = 0
    #idx2word = {v: k for k, v in word2idx.items()}

    return word2idx

def vectorize(data, word2idx, story_maxlen, question_maxlen):
    Xs = []
    Xq = []
    Y = []
    stories, questions, answers = data

    for story, question, answer in zip(stories, questions, answers):
        xs = [word2idx[w.lower()] for w in nltk.word_tokenize(story)]
        xq = [word2idx[w.lower()] for w in nltk.word_tokenize(question)]
        Xs.append(xs)
        Xq.append(xq)
        Y.append(word2idx[answer.lower()])

    return pad_sequences(Xs, maxlen=story_maxlen), pad_sequences(Xq, maxlen=question_maxlen), np_utils.to_categorical(Y, num_classes=len(word2idx))

text1 = 'Hello my name is, my name is Slim Shady.'
text_arr1 = nltk.word_tokenize(text1)
#print(text_arr1)
text2 = 'Who is Slim?'
text_arr2 = nltk.word_tokenize(text1)
text_arr = [text1, text2]
max_len_instance = len(nltk.word_tokenize((max(text_arr, key=len))))
#print(max_len_instance)

word2idx = build_vocababulary(text_arr)
vocabulary_size = len(word2idx)
#print(word2idx)

xs = [[word2idx[w.lower()] for w in nltk.word_tokenize(story)] for story in text_arr]
#print(xs)
#print(pad_sequences(xs, maxlen=max_len_instance))

a = "Hello, my name is Slim!"
v_size = 5
data = [[1, 1, 2, 3, 0, 2], [1,0,4]]
e = [to_categorical(d, v_size) for d in data]
print(e[0])
