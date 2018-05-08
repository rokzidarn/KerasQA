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

def parse_file(directory, file):
    root = Et.parse(os.path.join(directory, file)).getroot()
    instances = []  # list of instances/stories
    questions = []  # 1 question per instance
    answers = []  # true answer per each question, ignore false

    # transforming text data to arrays
    for instance in root:  # instance/story
        for question in instance[1]:  # multiple questions
            if question[0].attrib['correct'] == 'True':  # 2 possible answers; true then false
                a = question[0].attrib['text']
                if len(nltk.word_tokenize(a)) == 1:  # 1 word answer
                    instances.append(instance[0].text)
                    questions.append(question.attrib['text'])
                    answers.append(a)
            else:  # 2 possible answers; false then true
                a = question[1].attrib['text']
                if len(nltk.word_tokenize(a)) == 1:
                    instances.append(instance[0].text)
                    questions.append(question.attrib['text'])
                    answers.append(a)

    return instances, questions, answers

def build_vocababulary(train_data, test_data):
    counter = collections.Counter()
    for stories, questions, answers in [train_data, test_data]:
        for story in stories:
            for word in nltk.word_tokenize(story):
                #print(word)
                counter[word.lower()] += 1
        for question in questions:
            for word in nltk.word_tokenize(question):
                counter[word.lower()] += 1
        for answer in answers:
            for word in nltk.word_tokenize(answer):
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

def data_encoding(max_len_instance, max_len_question, vocabulary_size):
    # inputs
    instance_input = Input(shape=(max_len_instance,))
    question_input = Input(shape=(max_len_question,))

    # story encoder memory
    instance_encoder = Embedding(input_dim=vocabulary_size, output_dim=96, input_length=max_len_instance)(instance_input)
    instance_encoder = Dropout(0.2)(instance_encoder)

    # question encoder
    question_encoder = Embedding(input_dim=vocabulary_size, output_dim=96, input_length=max_len_question)(question_input)
    question_encoder = Dropout(0.2)(question_encoder)

    # match between story and question
    match = dot([instance_encoder, question_encoder], axes=[2, 2])

    # encode story into vector space of question
    instance_encoder_c = Embedding(input_dim=vocabulary_size, output_dim=max_len_question, input_length=max_len_instance)(instance_input)
    instance_encoder_c = Dropout(0.2)(instance_encoder_c)

    # combine match and story vectors
    response = add([match, instance_encoder_c])
    response = Permute((2, 1))(response)

    return (instance_input, question_input, question_encoder, response)

def plot_acc(history_dict, epochs):
    acc = history_dict['acc']
    val_acc = history_dict['val_acc']

    plt.plot(epochs, acc, 'r', label='Training acc')
    plt.plot(epochs, val_acc, 'g', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

# MAIN

# file locations
data_dir = 'Data'
train_file = 'train-data.xml'  # combined train and dev samples
test_file = 'test-data.xml'
small_file = 'small-data.xml'

# getting data
train_data = parse_file(data_dir, train_file)
test_data = parse_file(data_dir, test_file)

# text stats
max_len_instance = len(str(max(train_data[0], key=len)).split())
max_len_question = len(str(max(train_data[1], key=len)).split())
max_len_answer = len(str(max(train_data[2], key=len)).split())
print('train data (I,Q,A): ', len(train_data[0]), len(train_data[1]), len(train_data[2]))
print('train data max lengths (I,Q,A):', max_len_instance, max_len_question, max_len_answer)

# building vocabulary
word2idx = build_vocababulary(train_data, test_data)
vocabulary_size = len(word2idx)

print('train + test distinct words: ', vocabulary_size)
#print(word2idx)

# vectorizing data
Xstrain, Xqtrain, Ytrain = vectorize(train_data, word2idx, max_len_instance, max_len_question)
Xstest, Xqtest, Ytest = vectorize(test_data, word2idx, max_len_instance, max_len_question)

#print(Xstrain[0])
#print(Xqtrain[0])
#print(Ytrain[0])

# encoding
(instance_input, question_input, question_encoder, response) = data_encoding(max_len_instance, max_len_question, vocabulary_size)

# creating network
answer = concatenate([response, question_encoder], axis=-1)
answer = LSTM(48)(answer)
answer = Dropout(0.2)(answer)
answer = Dense(vocabulary_size)(answer)
output = Activation("softmax")(answer)
# output = Activation("sigmoid")(answer)
model = Model(inputs=[instance_input, question_input], outputs=output)
model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
# model.compile(optimizer="sgd","adam", loss="binary_crossentropy", metrics=["mae"])

# training
history = model.fit([Xstrain, Xqtrain], [Ytrain], batch_size=32, epochs=64, validation_data=([Xstest, Xqtest], [Ytest]))

history_dict = history.history  # data during training, history_dict.keys()
print("validaton acc: ", round(max(history_dict['val_acc']), 3))
epochs = range(1, 64 + 1)

plot_acc(history_dict, epochs)

# TODO: check vectorize method!!!
# TODO: rewrite build_vocabulary() function

# TODO: different encoding structure
# TODO: transform multiword answers to oneword by excluding unnecessary words with nltk while parsing
# TODO: predict by saving true and false answers of train data and use argmax on true and false anwsers use max of them