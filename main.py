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

def parse_file(directory, file):
    root = Et.parse(os.path.join(directory, file)).getroot()
    instances = []  # list of instances/stories
    questions = []  # 1 question per instance
    answers = []  # true answer per each question, ignore false

    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    stopwords = nltk.corpus.stopwords.words('english')

    # transforming text data to arrays
    for instance in root:  # instance/story
        for question in instance[1]:  # multiple questions
            if question[0].attrib['correct'] == 'True':  # 2 possible answers; true then false
                a = question[0].attrib['text']
                answer_filtered = tokenizer.tokenize(a)
                if len(answer_filtered) == 1:
                    instances.append(instance[0].text)
                    questions.append(question.attrib['text'])
                    answers.append(answer_filtered[0])
            else:  # 2 possible answers; false then true
                a = question[1].attrib['text']
                answer_filtered = tokenizer.tokenize(a)
                if len(nltk.word_tokenize(a)) == 1:
                    instances.append(instance[0].text)
                    questions.append(question.attrib['text'])
                    answers.append(answer_filtered[0])

    return instances, questions, answers

def build_vocababulary(train_data, test_data):
    counter = collections.Counter()
    counter_answer = collections.Counter()
    i = 0
    for stories, questions, answers in [train_data, test_data]:
        for story in stories:
            for word in nltk.word_tokenize(story):
                counter[word.lower()] += 1
        for question in questions:
            for word in nltk.word_tokenize(question):
                counter[word.lower()] += 1
        for answer in answers:
            i += 1
            for word in nltk.word_tokenize(answer):
                counter[word.lower()] += 1
                counter_answer[word.lower()] += 1

    word2idx = {w: (i+1) for i, (w, _) in enumerate(counter.most_common())}
    word2idx["PAD"] = 0
    # idx2word = {v: k for k, v in word2idx.items()}

    word2idx_answer = {w: (i + 1) for i, (w, _) in enumerate(counter_answer.most_common())}
    word2idx_answer["PAD"] = 0

    mca = counter_answer.most_common()[0]
    print('Most common answer: ', mca)
    print('Baseline: ', round(mca[1] / (len(train_data[2]) + len(test_data[2])), 3))

    return word2idx, word2idx_answer

def vectorize(data, word2idx, word2idx_answer, story_maxlen, question_maxlen):
    Xi = []
    Xq = []
    Y = []
    instances, questions, answers = data

    for instance, question, answer in zip(instances, questions, answers):
        xi = [word2idx[w.lower()] for w in nltk.word_tokenize(instance)]
        xq = [word2idx[w.lower()] for w in nltk.word_tokenize(question)]
        Xi.append(xi)
        Xq.append(xq)
        Y.append(word2idx[answer.lower()])

    return pad_sequences(Xi, maxlen=story_maxlen), pad_sequences(Xq, maxlen=question_maxlen), np_utils.to_categorical(Y, num_classes=len(word2idx))

def data_encoding(max_len_instance, max_len_question, vocabulary_size, embedding_size, dropout):
    # inputs
    instance_input = Input(shape=(max_len_instance,))
    question_input = Input(shape=(max_len_question,))

    # story encoder memory
    instance_encoder = Embedding(input_dim=vocabulary_size, output_dim=embedding_size, input_length=max_len_instance)(instance_input)
    instance_encoder = Dropout(dropout)(instance_encoder)

    # question encoder
    question_encoder = Embedding(input_dim=vocabulary_size, output_dim=embedding_size, input_length=max_len_question)(question_input)
    question_encoder = Dropout(dropout)(question_encoder)

    # match between story and question
    match = dot([instance_encoder, question_encoder], axes=[2, 2])

    # encode story into vector space of question
    instance_encoder_c = Embedding(input_dim=vocabulary_size, output_dim=max_len_question, input_length=max_len_instance)(instance_input)
    instance_encoder_c = Dropout(dropout)(instance_encoder_c)

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
max_len_instance = len(nltk.word_tokenize((max(train_data[0], key=len))))
max_len_question = len(nltk.word_tokenize((max(train_data[1], key=len))))
max_len_answer = len(nltk.word_tokenize((max(train_data[2], key=len))))
print('Train data (I,Q,A): ', len(train_data[0]), len(train_data[1]), len(train_data[2]))
print('Train data max lengths (I,Q,A):', max_len_instance, max_len_question, max_len_answer)

# building vocabulary
word2idx, word2idx_answer = build_vocababulary(train_data, test_data)
vocabulary_size = len(word2idx)
vocabulary_size_answer = len(word2idx_answer)

print('Train + test distinct words: ', vocabulary_size)
#print(word2idx)

# vectorizing data
Xitrain, Xqtrain, Ytrain = vectorize(train_data, word2idx, word2idx_answer, max_len_instance, max_len_question)
Xitest, Xqtest, Ytest = vectorize(test_data, word2idx, word2idx_answer, max_len_instance, max_len_question)

# params
embedding_size = 128
dropout = 0.3
latent_size = 64
answer_dropout = 0.2
epochs = 32

# encoding
(instance_input, question_input, question_encoder, response) = data_encoding(max_len_instance, max_len_question,
                                                                             vocabulary_size, embedding_size, dropout)

# creating network
answer = concatenate([response, question_encoder], axis=-1)
answer = LSTM(latent_size)(answer)
answer = Dropout(answer_dropout)(answer)
answer = Dense(vocabulary_size)(answer)
output = Activation("softmax")(answer)
# output = Activation("sigmoid")(answer)
model = Model(inputs=[instance_input, question_input], outputs=output)
model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
# model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["mae"])

# training
history = model.fit([Xitrain, Xqtrain], [Ytrain], batch_size=32, epochs=32, validation_data=([Xitest, Xqtest], [Ytest]))

history_dict = history.history  # data during training, history_dict.keys()
print("Max validaton acc: ", round(max(history_dict['val_acc']), 3))
gprah_epochs = range(1, epochs + 1)

plot_acc(history_dict, gprah_epochs)

# TODO: different encoding structure
# TODO: predict by saving true and false answers of test data and use argmax on possible anwsers -> predict_proba(Y)
# TODO: SQUAD

# TODO: tokenize in parse_file() then ' '.join() then use vectorizer
"""
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+')
arr = tokenizer.tokenize(text)

vectorizer = CountVectorizer(lowercase=True, stop_words='english')
vectorizer.fit(data)
word2idx = vectorizer.vocabulary_
"""
