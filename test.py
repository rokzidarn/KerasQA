import os
import nltk
from xml.etree import ElementTree as Et
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from keras.layers import Dropout
from keras import layers
from keras import Input
from keras.models import Model
import matplotlib.pyplot as plt
import itertools

def parse_file(directory, file):
    root = Et.parse(os.path.join(directory, file)).getroot()
    instances = []  # list of instances/stories
    questions = []  # 1 question per instance
    answers = []  # true answer per each question, ignore false
    text = ''
    text_answers = ''

    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

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
                    text = text + ' ' + instance[0].text + ' ' + question.attrib['text'] + ' ' + answer_filtered[0]
                    text_answers = text_answers + ' ' + answer_filtered[0]
            else:  # 2 possible answers; false then true
                a = question[1].attrib['text']
                answer_filtered = tokenizer.tokenize(a)
                if len(answer_filtered) == 1:
                    instances.append(instance[0].text)
                    questions.append(question.attrib['text'])
                    answers.append(answer_filtered[0])
                    text = text + ' ' + instance[0].text + ' ' + question.attrib['text'] + ' ' + answer_filtered[0]
                    text_answers = text_answers + ' ' + answer_filtered[0]

    return instances, questions, answers, text.lower(), text_answers.lower()

def build_vocababulary(text, text_answers):
    words = nltk.word_tokenize(text)
    fdist = nltk.FreqDist(words)
    word2idx = {w: (i + 1) for i, (w, _) in enumerate(fdist.most_common())}
    word2idx["PAD"] = 0

    words = nltk.word_tokenize(text_answers)
    fdist_answers = nltk.FreqDist(words)
    word2idx_answers = {w: (i + 1) for i, (w, _) in enumerate(fdist_answers.most_common())}
    word2idx_answers["PAD"] = 0

    #print('Baseline: ', round(fdist_answers.most_common(1)[0][1] / len(word2idx_answers), 3))

    return word2idx, word2idx_answers

def vectorize(data, word2idx, word2idx_answers, story_maxlen, question_maxlen):
    Xi = []
    Xq = []
    Y = []
    instances, questions, answers, _, _ = data

    for instance, question, answer in zip(instances, questions, answers):
        xi = [word2idx[w.lower()] if w.lower() in word2idx else 0 for w in nltk.word_tokenize(instance)]
        xq = [word2idx[w.lower()] if w.lower() in word2idx else 0 for w in nltk.word_tokenize(question)]
        Xi.append(xi)
        Xq.append(xq)
        Y.append(word2idx_answers[answer.lower()] if answer.lower() in word2idx_answers else 0)

    return pad_sequences(Xi, maxlen=story_maxlen), pad_sequences(Xq, maxlen=question_maxlen), np_utils.to_categorical(Y, num_classes=len(word2idx_answers))

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
word2idx_all, word2idx_answers = build_vocababulary(train_data[3]+' '+test_data[3], train_data[4]+' '+test_data[4])
word2idx = dict(itertools.islice(word2idx_all.items(), 1500))
vocabulary_size = len(word2idx)
vocabulary_size_answers = len(word2idx_answers)

#print('Answers word2IDx: ', word2idx_answers)

# vectorizing data
Xitrain, Xqtrain, Ytrain = vectorize(train_data, word2idx, word2idx_answers, max_len_instance, max_len_question)
Xitest, Xqtest, Ytest = vectorize(test_data, word2idx, word2idx_answers, max_len_instance, max_len_question)

# params
epochs = 24
dropout_rate = 0.2

# model
text_input = Input(shape=(max_len_instance,))
embedded_text = layers.Embedding(64, vocabulary_size)(text_input)
encoded_text = layers.LSTM(32)(embedded_text)
encoded_text = Dropout(dropout_rate)(encoded_text)

question_input = Input(shape=(max_len_question,))
embedded_question = layers.Embedding(32, vocabulary_size)(question_input)
encoded_question = layers.LSTM(16)(embedded_question)
encoded_question = Dropout(dropout_rate)(encoded_question)

concatenated = layers.concatenate([encoded_text, encoded_question], axis=-1)
answer = layers.Dense(vocabulary_size_answers, activation='softmax')(concatenated)
answer = Dropout(dropout_rate)(answer)

model = Model([text_input, question_input], answer)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

# training
history = model.fit([Xitrain, Xqtrain], [Ytrain], batch_size=64, epochs=epochs, validation_data=([Xitest, Xqtest], [Ytest]))

history_dict = history.history  # data during training, history_dict.keys()
print("Max validaton acc: ", round(max(history_dict['val_acc']), 3))
gprah_epochs = range(1, epochs + 1)

plot_acc(history_dict, gprah_epochs)
