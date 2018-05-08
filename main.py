import os
import collections
import nltk
import itertools
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
            # TODO: if question.attrib['type'] == 'text':
            if question[0].attrib['correct'] == 'True':  # 2 possible answers; true then false
                a = question[0].attrib['text']
                if len(nltk.word_tokenize(a)) == 1:
                    instances.append(instance[0].text)
                    questions.append(question.attrib['text'])
                    answers.append(a)
                    break
            else:  # 2 possible answers; false then true
                a = question[1].attrib['text']
                if len(nltk.word_tokenize(a)) == 1:
                    instances.append(instance[0].text)
                    questions.append(question.attrib['text'])
                    answers.append(a)
                    break

    return instances, questions, answers

def build_vocababulary(train_data, test_data):
    counter = collections.Counter()
    i = 0
    for stories, questions, answers in [train_data, test_data]:
        for story in stories:
            print(i)
            i += 1
            for sent in story:
                for word in nltk.word_tokenize(sent):
                    counter[word.lower()] += 1
            for question in questions:
                for word in nltk.word_tokenize(question):
                    counter[word.lower()] += 1
            for answer in answers:
                for word in nltk.word_tokenize(answer):
                    counter[word.lower()] += 1
    word2idx = {w: (i+1) for i, (w, _) in enumerate(counter.most_common())}
    word2idx["PAD"] = 0
    idx2word = {v: k for k, v in word2idx.items()}

    return word2idx, idx2word

def vectorize(data, word2idx, story_maxlen, question_maxlen):
    Xs = []
    Xq = []
    Y = []
    stories, questions, answers = data

    for story, question, answer in zip(stories, questions, answers):
        xs = [[word2idx[w.lower()] for w in nltk.word_tokenize(s)] for s in story]
        xs = list(itertools.chain.from_iterable(xs))
        xq = [word2idx[w.lower()] for w in nltk.word_tokenize(question)]
        Xs.append(xs)
        Xq.append(xq)
        Y.append(word2idx[answer.lower()])

    return pad_sequences(Xs, maxlen=story_maxlen), pad_sequences(Xq, maxlen=question_maxlen), np_utils.to_categorical(Y, num_classes=len(word2idx))

def plot_acc(history_dict, epochs):
    acc = history_dict['acc']
    val_acc = history_dict['val_acc']

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

# MAIN

# file locations
data_dir = 'Data'
train_file = 'train-data.xml'
dev_file = 'dev-data.xml'
test_file = 'test-data.xml'
small_train_file = 'small-train-data.xml'
small_test_file = 'small-test-data.xml'

# getting data
train_data = parse_file(data_dir, train_file)
test_data = parse_file(data_dir, test_file)

# TODO: check answers

# text stats
max_words = 10000  # 10k most common words
max_len_instance = len(str(max(train_data[0], key=len)).split())
max_len_question = len(str(max(train_data[1], key=len)).split())
max_len_answer = len(str(max(train_data[2], key=len)).split())
print('data (I,Q,A): ', len(train_data[0]), len(train_data[1]), len(train_data[2]))

# building vocabulary
word2idx, idx2word = build_vocababulary(train_data, test_data)
vocabulary_size = len(word2idx)

print('distinct words: ', vocabulary_size)
print('max lengths (I,Q,A):', max_len_instance, max_len_question, max_len_answer)

# vectorizing data
Xstrain, Xqtrain, Ytrain = vectorize(train_data, word2idx, max_len_instance, max_len_question)
Xstest, Xqtest, Ytest = vectorize(test_data, word2idx, max_len_instance, max_len_question)

# creating network
EMBEDDING_SIZE = 64
LATENT_SIZE = 32

# inputs
instance_input = Input(shape=(max_len_instance,))
question_input = Input(shape=(max_len_question,))

# story encoder memory
story_encoder = Embedding(input_dim=vocabulary_size, output_dim=EMBEDDING_SIZE, input_length=max_len_instance)(instance_input)
story_encoder = Dropout(0.2)(story_encoder)

# question encoder
question_encoder = Embedding(input_dim=vocabulary_size, output_dim=EMBEDDING_SIZE, input_length=max_len_question)(question_input)
question_encoder = Dropout(0.2)(question_encoder)

# match between story and question
match = dot([story_encoder, question_encoder], axes=[2, 2])

# encode story into vector space of question
story_encoder_c = Embedding(input_dim=vocabulary_size, output_dim=max_len_question, input_length=max_len_instance)(instance_input)
story_encoder_c = Dropout(0.2)(story_encoder_c)

# combine match and story vectors
response = add([match, story_encoder_c])
response = Permute((2, 1))(response)

# combine response and question vectors
answer = concatenate([response, question_encoder], axis=-1)
answer = LSTM(LATENT_SIZE)(answer)
answer = Dropout(0.2)(answer)
answer = Dense(vocabulary_size)(answer)
output = Activation("softmax")(answer)
model = Model(inputs=[instance_input, question_input], outputs=output)
model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

# training
BATCH_SIZE = 32
NUM_EPOCHS = 100
history = model.fit([Xstrain, Xqtrain], [Ytrain], batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, validation_data=([Xstest, Xqtest], [Ytest]))

history_dict = history.history  # data during training, history_dict.keys()
epochs = range(1, NUM_EPOCHS + 1)

plot_acc(history_dict, epochs)
