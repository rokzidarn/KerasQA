import os
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

    mca = fdist_answers.most_common(1)

    return word2idx, word2idx_answers, mca

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
        Y.append(word2idx_answers[answer.lower()])

    return pad_sequences(Xi, maxlen=story_maxlen), pad_sequences(Xq, maxlen=question_maxlen), np_utils.to_categorical(Y, num_classes=len(word2idx_answers))

def plot_acc(history_dict, epochs):
    acc = history_dict['acc']
    val_acc = history_dict['val_acc']

    fig = plt.figure()
    plt.plot(epochs, acc, 'r', label='Training acc')
    plt.plot(epochs, val_acc, 'g', label='Testing acc')
    plt.title('Training and testing accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    #fig.savefig('gpu_test.png')

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
word2idx, word2idx_answers, mca = build_vocababulary(train_data[3]+' '+test_data[3], train_data[4]+' '+test_data[4])
vocabulary_size = len(word2idx)
vocabulary_size_answer = len(word2idx_answers)

print('Train + test distinct words: ', vocabulary_size)
print('Baseline: ', round(mca[0][1] / len(train_data[2] + test_data[2]), 3))

# vectorizing data
Xitrain, Xqtrain, Ytrain = vectorize(train_data, word2idx, word2idx_answers, max_len_instance, max_len_question)
Xitest, Xqtest, Ytest = vectorize(test_data, word2idx, word2idx_answers, max_len_instance, max_len_question)

# params
embedding_size = 128
dropout = 0.3
latent_size = 128
answer_dropout = 0.2
epochs = 30

# encoding
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

# creating network
answer = concatenate([response, question_encoder], axis=-1)
answer = LSTM(latent_size)(answer)
answer = Dropout(answer_dropout)(answer)
answer = Dense(vocabulary_size_answer)(answer)
output = Activation("softmax")(answer)
model = Model(inputs=[instance_input, question_input], outputs=output)
model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

# training
history = model.fit([Xitrain, Xqtrain], [Ytrain], batch_size=32, epochs=epochs, validation_data=([Xitest, Xqtest], [Ytest]))

history_dict = history.history  # data during training, history_dict.keys()
print("Max validaton acc: ", round(max(history_dict['val_acc']), 3))
gprah_epochs = range(1, epochs + 1)

plot_acc(history_dict, gprah_epochs)
# ACC: 0.443
