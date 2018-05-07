import os
import numpy as np
from functools import reduce
from xml.etree import ElementTree as Et
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# file locations
data_dir = 'Data'
train_file = 'train-data.xml'
dev_file = 'dev-data.xml'
test_file = 'test-data.xml'
first_file = 'first-data.xml'

# xml file parsing
root = Et.parse(os.path.join(data_dir, first_file)).getroot()
instances = []  # list of instances/stories
questions = []
answers = []

num_instances = 0
num_questions = 0
num_answers = 0  # only true answers

# transforming text data to arrays
for instance in root:  # instance/story
    num_instances += 1
    instances.append(instance[0].text)
    instance_questions = []
    for question in instance[1]:  # multiple questions
        num_questions += 1
        instance_questions.append(question.attrib['text'])
        if question[0].attrib['correct'] == 'True':  # 2 possible answers; true then false
            num_answers += 1
            answers.append(question[0].attrib['text'])
        else:  # 2 possible answers; false then true
            num_answers += 1
            answers.append(question[1].attrib['text'])
    questions.append(instance_questions)

# vocabulary stats
max_len_instance = len(str(max(instances, key=len)).split())
max_len_question = len(str(max(questions, key=len)).split())
max_len_answer = len(str(max(answers, key=len)).split())

print('#data (I,Q,A):', num_instances, num_questions, num_answers)
print('#max lengths (I,Q,A):', max_len_instance, max_len_question, max_len_answer)

max_words = 10000  # 10k most common words

# tokenizing + word index
tokenizer = Tokenizer(num_words=max_words)

# instances
all_instances = ' '.join(instances)
tokenizer.fit_on_texts(all_instances)  # words represented as a number
word_index_instances = tokenizer.word_index  # dictionary of distinct words -> (key, value) - ('word', index) -> ('the', 152)
sequences = tokenizer.texts_to_sequences(all_instances)  # each instance represented as numerical array
instances_data = pad_sequences(sequences, maxlen=max_len_instance)  # pads sequences to the same length

# questions
qs = reduce(lambda x, y: x+y, questions)
all_questions = ' '.join(qs)
tokenizer.fit_on_texts(all_questions)
word_index_questions = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(all_questions)
questions_data = pad_sequences(sequences, maxlen=max_len_question)

# answers
all_answers = ' '.join(answers)
tokenizer.fit_on_texts(all_answers)
word_index_answers = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(all_answers)
answers_data = pad_sequences(sequences, maxlen=max_len_answer)

# print(word_index)
print('#distinct words (I,Q,A):', len(word_index_instances), len(word_index_questions), len(word_index_answers))

# glove embeddings
glove_dir = '../Glove'
embeddings_index = {}  # 400k pretrained word embeddings
f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'), encoding="utf8")
# (word, vector representation of the word) -> ('the', [-0.038194 -0.24487   0.72812 ...])

for line in f:
    values = line.split()
    word = values[0]
    coefficients = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefficients
f.close()

# creation of embedding matrices
embedding_dim = 100

#instances
embedding_matrix_instances = np.zeros((max_words, embedding_dim))
for word, i in word_index_instances.items():  # goes through all the words in the instances, each word represented by unique number
    if i < max_words:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:  # if the word is found in the pretrained embeddings, get vector for the word
            embedding_matrix_instances[i] = embedding_vector
        # words not found in the embedding index will be all zeros

#questions
embedding_matrix_questions = np.zeros((max_words, embedding_dim))
for word, i in word_index_questions.items():  # goes through all the words in the instances, each word represented by unique number
    if i < max_words:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:  # if the word is found in the pretrained embeddings, get vector for the word
            embedding_matrix_questions[i] = embedding_vector
        # words not found in the embedding index will be all zeros

#answers
embedding_matrix_answers = np.zeros((max_words, embedding_dim))
for word, i in word_index_answers.items():  # goes through all the words in the instances, each word represented by unique number
    if i < max_words:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:  # if the word is found in the pretrained embeddings, get vector for the word
            embedding_matrix_answers[i] = embedding_vector
        # words not found in the embedding index will be all zeros

# model
