import os
import numpy as np
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
all_text = []  # instances + questions + answers

num_instances = 0
num_questions = 0

# transforming text data to arrays
for instance in root:  # instance/story
    num_instances += 1
    instances.append(instance[0].text)
    all_text.append(instance[0].text)
    instance_questions = []
    question_answers = []
    for question in instance[1]:  # multiple questions
        num_questions += 1
        instance_questions.append(question.attrib['text'])
        all_text.append(question.attrib['text'])
        if question[0].attrib['correct'] == 'True':  # 2 possible answers; first true, second false
            question_answers.append([question[0].attrib['text'], question[1].attrib['text']])
        else:
            question_answers.append([question[1].attrib['text'], question[0].attrib['text']])
        all_text.append(question[0].attrib['text'])
        all_text.append(question[1].attrib['text'])
    questions.append(instance_questions)
    answers.append(question_answers)

# text stats
print('#instances:', num_instances, '\n#questions:', num_questions)
longest_instance = str(max(instances, key=len))

max_words = 10000  # 10k most common words
max_len = 100  # first 100 words of the instance
# TODO: print(len(longest_instance.split())); define max_len -> (152, 793)

# tokenizing + word index
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(all_text)  # words represented as a number
word_index = tokenizer.word_index  # dictionary of distinct words -> (key, value) - ('word', index) -> ('the', 152)
sequences = tokenizer.texts_to_sequences(all_text)  # each instance represented as numerical array
instances_data = pad_sequences(sequences, maxlen=max_len)  # pads sequences to the same length

# print(word_index)
print('#distinct words:', len(word_index))  # number of distinct words

# glove embeddings
embeddings_index = {}  # 400k pretrained word embeddings
f = open(os.path.join(data_dir, 'glove.6B.100d.txt'), encoding="utf8")
# (word, vector representation of the word) -> ('the', [-0.038194 -0.24487   0.72812 ...])

for line in f:
    values = line.split()
    word = values[0]
    coefficients = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefficients
f.close()

embedding_dim = 100
embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():  # goes through all the words in the reviews, each word represented by unique number
    if i < max_words:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:  # if the word is found in the pretrained embeddings, get vector for the word
            embedding_matrix[i] = embedding_vector
        # words not found in the embedding index will be all zeros

print(embeddings_index['the'])
# TODO: add glove files