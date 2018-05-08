import os
import numpy as np
from functools import reduce
from xml.etree import ElementTree as Et
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras import layers
from keras import Input

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

# print(word_index)
print('#distinct words (I,Q,A):', len(word_index_instances))