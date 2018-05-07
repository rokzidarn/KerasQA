import os
from xml.etree import ElementTree as Et

def parse_file(directory, file):
    root = Et.parse(os.path.join(directory, file)).getroot()
    instances = []  # list of instances/stories
    questions = []  # 1 question per instance
    answers = []  # true answer per each question, ignore false

    # transforming text data to arrays
    for instance in root:  # instance/story
        instances.append(instance[0].text)
        for question in instance[1]:  # multiple questions
            questions.append(question.attrib['text'])
            if question[0].attrib['correct'] == 'True':  # 2 possible answers; true then false
                answers.append(question[0].attrib['text'])
            else:  # 2 possible answers; false then true
                answers.append(question[1].attrib['text'])
            break

    return instances, questions, answers

# MAIN

# file locations
data_dir = 'Data'
train_file = 'train-data.xml'
dev_file = 'dev-data.xml'
test_file = 'test-data.xml'
first_file = 'first-data.xml'

train_data = parse_file(data_dir, train_file)

# vocabulary stats
max_words = 10000  # 10k most common words
max_len_instance = len(str(max(train_data[0], key=len)).split())
max_len_question = len(str(max(train_data[1], key=len)).split())
max_len_answer = len(str(max(train_data[2], key=len)).split())

print('data (I,Q,A): ', len(train_data[0]), len(train_data[1]), len(train_data[2]))
print('max lengths (I,Q,A):', max_len_instance, max_len_question, max_len_answer)

# TODO: fix IQA lengths
# TODO: build vocabulary + vectorize
