import os
from xml.etree import ElementTree as et

# file locations
data_dir = 'Data'
train_file = 'train-data.xml'
dev_file = 'dev-data.xml'
test_file = 'test-data.xml'
first_file = 'first-data.xml'

# xml file parsing
root = et.parse(os.path.join(data_dir, first_file)).getroot()
instances = []
questions = []
answers = []
num_instances = 0
num_questions = 0

for instance in root:  # instance/story
    num_instances = num_instances + 1
    instances.append(instance[0].text)
    instance_questions = []
    question_answers = []
    for question in instance[1]:  # multiple questions
        num_questions = num_questions + 1
        instance_questions.append(question.attrib['text'])
        if question[0].attrib['correct'] == 'True':  # 2 possible answers; first true, second false
            question_answers.append([question[0].attrib['text'], question[1].attrib['text']])
        else:
            question_answers.append([question[1].attrib['text'], question[0].attrib['text']])
    questions.append(instance_questions)
    answers.append(question_answers)

print(instances)
print(questions)
print(answers)
print(num_instances, num_questions)
