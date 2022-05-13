import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Loading dataset for training
lemmatizer = WordNetLemmatizer()

datastore = json.loads(open('jsons/intents.json').read())
emotionData = json.loads(open('jsons/emotions.json').read())

# Necessary variable
words = []
classes = []
documents = []
ignore_letters = ['?', '!', ',', '.']

# Appending item from intents.json
for item_ints in datastore['intents']:
    for pattern in item_ints['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list,item_ints['tag']))
        if item_ints['tag'] not in classes:
            classes.append(item_ints['tag'])

# Appending item from emotions.json
for item_emo in emotionData:
    word_list = nltk.word_tokenize(item_emo['Text'])
    words.extend(word_list)
    documents.append((word_list,item_emo['Emotion']))
    if item_emo['Emotion'] not in classes:
        classes.append(item_emo['Emotion'])

# Stemming the word by WordNetLemmatizer
words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]

# Remove duplicated value by casting to a set then casting it back to a list by sorted()
words = sorted(set(words))
classes = sorted(set(classes))

# Saving words and classes as a pickle file for using it in chatbot
pickle.dump(words, open('assets/word.pkl', 'wb'))
pickle.dump(classes, open('assets/classes.pkl', 'wb'))

# Training data
training = []
output_empty = [0] * len(classes)

# Make the text in documents to be the numeric data
for item_doc in documents:
    bag = []
    word_patterns = item_doc[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(item_doc[1])] = 1
    training.append([bag, output_row])

# Shuffle the data and cast it into an array
random.shuffle(training)
training = np.array(training)

# Separate the training data: text into x and tag into y
train_x = list(training[:, 0])
train_y = list(training[:, 1])

# Build the model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]), ), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.summary()

# Training and save the file for using in chatbot for predicate
hist = model.fit(np.array(train_x), np.array(train_y), epochs=2000, batch_size=5, verbose=1)
model.save('model/chatbot_model.h5', hist)
print('Done')