import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

import tensorflow as tf
from tensorflow.keras.models import load_model

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Loading all the data that was trained
lemmatizer = WordNetLemmatizer()
datastore = json.loads(open('jsons/intents.json').read())
emotionData = json.loads(open('jsons/emotions.json').read())
emotionRes = json.loads(open('jsons/emotion_responses.json').read())

words = pickle.load(open('assets/word.pkl', 'rb'))
classes = pickle.load(open('assets/classes.pkl', 'rb'))
model = load_model('model/chatbot_model.h5')

# make the function that contain commands that cleaning up the messages
def clean_up_sentence(msg):
    cleaned = nltk.word_tokenize(msg)
    cleaned = [lemmatizer.lemmatize(word) for word in cleaned]
    return cleaned

def bag_of_words(msg):
    cleaned = clean_up_sentence(msg)
    bag = [0] * len(words)
    for w in cleaned:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(msg):
    bow = bag_of_words(msg)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json, emotions_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    list_of_emotions = emotions_json
    for item_ints in list_of_intents:
        if item_ints['tag'] == tag:
            result = random.choice(item_ints['responses'])
            break
    for item_emo in list_of_emotions:
        if item_emo['Emotion'] == tag:
            for emo in emotionRes:
                if emo['Emotion'] == tag:
                    result = random.choice(emo['Responses']) 
                    break
    return result

# For testing in terminal

"""
# Chatbot Start
print('This is testing chatbot enjoy XD')
name = input("Enter your name : ")

while True:
    message = input(name.upper() + ": ")
    ints = predict_class(message)
    res = get_response(ints, datastore, emotionData)
    print("BOT: " + res)
    if ints[0]['intent'] == 'goodbye':
        print('Chatbot shutdown')
        break
"""