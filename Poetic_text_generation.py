import numpy as np
from sklearn.feature_selection import SequentialFeatureSelector
import random
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.layers import Activation, Dense, LSTM
from tensorflow.python import tf2 as _tf2
from tensorflow.python.platform import _pywrap_tf2

file_path = _tf2.keras.utils.get_file("shakespeare.txt","https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt")

text = open(file_path,'rb').read().decode(encoding='utf-8').lower()

text = text[30000:80000]

characters = sorted(set(text))
char_to_index  = dict((c,i) for  i,c  in enumerate(characters))
index_to_char   = dict((i,c) for  i,c  in enumerate(characters))

SEQ_LENGTH  = 40
STEP_SIZE = 3

sentences =[]
next_char = []

for i in range(0,len(text)-SEQ_LENGTH):
    sentences.append(text[i:i+SEQ_LENGTH])
    next_char.append(text[i+SEQ_LENGTH])

x = np.zeros((len(sentences),SEQ_LENGTH,len(characters)),dtype=
             np.bool)

y = np.zeros((len(sentences),len(characters)),dtype=
             np.bool)

for i ,satz in enumerate(sentences):
    for t,char in enumerate(characters):
        x[i,t,char_to_index[char]] =1
        y[i,t,char_to_index[next_char]]=1

model  = Sequential()
model.add(LSTM(128,input_shape = (SEQ_LENGTH,len(characters))))
model.add(Dense(len(characters)))
model.add(Activation = 'softmax')
model.compile(loss='categorical_crossentropy' ,optimizer = RMSprop(lr = 0.1) )
model.fit(x,y,batch_size = 256,epochs = 4)

def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generated_text(length,temperature):
    start_index= random.randInt(0,len(text)-SEQ_LENGTH-1)
    generated = ""
    sentence = text[start_index:start_index+SEQ_LENGTH]

    for i in range(length):
        x_prediction = np.zeros((0,SEQ_LENGTH,len(characters)))
        for t,char in enumerate(sentence):
            x_prediction[i,t,char_to_index[char]] =1

        prediction = model.predict(x_prediction,verbose =0)[0]
        next_index = sample(prediction,temperature)
        next_character  = index_to_char[next_index]
        generated += next_character
        sentence  = sentence[1:]+ next_character

        return generated
    
print(generated_text(300,0.2))
