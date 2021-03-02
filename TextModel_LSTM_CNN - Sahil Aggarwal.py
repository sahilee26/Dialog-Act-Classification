import pandas as pd
import numpy as np
import os
import pickle
import numpy as np
import re
import math
import itertools
from collections import Counter
from New_Utils import *
from keras.layers import Input, Dense, Embedding, Conv2D, MaxPool2D, LSTM
from keras.layers import Reshape, Flatten, Dropout, Concatenate
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from keras.optimizers import *
from keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import *
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.regularizers import *
import tensorflow as tf
from keras.layers.normalization import BatchNormalization


data_train = pd.read_csv("train_sent_emo.txt", encoding="utf-16",sep='\t',header=None)

train = [s.strip() for s in data_train[1]]

text = train
max1=0

for i in range(0,len(text)):
	data=text[i].split(" ")
	max2=len(data)
	if(max2>max1):
		max1=max2


sequence_length = max1

train_label=data_train[11]

lbl_dict={}
index=0
for dial_lbls in train_label:
	if dial_lbls not in lbl_dict:
		lbl_dict[dial_lbls]=index
		index=index+1

print(lbl_dict)

print(data_train[1])

tokenizer=load_create_tokenizer(data_train[0],None,True)
X_train=load_create_padded_data(X_train=train,savetokenizer=False,isPaddingDone=False,maxlen=sequence_length,tokenizer_path='./New_Tokenizer.tkn')
word_index=tokenizer.word_index
embedding_matrix=load_create_embedding_matrix(word_index,len(word_index)+1,300,'glove.6B.300d.txt',False,True,'./Emb_Mat.mat')
f=open("Emb_Mat.mat","rb")
embedding_matrix=pickle.load(f)
f.close()

def create_label(label):
	
    Y=[]
    for i in label:
    	xxx=np.empty(int(len(lbl_dict)))
    	xxx.fill(0)
    	j=lbl_dict.get(i)
    	xxx[j]=1
    	Y.append(xxx)
    return Y

label = train_label
Y_train = create_label(label)

print(Y_train[1])
y_train=np.array([i for i in Y_train])

embedding_dim = 300

T=[]
M=[]

print(y_train[1])

print("Creating Model...")
inputs = Input(shape=(sequence_length,), dtype='int32')
embedding = Embedding(input_dim=len(word_index)+1, output_dim=embedding_dim, weights=[embedding_matrix], input_length=sequence_length)(inputs)
conv_0 = Conv1D(filters=16, kernel_size=3, padding='same', kernel_initializer='normal')(embedding)
conv_0 = LeakyReLU(alpha=0.1)(conv_0)
maxpool_0 = MaxPooling1D(pool_size=2)(conv_0)

conv_1 = Conv1D(filters=16, kernel_size=4, padding='same', kernel_initializer='normal')(embedding)
conv_1 = LeakyReLU(alpha=0.1)(conv_1)
maxpool_1 = MaxPooling1D(pool_size=2)(conv_1)

conv_2 = Conv1D(filters=16, kernel_size=5, padding='same', kernel_initializer='normal')(embedding)
conv_2 = LeakyReLU(alpha=0.1)(conv_2)
maxpool_2 = MaxPooling1D(pool_size=2)(conv_2)
x = Concatenate()([maxpool_0, maxpool_1, maxpool_2])
x = Conv1D(filters=32, kernel_size=5, padding='same', kernel_initializer='normal')(embedding)
x = LeakyReLU(alpha=0.1)(x)
x = MaxPooling1D(pool_size=2)(x)

x = Bidirectional(LSTM(300))(x)


predictions = Dense(1000, activation='sigmoid')(x)
predictions = LeakyReLU(alpha=0.1)(predictions)

predictions1 = Dense(44)(predictions)
checkpoint = ModelCheckpoint('weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
adam = Adam(lr=0.01, decay=0.3)

model = Model(inputs, outputs=predictions1)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()
model.fit([X_train], y_train, epochs=20, callbacks=[checkpoint], batch_size=64, validation_split=0.2)
