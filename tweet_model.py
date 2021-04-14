import pandas as pd
import numpy as np
import os
import pickle
import numpy as np
import re
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


test=[]
train=[]
train_Text='./tweet_train.txt'
test_Text='./tweet_test.txt'

with open(train_Text) as f:
    for line in f:
    	train.append(line)

f.close()


with open(test_Text) as f:
    for line in f:
    	test.append(line)

f.close()

train = [s.strip() for s in train]
test = [s.strip() for s in test]

text = train + test
max1=0


for i in range(0,len(text)):
	data=text[i].split(" ")
	max2=len(data)
	if(max2>max1):
		max1=max2


sequence_length = max1

'''
text = [s.split(" ") for s in text]
train = [s.split(" ") for s in train]
test = [s.split(" ") for s in test]

sequence_length = max(len(x) for x in text)
'''

train_char=[]

for i in train:
	data=i.split(" ")
	train_char.append(data)

max2=0
dict1={}
t=1
for i in range(0,len(train_char)):
	for j in train_char[i]:
		data=list(j)
		if(len(data)>max2):
			max2=len(data)
		for k in data:
			if k not in dict1:
				dict1[k]=t
				t=t+1 



test_char=[]

for i in test:
	data=i.split(" ")
	test_char.append(data)

max3=0
dict2={}
t=1
for i in range(0,len(test_char)):
	for j in test_char[i]:
		data=list(j)
		if(len(data)>max2):
			max3=len(data)

if(max3>max2):
	max2=max3

X_train_char=[]
X_test_char=[]

for i in range(0,len(train_char)):
	word1=[]
	for j in train_char[i]:
		word=[]
		data=list(j)
		for k in data:
			word.append(dict1.get(k))
		word1.append(word)
	word1=pad_sequences(word1,maxlen=max2)
	X_train_char.append(word1)


for i in range(0,len(test_char)):
	word1=[]
	for j in test_char[i]:
		word=[]
		data=list(j)
		for k in data:
			if k in dict1:
				word.append(dict1.get(k))
			else:
				word.append(int(0))
		word1.append(word)
	word1=pad_sequences(word1,maxlen=max2)
	X_test_char.append(word1)



X_train_char=pad_sequences(X_train_char,maxlen=sequence_length)
X_test_char=pad_sequences(X_test_char,maxlen=sequence_length)

X_train_char=np.array([i for i in X_train_char])
X_test_char=np.array([i for i in X_test_char])	

embedding_matrix1=load_create_embedding_matrix(dict1,len(dict1)+1,300,'./glove.840B.300d-char.txt',False,True,'./Emb_Mat1.mat')

tokenizer=load_create_tokenizer(train,None,True)
X_train=load_create_padded_data(X_train=train,savetokenizer=False,isPaddingDone=False,maxlen=sequence_length,tokenizer_path='./New_Tokenizer.tkn')
X_test=load_create_padded_data(X_train=test,savetokenizer=False,isPaddingDone=False,maxlen=sequence_length,tokenizer_path='./New_Tokenizer.tkn')
word_index=tokenizer.word_index
embedding_matrix=load_create_embedding_matrix(word_index,len(word_index)+1,200,'./glove.twitter.27B.200d.txt',False,True,'./Emb_Mat.mat')


test_label=[]
train_label=[]
train_Text='./tweet_train_label.txt'
test_Text='./tweet_test_label.txt'

with open(train_Text) as f:
    for line in f:
    	line=line.strip()
    	line=line.upper()
    	train_label.append(line)

f.close()


with open(test_Text) as f:
    for line in f:
    	line=line.strip()
    	line=line.upper()
    	test_label.append(line)

f.close()


lbl_dict={}
index=0
for dial_lbls in train_label:
	if dial_lbls not in lbl_dict:
		lbl_dict[dial_lbls]=index
		index=index+1

print(len(lbl_dict))

def create_label(label):
	
    Y=[]
    for i in label:
    	xxx=np.empty(int(len(lbl_dict)))
    	xxx.fill(-1)
    	j=lbl_dict.get(i)
    	xxx[j]=1
    	Y.append(xxx)
    return Y

label = train_label
Y_train = create_label(label)

label = test_label
Y_test = create_label(label)

y_train=np.array([i for i in Y_train])
y_test=np.array([i for i in Y_test])

embedding_dim = 200
embedding_dim1=200

#########################3
T=[]
K=[]

for i in range(0, len(y_train)):
    	data=[]
    	data.append(float(0))
    	T.append(data)
    	
for i in range(0, len(y_test)):
    	data=[]
    	data.append(float(0))
    	K.append(data)
    	
T=np.array([i for i in T])
K=np.array([i for i in K])

################ for additional feature

f=open("final_feature_train1.pkl","rb")
Z=pickle.load(f)
f.close()

f=open("final_feature_test1.pkl","rb")
A=pickle.load(f)
f.close()

T=[]
M=[]

for i in Z:
    data=i.split(" ")
    print(len(data))
    for j in range(0,len(data)):
    	data[j]=float(data[j])
    T.append(data)
    
for i in A:
    data=i.split(" ")
    print(len(data))
    for j in range(0,len(data)):
    	data[j]=float(data[j])
    M.append(data)

T=np.array([i for i in T])
M=np.array([i for i in M])

feature_length = max(len(x) for x in T)

############## for CNN-SVM (original)


	def myloss(y_true, y_pred, batch_size, penalty_parameter):
	   
	    regularization_loss = tf.reduce_mean(tf.square(model.layers[len(model.layers)-1].get_weights()[0]))
	    print("done ##################")
	    hinge_loss = tf.reduce_mean(tf.square(tf.maximum(tf.zeros([batch_size, 7]), 1 - y_true * y_pred)))
	    print("done#########################3")
	    loss = regularization_loss + penalty_parameter * hinge_loss
	    print("returning loss##############")
	    return loss

	def custom_loss(batch_size, penalty_parameter):
	    
	    def new_loss(y_true, y_pred):
	    	#h=h
	    	return myloss(y_true, y_pred, batch_size, penalty_parameter)
	    return new_loss
    

############## for CNN-SVM (modified)


def myloss(y_true, y_pred, batch_size, penalty_parameter, class_weights):
    print(y_true.shape)
    regularization_loss = tf.reduce_mean(tf.square(model.layers[len(model.layers)-1].get_weights()[0]))
    print("done ##################")
    hinge_loss = tf.square(tf.maximum(tf.zeros([batch_size, 7]), 1 - y_true * y_pred))
    print("done#########################3")
    y_ints = [y.argmax() for y in y_true]
    y_ints1 = [y.argmax() for y in y_pred]
    extra = tf.ones(batch_size)
    for i in range(0,len(y_ints)):
    	if(y_ints[i]!=y_ints1[i]):
    		k=class_weights.get(y_ints[i])
    		extra[i]=k
    hinge_loss = tf.multiply(hinge_loss, extra)
    hinge_loss = tf.reduce_mean(hinge_loss)	
    loss = regularization_loss + penalty_parameter * hinge_loss
    print("returning loss##############")
    return loss

def custom_loss(batch_size, penalty_parameter, class_weights):
    
    def new_loss(y_true, y_pred):
    	#h=h
    	return myloss(y_true, y_pred, batch_size, penalty_parameter, class_weights)
    return new_loss
    



model = Sequential()
model.add(Embedding(input_dim=len(word_index)+1, output_dim=embedding_dim, weights=[embedding_matrix], input_length=sequence_length))
model.add(Conv1D(filters=32, kernel_size=5, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.1))
model.add(Conv1D(filters=64, kernel_size=5, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(8, W_regularizer=l2(0.3)))
model.add(Activation('linear'))
checkpoint = ModelCheckpoint('./CNN1D/weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
adam = Adam(lr=0.01, decay=0.3)
model.compile(optimizer=adam, loss='squared_hinge', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, epochs=200, callbacks=[checkpoint], batch_size=80, validation_data=(X_test, y_test))
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

predicted=model.predict(X_test)
print("DONE")

################### for CNN-softmax

model = Sequential()
model.add(Embedding(input_dim=len(dict1)+1, output_dim=embedding_dim1, weights=[embedding_matrix1], input_length=(sequence_length, max2)))

model = Sequential()
model.add(Embedding(input_dim=len(word_index)+1, output_dim=embedding_dim, weights=[embedding_matrix], input_length=sequence_length))
model.add(Conv1D(filters=32, kernel_size=5, padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.1))
model.add(Conv1D(filters=64, kernel_size=5, padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(1000))
model.add(LeakyReLU(alpha=0.1))
model.add(Dense(len(lbl_dict), activation='softmax'))
checkpoint = ModelCheckpoint('./CNN1D/weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
#stop = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='max')
adam = Adam(lr=0.01, decay=0.3)
model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, class_weight=class_weights, epochs=100, callbacks=[checkpoint], batch_size=30, validation_data=(X_test, y_test))
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

predicted=model.predict(X_test)
print("DONE")


################### for CNN-SVM additional features

model = Sequential()
model.add(Embedding(input_dim=len(word_index)+1, output_dim=embedding_dim, weights=[embedding_matrix], input_length=sequence_length))
model.add(Conv1D(filters=32, kernel_size=5, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.1))
model.add(Conv1D(filters=64, kernel_size=5, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
auxiliary_input = Input(shape=(213,), name='aux_input')
model1 = Sequential()
model1.add(auxiliary_input)
model1.add(merge([model, auxiliary_input], mode='concat', concat_axis=1))
model1.add(Dense(1000, activation='relu'))
model1.add(Dense(8, W_regularizer=l2(0.5)))
model1.add(Activation('linear'))
checkpoint = ModelCheckpoint('./CNN1D/weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
adam = Adam(lr=0.01, decay=0.3)
model1.compile(loss='squared_hinge', optimizer=adam, metrics=['accuracy'])
print(model.summary())
model1.fit([X_train,T], y_train, epochs=100, callbacks=[checkpoint], batch_size=50)
#scores = model1.evaluate(X_test, y_test, verbose=0)
#print("Accuracy: %.2f%%" % (scores[1]*100))

#predicted=model1.predict(X_test)
print("DONE")
##############################################


T=[]

>>> for i in Z:
...     data=i.split(" ")
...     for j in range(0,len(data)):
...             data[j]=float(data[j])
...     T.append(data)
... 


T=np.array([i for i in T])
############################################################## final model dc=0.3, pp=1.5, without batch (working the best till now)

print("Creating Model...")
inputs = Input(shape=(sequence_length,), dtype='int32')
embedding = Embedding(input_dim=len(word_index)+1, output_dim=embedding_dim, weights=[embedding_matrix], input_length=sequence_length)(inputs)
conv_0 = Conv1D(filters=64, kernel_size=10, padding='same', kernel_initializer='normal', activation='relu')(embedding)
maxpool_0 = MaxPooling1D(pool_size=2)(conv_0)
#t= BatchNormalization()(maxpool_0)
dropout = Dropout(0.1)(maxpool_0)
conv_1 = Conv1D(filters=128, kernel_size=10, padding='same', kernel_initializer='normal', activation='relu')(dropout)
maxpool_1 = MaxPooling1D(pool_size=2)(conv_1)
flatten = Flatten()(maxpool_1)
auxiliary_input = Input(shape=(feature_length,), name='aux_input')
x = Concatenate()([auxiliary_input, flatten])
predictions = Dense(1000, activation='relu')(x)
predictions1 = Dense(7)(predictions)
checkpoint = ModelCheckpoint('./CNN1D/weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
adam = Adam(lr=0.01, decay=0.3)
model_loss= custom_loss(penalty_parameter=1, batch_size=30)
model = Model(inputs=[inputs, auxiliary_input], outputs=predictions1)
model.compile(optimizer=adam, loss=model_loss, metrics=['accuracy'])
model.fit([X_train, T], y_train, epochs=200, callbacks=[checkpoint], batch_size=30, validation_data=([X_test,K], y_test))


################## custom loss function

for i in range(0,len(model.layers)):
    	if(i==len(model.layers)):
    		h=model.layers[i].get_weights()[0]


 global h
    for i in range(0,len(model.layers)):
    	if(i==len(model.layers)):
    		h=model.layers[i].get_weights()[0]

def myloss(y_true, y_pred, batch_size, penalty_parameter):
   
    regularization_loss = tf.reduce_mean(tf.square(model.layers[len(model.layers)-1].get_weights()[0]))
    print("done ##################")
    print(regularization_loss.shape)
    hinge_loss = tf.reduce_mean(tf.square(tf.maximum(tf.zeros([40, 8]), 1 - y_true * y_pred)))
    print("done#########################3")
    loss = regularization_loss + penalty_parameter * hinge_loss
    print("returning loss##############")
    return loss



def custom_loss(batch_size, penalty_parameter):
    print("in custom loss")
    
    def new_loss(y_true, y_pred):
    	#h=h
    	#print(batch_size)
    	#print(y_pred.shape)
    	return myloss(y_true, y_pred, batch_size, penalty_parameter)
    return new_loss


##################################

print("Creating Model...")
inputs = Input(shape=(sequence_length,), dtype='int32')
embedding = Embedding(input_dim=len(word_index)+1, output_dim=embedding_dim, weights=[embedding_matrix], input_length=sequence_length)(inputs)
reshape = Reshape((sequence_length,embedding_dim,1))(embedding)
conv_0 = Conv2D(32, kernel_size=(5, 5), strides=(1, 1),activation='relu')(reshape)
maxpool_0 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv_0)
t= BatchNormalization()(maxpool_0)
dropout = Dropout(0.1)(t)
conv_1 = Conv2D(64, (5, 5), activation='relu')(dropout)
maxpool_1 = MaxPooling2D(pool_size=(2, 2))(conv_1)
flatten = Flatten()(maxpool_1)
auxiliary_input = Input(shape=(feature_length,), name='aux_input')
x = Concatenate()([auxiliary_input, flatten])
predictions = Dense(1000, activation='relu')(x)
predictions1 = Dense(8, activation='softmax')(predictions)
checkpoint = ModelCheckpoint('./CNN1D/weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
adam = Adam(lr=0.1, decay=0.3)
#model_loss= custom_loss(penalty_parameter=1, batch_size=15)
model = Model(inputs=[inputs, auxiliary_input], outputs=predictions1)
model.compile(optimizer=adam, loss="categorical_crossentropy", metrics=['accuracy'])
model.fit([X_train, T], y_train, epochs=200, callbacks=[checkpoint], batch_size=32, validation_data=([X_test,K], y_test))



weights.048-0.7118.hdf5

weights.034-0.7161.hdf5


###########
Y_predicted=[]
for i in predicted:
    pos=i.argmax()
    Y_predicted.append(pos)
    
    
    
Y_test=[]
for i in y_test:
    pos=i.argmax()
    Y_test.append(pos)
    
    
    
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split

print(accuracy_score(Y_test, Y_predicted))
print(classification_report(Y_test, Y_predicted,digits=4))
print(confusion_matrix(Y_test, Y_predicted))
    
    
    
    
      		 precision    recall  f1-score   support

          STM     0.6296    0.6210    0.6253       219
          THT     0.8214    0.5111    0.6301        45
          QUE     0.8037    0.8113    0.8075       106
          EXP     0.7198    0.8695    0.7876       452
          SUG     0.7143    0.1316    0.2222        38
          REQ     0.7391    0.4359    0.5484        39
          OTH     0.6667    0.0645    0.1176        31

avg / total     0.7119    0.7118    0.6886       930




###################################


print("Creating Model...")
inputs = Input(shape=(sequence_length,), dtype='int32')
embedding = Embedding(input_dim=len(word_index)+1, output_dim=embedding_dim, weights=[embedding_matrix], input_length=sequence_length)(inputs)
conv_0 = Conv1D(filters=32, kernel_size=5, padding='same', kernel_initializer='normal', activation='relu')(embedding)
maxpool_0 = MaxPooling1D(pool_size=2)(conv_0)
#t= BatchNormalization()(maxpool_0)
dropout = Dropout(0.1)(maxpool_0)
conv_1 = Conv1D(filters=64, kernel_size=5, padding='same', kernel_initializer='normal', activation='relu')(dropout)
maxpool_1 = MaxPooling1D(pool_size=2)(conv_1)
flatten = Flatten()(maxpool_1)
auxiliary_input = Input(shape=(feature_length,), name='aux_input')
x = Concatenate()([auxiliary_input, flatten])
predictions = Dense(1000, activation='relu')(x)
predictions1 = Dense(7, activation='softmax', W_regularizer=l2(0.001))(predictions)
checkpoint = ModelCheckpoint('./CNN1D/weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
adam = SGD(lr=0.01, decay=0.00001)
#model_loss= custom_loss(penalty_parameter=1.5, batch_size=15)
model = Model(inputs=[inputs, auxiliary_input], outputs=predictions1)
model.compile(optimizer=adam, loss="categorical_crossentropy", metrics=['accuracy'])
model.fit([X_train, T], y_train, epochs=200, callbacks=[checkpoint], batch_size=32, validation_data=([X_test,K], y_test))








####################################
##############    CNN-SVM for 7 class
0.7161290322580646
>>> print(classification_report(Y_test, Y_predicted,digits=4))
             precision    recall  f1-score   support

          0     0.6388    0.6388    0.6388       227
          1     0.7250    0.5918    0.6517        49
          2     0.7429    0.8423    0.7895       463
          3     0.5000    0.1364    0.2143        44
          4     0.8125    0.4815    0.6047        27
          5     0.7000    0.2188    0.3333        32
          6     0.7600    0.8636    0.8085        88

avg / total     0.7072    0.7161    0.6990       930

>>> print(confusion_matrix(Y_test, Y_predicted))
[[145   8  67   2   0   1   4]
 [ 13  29   6   0   0   1   0]
 [ 48   1 390   4   2   1  17]
 [  8   0  29   6   0   0   1]
 [  3   1   8   0  13   0   2]
 [  8   0  17   0   0   7   0]
 [  2   1   8   0   1   0  76]]

model.load_weights("./CNN1D/weights.006-0.6720.hdf5")


################### for CNN softmax
0.6720430107526881
>>> print(classification_report(Y_test, Y_predicted,digits=4))
             precision    recall  f1-score   support

          0     0.5789    0.5815    0.5802       227
          1     0.6750    0.5510    0.6067        49
          2     0.7034    0.8402    0.7657       463
          3     0.3750    0.0682    0.1154        44
          4     0.7000    0.2593    0.3784        27
          5     0.2000    0.0312    0.0541        32
          6     0.7674    0.7500    0.7586        88

avg / total     0.6446    0.6720    0.6449       930

>>> print(confusion_matrix(Y_test, Y_predicted))
[[132   7  82   1   0   1   4]
 [ 14  27   8   0   0   0   0]
 [ 52   1 389   3   2   3  13]
 [ 16   1  23   3   0   0   1]
 [  3   2  13   0   7   0   2]
 [  7   1  22   0   1   1   0]
 [  4   1  16   1   0   0  66]]


########################### for word and character embedding softmax

print("Creating Model...")
inputs = Input(shape=(X_train_char.shape[1],X_train_char.shape[2]), dtype='int32')
embedding = Embedding(input_dim=len(dict1)+1, output_dim=embedding_dim1, weights=[embedding_matrix1], input_length=(X_train_char.shape[1],X_train_char.shape[2]))(inputs)
conv_0 = Conv2D(64, kernel_size=(5, 5), strides=(1, 1),activation='relu')(embedding)
maxpool_0 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv_0)
dropout = Dropout(0.1)(maxpool_0)
conv_1 = Conv2D(64, kernel_size=(5, 5), strides=(1, 1),activation='relu')(dropout)
maxpool_1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv_1)
flatten = Flatten()(maxpool_1)
inputs1 = Input(shape=(sequence_length,), dtype='int32')
embedding1 = Embedding(input_dim=len(word_index)+1, output_dim=embedding_dim, weights=[embedding_matrix], input_length=sequence_length)(inputs1)
conv_01 = Conv1D(filters=32, kernel_size=5, padding='same', kernel_initializer='normal', activation='relu')(embedding1)
maxpool_01 = MaxPooling1D(pool_size=2)(conv_01)
#t= BatchNormalization()(maxpool_0)
dropout1 = Dropout(0.1)(maxpool_01)
conv_11 = Conv1D(filters=64, kernel_size=5, padding='same', kernel_initializer='normal', activation='relu')(dropout1)
maxpool_11 = MaxPooling1D(pool_size=2)(conv_11)
flatten1 = Flatten()(maxpool_11)
x = Concatenate()([flatten1, flatten])
predictions = Dense(1000, activation='relu')(x)
predictions1 = Dense(len(lbl_dict), activation='softmax')(predictions)
checkpoint = ModelCheckpoint('./CNN1D/weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
adam = Adam(lr=0.01, decay=0.3)
model = Model(inputs=[inputs,inputs1], outputs=predictions1)
model.compile(optimizer='SGD', loss="categorical_crossentropy", metrics=['accuracy'])
model.fit([X_train_char,X_train], y_train, epochs=200, callbacks=[checkpoint], batch_size=128, validation_data=([X_test_char,X_test], y_test))


############################## for char embedding SVM

print("Creating Model...")
inputs = Input(shape=(X_train_char.shape[1],X_train_char.shape[2]), dtype='int32')
embedding = Embedding(input_dim=len(dict1)+1, output_dim=embedding_dim1, weights=[embedding_matrix1], input_length=(X_train_char.shape[1],X_train_char.shape[2]))(inputs)
reshape = Reshape((X_train_char.shape[1]*X_train_char.shape[2],embedding_dim1))(embedding)
conv_0 = Conv1D(filters=32, kernel_size=5, padding='same', kernel_initializer='normal', activation='relu')(reshape)
maxpool_0 = MaxPooling1D(pool_size=2)(conv_0)
conv_1 = Conv1D(filters=32, kernel_size=5, padding='same', kernel_initializer='normal', activation='relu')(maxpool_0)
maxpool_1 = MaxPooling1D(pool_size=2)(conv_1)
layer_1 = LSTM(100)(maxpool_1)
#flatten = Flatten()(layer_1)
auxiliary_input = Input(shape=(feature_length,), name='aux_input')
x = Concatenate()([auxiliary_input, layer_1])
predictions = Dense(1000, activation='relu')(x)
#attention_probs = Dense(1000, activation='softmax', name='attention_vec')(predictions)
#predictions = merge([predictions, attention_probs], output_shape=32, name='attention_mul', mode='mul')
predictions1 = Dense(7)(predictions)
checkpoint = ModelCheckpoint('./CNN1D/weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
adam = Adam(lr=0.01, decay=0.3)
model_loss= custom_loss(penalty_parameter=1.5, batch_size=10)
model = Model(inputs=[inputs, auxiliary_input], outputs=predictions1)
model.compile(optimizer=adam, loss=model_loss, metrics=['accuracy'])
model.fit([X_train_char, T], y_train, epochs=200, callbacks=[checkpoint], batch_size=10, validation_data=([X_test_char,K], y_test))


##################### for word and character embedding SVM
print("Creating Model...")
inputs = Input(shape=(X_train_char.shape[1],X_train_char.shape[2]), dtype='int32')
#embedding = Embedding(input_dim=len(dict1)+1, output_dim=embedding_dim1, weights=[embedding_matrix1], input_length=(X_train_char.shape[1],X_train_char.shape[2]))(inputs)
embedding = Embedding(input_dim=len(dict1)+1, output_dim=embedding_dim1, embeddings_initializer='uniform', input_length=(X_train_char.shape[1],X_train_char.shape[2]), trainable=True)(inputs)
reshape = Reshape((X_train_char.shape[1]*X_train_char.shape[2],embedding_dim1))(embedding)
conv_0 = Conv1D(filters=32, kernel_size=5, padding='same', kernel_initializer='normal', activation='relu')(reshape)
maxpool_0 = MaxPooling1D(pool_size=2)(conv_0)
conv_1 = Conv1D(filters=32, kernel_size=5, padding='same', kernel_initializer='normal', activation='relu')(maxpool_0)
maxpool_1 = MaxPooling1D(pool_size=2)(conv_1)
layer_1 = LSTM(200)(maxpool_1)
flatten = Flatten()(layer_1)
inputs1 = Input(shape=(sequence_length,), dtype='int32')
embedding1 = Embedding(input_dim=len(word_index)+1, output_dim=embedding_dim, weights=[embedding_matrix], input_length=sequence_length)(inputs1)
conv_01 = Conv1D(filters=32, kernel_size=5, padding='same', kernel_initializer='normal', activation='relu')(embedding1)
maxpool_01 = MaxPooling1D(pool_size=2)(conv_01)
conv_11 = Conv1D(filters=64, kernel_size=5, padding='same', kernel_initializer='normal', activation='relu')(maxpool_01)
maxpool_11 = MaxPooling1D(pool_size=2)(conv_11)
flatten1 = Flatten()(maxpool_11)
x = Concatenate()([flatten1, layer_1])
auxiliary_input = Input(shape=(feature_length,), name='aux_input')
x = Concatenate()([auxiliary_input, x])
predictions = Dense(1000, activation='relu')(x)
predictions1 = Dense(len(lbl_dict), activation='softmax')(predictions)
checkpoint = ModelCheckpoint('./CNN1D/weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
adam = Adam(lr=0.01, decay=0.3)
model_loss= custom_loss(penalty_parameter=1, batch_size=10)
model = Model(inputs=[inputs,inputs1,auxiliary_input], outputs=predictions1)
model.compile(optimizer=adam, loss=model_loss, metrics=['accuracy'])
model.fit([X_train_char,X_train,T], y_train, epochs=200, callbacks=[checkpoint], batch_size=10, validation_data=([X_test_char,X_test,K], y_test))



########################## 25-72.90, pp=1.5, final model


print("Creating Model...")
inputs = Input(shape=(sequence_length,), dtype='int32')
embedding = Embedding(input_dim=len(word_index)+1, output_dim=embedding_dim, weights=[embedding_matrix], input_length=sequence_length)(inputs)
conv_0 = Conv1D(filters=32, kernel_size=5, padding='same', kernel_initializer='normal')(embedding)
conv_0 = LeakyReLU(alpha=0.1)(conv_0)
maxpool_0 = MaxPooling1D(pool_size=2)(conv_0)
#t= BatchNormalization()(maxpool_0)
dropout = Dropout(0.1)(maxpool_0)
conv_1 = Conv1D(filters=64, kernel_size=5, padding='same', kernel_initializer='normal')(dropout)
conv_1 = LeakyReLU(alpha=0.1)(conv_1)
maxpool_1 = MaxPooling1D(pool_size=2)(conv_1)
flatten = Flatten()(maxpool_1)
#layer_1 = TimeDistributed(Dense(32), input_shape=(sequence_length, embedding_dim))(conv_0)
#layer_2 = Bidirectional(LSTM(300))(layer_1)
#layer_2 = Dense(256)(layer_2)
#layer_2 = LeakyReLU(alpha=0.1)(layer_2)
auxiliary_input = Input(shape=(feature_length,), name='aux_input')
#x = Concatenate()([layer_2, flatten])
#x = Dense(1000)(x)
#x = LeakyReLU(alpha=0.1)(x)
x = Concatenate()([auxiliary_input, flatten])
predictions = Dense(1000)(x)
predictions = LeakyReLU(alpha=0.1)(predictions)
#attention_probs = Dense(1000, activation='softmax', name='attention_vec')(predictions)
#predictions = merge([predictions, attention_probs], output_shape=32, name='attention_mul', mode='mul')
predictions1 = Dense(7)(predictions)
checkpoint = ModelCheckpoint('./CNN1D/weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
adam = Adam(lr=0.01, decay=0.3)
model_loss= custom_loss(penalty_parameter=1.5, batch_size=30)
model = Model(inputs=[inputs, auxiliary_input], outputs=predictions1)
model.compile(optimizer=adam, loss=model_loss, metrics=['accuracy'])
model.fit([X_train, T], y_train, epochs=200, callbacks=[checkpoint], batch_size=30, validation_data=([X_test,K], y_test))



##################################################3 for class weights

class_weights = class_weight.compute_class_weight('balanced', np.unique(y_ints), y_ints)
y_ints = [y.argmax() for y in y_train]
class_weight_dict = dict(enumerate(class_weights))


import numpy as np
import math

# labels_dict : {ind_label: count_label}
# mu : parameter to tune 

def create_class_weight(labels_dict,mu=1.5):
	total = np.sum(labels_dict.values())
	keys = labels_dict.keys()
	class_weight = dict()
	for key in keys:
		score = math.log(mu*total/float(labels_dict[key]))
		class_weight[key] = score if score > 1.0 else 1.0
	return class_weight

lbl_dict_count1={}

class_weights=create_class_weight(lbl_dict_count1)


for word in test_label:
	t=lbl_dict.get(word)
	if t not in lbl_dict_count1:
		lbl_dict_count1[t]=1
	else:
		k=lbl_dict_count1.get(t)
		k=k+1
		lbl_dict_count1[t]=k
		
		
		
#####################################

def myloss(y_true, y_pred, batch_size, penalty_parameter, class_weights):
    print(y_true.shape)
    regularization_loss = tf.reduce_mean(tf.square(model.layers[len(model.layers)-1].get_weights()[0]))
    print("done ##################")
    hinge_loss = tf.square(tf.maximum(tf.zeros([batch_size, 7]), 1 - y_true * y_pred))
    print("done#########################3" + batch_size)
    y_true1 = tf.placeholder(tf.float32, shape=([batch_size, 7]))
    y_pred1 = tf.placeholder(tf.float32, shape=([batch_size, 7]))
    sess = tf.Session()
    with sess.as_default():
    	y_true1=tf.identity(y_true)
    	y_pred1=tf.identity(y_pred)
    	print(y_true1.eval())
    	print(y_pred1.eval())
    
    y_ints = [y.argmax() for y in y_true1]
    y_ints1 = [y.argmax() for y in y_pred1]
    extra = tf.placeholder(tf.float32, shape=(batch_size))
    extra1=[]
    for i in range(0,len(y_ints)):
    	if(y_ints[i]!=y_ints1[i]):
    		k=class_weights.get(y_ints[i])
    		extra1[i]=k
    	else:
    		extra1[i]=1
    hinge_loss = tf.multiply(hinge_loss, extra)
    with tf.Session() as sess:
    	sess.run(hinge_loss, feed_dict={extra: extra1})
    hinge_loss = tf.reduce_mean(hinge_loss)	
    loss = regularization_loss + penalty_parameter * hinge_loss
    print("returning loss##############")
    return loss

def custom_loss(batch_size, penalty_parameter, class_weights):
    
    def new_loss(y_true, y_pred):
    	#h=h
    	return myloss(y_true, y_pred, batch_size, penalty_parameter, class_weights)
    return new_loss



################################## model


print("Creating Model...")
inputs = Input(shape=(sequence_length,), dtype='int32')
embedding = Embedding(input_dim=len(word_index)+1, output_dim=embedding_dim, weights=[embedding_matrix], input_length=sequence_length)(inputs)
conv_0 = Conv1D(filters=16, kernel_size=5, padding='same', kernel_initializer='normal')(embedding)
conv_0 = LeakyReLU(alpha=0.1)(conv_0)
maxpool_0 = MaxPooling1D(pool_size=2)(conv_0)
#t= BatchNormalization()(maxpool_0)
#dropout = Dropout(0.1)(maxpool_0)
#flatten = Flatten()(maxpool_0)
conv_1 = Conv1D(filters=16, kernel_size=4, padding='same', kernel_initializer='normal')(embedding)
conv_1 = LeakyReLU(alpha=0.1)(conv_1)
maxpool_1 = MaxPooling1D(pool_size=2)(conv_1)
#flatten1 = Flatten()(maxpool_1)
conv_2 = Conv1D(filters=16, kernel_size=3, padding='same', kernel_initializer='normal')(embedding)
conv_2 = LeakyReLU(alpha=0.1)(conv_2)
maxpool_2 = MaxPooling1D(pool_size=2)(conv_2)
x = Concatenate()([maxpool_0, maxpool_1, maxpool_2])
#flatten2 = Flatten()(maxpool_2)
#layer_1 = TimeDistributed(Dense(32), input_shape=(sequence_length, embedding_dim))(conv_0)
layer_2 = Bidirectional(LSTM(300, return_sequences=True))(x)
#layer_2 = Bidirectional(LSTM(300))(x)
layer_2 = SeqSelfAttention(attention_width=20, attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL, attention_activation=None, kernel_regularizer=l2(0.1), use_attention_bias=False, name='Attention')(layer_2)
layer_2 = Flatten()(layer_2)
#layer_2 = Dense(256)(layer_2)
#layer_2 = LeakyReLU(alpha=0.1)(layer_2)
auxiliary_input = Input(shape=(feature_length,), name='aux_input')
#x = Concatenate()([layer_2, flatten])
#x = Dense(1000)(x)
#x = LeakyReLU(alpha=0.1)(x)
x = Concatenate()([auxiliary_input, layer_2])
predictions = Dense(1000)(x)
predictions = LeakyReLU(alpha=0.1)(predictions)
#attention_probs = Dense(1000, activation='softmax', name='attention_vec')(predictions)
#predictions = merge([predictions, attention_probs], output_shape=32, name='attention_mul', mode='mul')
predictions1 = Dense(7)(predictions)
checkpoint = ModelCheckpoint('./CNN1D/weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
adam = Adam(lr=0.01, decay=0.3)
model_loss= custom_loss(penalty_parameter=1.5, batch_size=30, , class_weights=class_weights)
model = Model(inputs=[inputs, auxiliary_input], outputs=predictions1)
model.compile(optimizer=adam, loss=model_loss, metrics=['accuracy'])
model.fit([X_train, T], y_train, epochs=200, callbacks=[checkpoint], batch_size=30, validation_data=([X_test,M], y_test))




##########################best model


[[107   5  49   3   2   0   5]
 [  1  47  13   0   0   1   0]
 [ 28  10 314   4   2   5   3]
 [  3   0  18  14   0   0   0]
 [  3   2  17   0  14   2   0]
 [  4   1   3   0   1   9   1]
 [ 11   0   2   0   2   0  18]]

