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

print(lbl_dict)

tokenizer=load_create_tokenizer(train,None,True)
X_train=load_create_padded_data(X_train=train,savetokenizer=False,isPaddingDone=False,maxlen=sequence_length,tokenizer_path='./New_Tokenizer.tkn')
X_test=load_create_padded_data(X_train=test,savetokenizer=False,isPaddingDone=False,maxlen=sequence_length,tokenizer_path='./New_Tokenizer.tkn')
word_index=tokenizer.word_index
embedding_matrix=load_create_embedding_matrix(word_index,len(word_index)+1,200,'./glove.twitter.27B.200d.txt',False,True,'./Emb_Mat.mat')
#f=open("Emb_Mat.mat","rb")
#embedding_matrix=pickle.load(f)
#f.close()





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

f=open("final_feature_train.pkl","rb")
Z=pickle.load(f)
f.close()

f=open("final_feature_test.pkl","rb")
A=pickle.load(f)
f.close()

T=[]
M=[]

''''
for i in Z:
    data=i.split(" ")
    #print(len(data))
    for j in range(0,len(data)):
    	data[j]=float(data[j])
    T.append(data)
    
for i in A:
    data=i.split(" ")
    #print(len(data))
    for j in range(0,len(data)):
    	data[j]=float(data[j])
    M.append(data)
'''

for i in range(0, len(y_train)):
    	data=[]
    	data.append(float(0))
    	T.append(data)
    	
for i in range(0, len(y_test)):
    	data=[]
    	data.append(float(0))
    	M.append(data)

T=np.array([i for i in T])
M=np.array([i for i in M])

feature_length = max(len(x) for x in T)


lbl_dict_count1={}
for word in test_label:
	t=lbl_dict.get(word)
	if t not in lbl_dict_count1:
		lbl_dict_count1[t]=1
	else:
		k=lbl_dict_count1.get(t)
		k=k+1
		lbl_dict_count1[t]=k
		
def create_class_weight(labels_dict,mu=1.2):
	total = np.sum(labels_dict.values())
	keys = labels_dict.keys()
	class_weight = dict()
	for key in keys:
		score = math.log(mu*total/float(labels_dict[key]))
		class_weight[key] = score #if score > 1.0 else 1.0
	return class_weight
	
class_weights=create_class_weight(lbl_dict_count1)
cwList = []
for i in range(7):
    cwList.append(class_weights[i])
cw_tensor = tf.constant(cwList,dtype = tf.float32)


def myloss(y_true, y_pred, batch_size, penalty_parameter, class_weights):
    print(y_true.shape)
    regularization_loss = tf.reduce_mean(tf.square(model.layers[len(model.layers)-1].get_weights()[0]))
    print("done ##################")
    #y_oh = tf.one_hot(tf.argmax(y_true,1),7)
    #hinge_loss = tf.square(tf.maximum(tf.zeros(tf.shape(y_true)), 1 - y_true * y_pred))
    hinge_loss = tf.reduce_mean(tf.square(tf.maximum(tf.zeros(tf.shape(y_true)), 1 - y_true * y_pred)))
    print("done#########################3" + str(tf.shape(y_true)))
    #hinge_loss = tf.square(tf.maximum(tf.zeros([batch_size, 7]), 1 - y_true * y_pred))
    #print("done#########################3" + str(batch_size))
    '''
    y_oh = tf.one_hot(tf.argmax(y_true,1),7)
    class_expand = tf.expand_dims(class_weights,1)
    weight = tf.reduce_sum(tf.matmul(y_oh,class_expand),axis = 1)
    
    cor_mask = tf.equal(tf.argmax(y_true,1) , tf.argmax(y_pred,1))
    wr_mask = tf.logical_not(cor_mask)
    
    hinge_loss_red = tf.reduce_mean(hinge_loss,axis = 1)
    cor = hinge_loss_red * tf.cast(cor_mask , dtype = tf.float32)
    wro_m = weight * tf.cast(wr_mask, dtype = tf.float32)
    wro = wro_m *hinge_loss_red
    
    tot_hinge = tf.reduce_mean(cor+wro)
    
    
    hinge_loss = tot_hinge #= tf.reduce_mean(hinge_loss)
    '''	
    loss = regularization_loss + penalty_parameter * hinge_loss
    print("returning loss##############")
    return loss

def custom_loss(batch_size, penalty_parameter, class_weights):
    
    def new_loss(y_true, y_pred):
    	#h=h
    	return myloss(y_true, y_pred, batch_size, penalty_parameter, class_weights)
    return new_loss


print("Creating Model...")
inputs = Input(shape=(sequence_length,), dtype='int32')
embedding = Embedding(input_dim=len(word_index)+1, output_dim=embedding_dim, weights=[embedding_matrix], input_length=sequence_length)(inputs)
conv_0 = Conv1D(filters=16, kernel_size=3, padding='same', kernel_initializer='normal')(embedding)
conv_0 = LeakyReLU(alpha=0.1)(conv_0)
maxpool_0 = MaxPooling1D(pool_size=2)(conv_0)
#layer_0 = Bidirectional(LSTM(200))(maxpool_0)
#t= BatchNormalization()(maxpool_0)
#dropout = Dropout(0.1)(maxpool_0)
#flatten = Flatten()(maxpool_0)
conv_1 = Conv1D(filters=16, kernel_size=4, padding='same', kernel_initializer='normal')(embedding)
conv_1 = LeakyReLU(alpha=0.1)(conv_1)
maxpool_1 = MaxPooling1D(pool_size=2)(conv_1)
#layer_1 = Bidirectional(LSTM(200))(maxpool_1)
#flatten1 = Flatten()(maxpool_1)
conv_2 = Conv1D(filters=16, kernel_size=5, padding='same', kernel_initializer='normal')(embedding)
conv_2 = LeakyReLU(alpha=0.1)(conv_2)
maxpool_2 = MaxPooling1D(pool_size=2)(conv_2)
x = Concatenate()([maxpool_0, maxpool_1, maxpool_2])
x = Conv1D(filters=32, kernel_size=5, padding='same', kernel_initializer='normal')(embedding)
x = LeakyReLU(alpha=0.1)(x)
x = MaxPooling1D(pool_size=2)(x)
#flatten2 = Flatten()(maxpool_2)
#layer_1 = TimeDistributed(Dense(32), input_shape=(sequence_length, embedding_dim))(conv_0)
x = Bidirectional(LSTM(300))(x)
#x = Concatenate()([layer_0, layer_1, layer_2])
x = Dense(1000)(x)
#x = LeakyReLU(alpha=0.1)(x)
#layer_2 = Bidirectional(LSTM(300))(x)
#layer_2 = SeqSelfAttention(attention_width=20, attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL, attention_activation=None, kernel_regularizer=l2(0.1), use_attention_bias=False, name='Attention')(layer_2)
#layer_2 = Flatten()(layer_2)
#layer_2 = Dense(256)(layer_2)
#layer_2 = LeakyReLU(alpha=0.1)(layer_2)
auxiliary_input = Input(shape=(feature_length,), name='aux_input')
#x = Concatenate()([layer_2, flatten])
#x = Dense(1000)(x)
#x = LeakyReLU(alpha=0.1)(x)
x = Concatenate()([auxiliary_input, x])
predictions = Dense(1000)(x)
predictions = LeakyReLU(alpha=0.1)(predictions)
#attention_probs = Dense(1000, activation='softmax', name='attention_vec')(predictions)
#predictions = merge([predictions, attention_probs], output_shape=32, name='attention_mul', mode='mul')
predictions1 = Dense(7)(predictions)
checkpoint = ModelCheckpoint('./CNN1D/weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
adam = Adam(lr=0.01, decay=0.3)
model_loss= custom_loss(penalty_parameter=1.5, batch_size=64, class_weights=cw_tensor)
model = Model(inputs=[inputs, auxiliary_input], outputs=predictions1)
model.compile(optimizer=adam, loss=model_loss, metrics=['accuracy'])
model.fit([X_train, T], y_train, epochs=20, callbacks=[checkpoint], batch_size=64, validation_data=([X_test,M], y_test))
