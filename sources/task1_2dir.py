#Generic libraries
from __future__ import print_function, division
import sys
import numpy as np
np.random.seed(1337)
import utils
import pickle


#Keras libraries
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, Dropout, Merge
from keras.layers import LSTM


MAX_SEQUENCE_LENGTH = 100
MAX_NB_WORDS = 27500

# training parameters
epochs = 10
b_size = 16
n_splits = 5
val_split = 0.2
dropout = 0.7
lstm_units = 20
embedding_dim = 50
dense_units = 25

#Load parameters from command line
#b_size, dropout, lstm_units, dense_units = int(sys.argv[len(sys.argv)-4]), float(sys.argv[len(sys.argv)-3]), int(sys.argv[len(sys.argv)-2]), int(sys.argv[len(sys.argv)-1])

# Load dataset
texts, labels = utils.load_dataset()
embeddings_index = utils.load_word_vectors(filename='glove.6B.' + str(embedding_dim) + "d.txt")


print("----------------------------------------\nPARAMETERS")
print("Epochs, Batch_size, Dropout, LSTM_Units, Embedding_dim, Dense_units")
print(epochs, b_size, dropout, lstm_units, embedding_dim, dense_units)
print("----------------------------------------")

# Vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
#data = pad_sequences(sequences, maxlen=None)
labels = np.asarray(labels, dtype='int64')
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
nb_validation_samples = int(val_split * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_test = data[-nb_validation_samples:]
y_test = labels[-nb_validation_samples:]

# dataset generalization
# positivi = 1; negativi = 0;
x_pos, x_neg = utils.build_vectors(x_train, y_train)
y_neg = utils.init_labels(0, len(x_neg))
y_pos = utils.init_labels(1, len(x_pos))
x_train_n = np.concatenate((x_pos, x_neg), axis=0)
y_train_n = np.concatenate((y_pos, y_neg), axis=0)
indexs = np.arange(x_train_n.shape[0])
np.random.shuffle(indexs)
x_train_n = x_train_n[indexs]
y_train_n = y_train_n[indexs]


# split the data into a training set and a validation set
#skf = StratifiedKFold(n_splits=n_splits)



# for train_index, test_index in skf.split(X_train, Y_train):
#     print("k-fold iteration: ", itera)
#     print("TRAIN:", train_index, "TEST:", test_index)
#     x_train, x_val = X_train[train_index], X_train[test_index]
#     y_train, y_val = Y_train[train_index], Y_train[test_index]

print('Preparing embedding matrix.')

# prepare embedding matrix
#nb_words = min(MAX_NB_WORDS, len(word_index))

nb_words = MAX_NB_WORDS
embedding_matrix = np.zeros((nb_words + 1, embedding_dim))
for word, iter in word_index.items():
     if iter > MAX_NB_WORDS:
         continue
     embedding_vector = embeddings_index.get(word)
     if embedding_vector is not None:
         # words not found in embedding index will be all-zeros.
         embedding_matrix[iter] = embedding_vector

#Scale embedding matrix in range [-1,1]
print('Scaling GloVe features..')
embedding_matrix1 = embedding_matrix 
a, b = -1, 1
min_value, max_value = np.min(embedding_matrix), np.max(embedding_matrix)
embedding_matrix = (((b - a) * (embedding_matrix - min_value))/(max_value - min_value)) + a


# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
embedding_layer = Embedding(nb_words + 1, embedding_dim, weights=[embedding_matrix], trainable=True)

print('Training model..')
# Create the model
left = Sequential()
left.add(embedding_layer)
left.add(Dense(dense_units, activation='relu'))
left.add(Dropout(dropout))
left.add(LSTM(lstm_units, dropout_W=dropout, dropout_U=dropout))  # base

right = Sequential()
right.add(embedding_layer)
right.add(Dense(dense_units, activation='relu'))
right.add(Dropout(dropout))
right.add(LSTM(lstm_units, dropout_W=dropout, dropout_U=dropout, go_backwards=True))  # base

model = Sequential()
model.add(Merge([left, right], mode='sum'))
model.add(Dense(1, activation='linear'))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy', 'precision', 'recall', 'fmeasure'])

#print(model.summary())
h = model.fit(x_train_n, y_train_n, nb_epoch=epochs, batch_size=b_size, validation_data=(x_test, y_test))

# Final evaluation of the model
scores = model.evaluate(x_test, y_test, verbose=0)
# Compute measures
print("Loss: ", (scores[0]))
print("Accuracy: %.2f%%" % (scores[1] * 100))
print("Precision: %.2f%%" % (scores[2] * 100))
print("Recall: %.2f%%" % (scores[3] * 100))
print("F-measure: %.2f%%" % (scores[4] * 100))

# Save on file
with open('batch_size'+ str(b_size) +'_dropout'+str(dropout*100)+ '_lstm_units'+str(lstm_units)+'_dense_units'+str(dense_units)+ '.pickle', 'w') as f:
    pickle.dump([h.history, scores, indices], f)



