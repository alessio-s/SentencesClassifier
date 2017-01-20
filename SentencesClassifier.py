from __future__ import print_function
import os
import numpy as np
np.random.seed(1337)

#Keras libraries
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Embedding
from keras.layers import Dropout
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

BASE_DIR = '.'
GLOVE_DIR = BASE_DIR + '/glove_features/'
MAX_SEQUENCE_LENGTH = 150
MAX_NB_WORDS = 4000		#150
EMBEDDING_DIM = 100
GLOVE_FILENAME = 'glove_vectors_' + str(EMBEDDING_DIM) + 'd.txt'

#training parameters
epoch = 10
b_size = 10
val_split = 0.2
dropout = 0.7
lstm_units = 100

# first, build index mapping words in the embeddings set
# to their embedding vector

print('Indexing word vectors.')

embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, GLOVE_FILENAME))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

# second, prepare text samples and their labels
print('Processing text dataset')

texts = []  # list of text samples
#labels_index = {}  # dictionary mapping label name to numeric id
labels = []  # list of label id

rows = open('./dataset/wikipedia.txt', 'r')
label = open('./dataset/wikipedia_labels.txt', 'r')
texts = rows.readlines()
labels = label.readlines()
print('Found %s texts.' % len(texts))

# finally, vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels = to_categorical(np.asarray(labels))
#labels = np.asarray(labels)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
nb_validation_samples = int(val_split * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_test = data[-nb_validation_samples:]
y_test = labels[-nb_validation_samples:]

print('Preparing embedding matrix.')

# prepare embedding matrix
nb_words = min(MAX_NB_WORDS, len(word_index))
embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    if i > MAX_NB_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
embedding_layer = Embedding(nb_words + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True)

print('Training model.')
# create the model

model = Sequential()
model.add(embedding_layer)
model.add(LSTM(lstm_units, activation='sigmoid', input_shape=(x_train.shape[0], x_train.shape[1]))) #base
model.add(Dropout(dropout))
model.add(Dense(y_train.shape[1], activation='sigmoid'))
#keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)	#ADADELTA: An Adaptive Learning Rate Method
model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy', 'precision', 'recall', 'fmeasure'])#mod =2


print(model.summary())
model.fit(x_train, y_train, validation_split=val_split, shuffle=True, nb_epoch=epoch, batch_size=b_size)
# Final evaluation of the model
scores = model.evaluate(x_test, y_test, verbose=0)
#nostro
y_test_new = np.argmax(y_test, axis=1)
y_train_new = np.argmax(y_train, axis=1)
preds_train = np.argmax(model.predict(x_train), axis=1)
preds_test = np.argmax(model.predict(x_test), axis=1)
acc_train = 100.*(preds_train == y_train_new).sum()/len(y_train_new)
acc_test   = 100.*(preds_test == y_test_new).sum()/len(y_test_new)
print("Train accuracy: ", acc_train)
print("Test accuracy: ", acc_test)
#scores = model.evaluate(x_test, y_test, verbose=0, batch_size=16)
print("Loss: ", (scores[0]))
print("Accuracy: %.2f%%" % (scores[1]*100))
print("Precision: %.2f%%" % (scores[2]*100))
print("Recall: %.2f%%" % (scores[3]*100))
print("F-measure: %.2f%%" % (scores[4]*100))

