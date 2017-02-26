import os
import numpy as np


def load_word_vectors(dir='./glove_features/', filename='glove_vectors_100d.txt'):

    print('Indexing word vectors.')

    embeddings_index = {}
    f = open(os.path.join(dir, filename))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Found %s word vectors.', len(embeddings_index))

    return embeddings_index


def load_dataset(name='./dataset/wikipedia.txt', labelsname='./dataset/wikipedia_labels.txt'):

    print('Processing text dataset')

    rows = open(name, 'r')
    label = open(labelsname, 'r')
    texts = rows.readlines()
    labels = label.readlines()

    print('Found %s texts.' % len(texts))

    return texts, labels


def load_dataset_imdb():
	print('Loading data...')
	(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=max_features)
	print(len(X_train), 'train sequences')
	print(len(X_test), 'test sequences')
	return X_train, y_train, X_test, y_test

def build_vectors(texts, labels, positivi = 1, negativi = 0):
    posi = (labels == positivi).sum()
    nega = (labels == negativi).sum()
    #print('positivi:', posi)
    #print('negativi:', nega)

    # suppongo nega > posi

    # in pratica vettori y sono inutili, dato che dividiamo texts in positivi e negativi
    x_pos_temp = []
    x_neg = []
    # riempio le liste x_pos_temp e x_neg usando texts
    for i in range(len(labels)):
        if (labels[i] == 0):
            x_neg.append(texts.__getitem__(i))
        else:
            x_pos_temp.append(texts.__getitem__(i))

    x_pos = []
    # nella prima parte x_pos= x_pos_temp
    for i in range(len(x_pos_temp)):
        x_pos.append(x_pos_temp.__getitem__(i))

    # gioco con gli indici per riempire x_pos in modo casuale
    indices = np.arange(len(x_pos_temp))
    #print("Ordine base degli indici", indices)
    np.random.shuffle(indices)
    #print("nuovo ordine degli indici", indices)

    while (len(x_pos) < len(x_neg)):
        for indice in indices:
            x_pos.append(x_pos_temp.__getitem__(indice))
        np.random.shuffle(indices)
        #print("nuovo ordine degli indici", indices)

    return x_pos, x_neg

def init_labels(valore, lun):
    vect = np.zeros(lun, dtype='int64')
    for i in range(lun):
        vect[i] = valore
    return vect

def build_minibatches(x_pos,x_neg, y_pos, y_neg, bs=10):
    num_ite = len(y_neg)//(bs/2)
    #print num_ite
    for i in range(num_ite):
        left_i, right_i = i*bs/2, (i+1)*bs/2
        X_batch = x_neg[left_i:right_i] + x_pos[left_i:right_i]
        y_batch = np.concatenate((y_neg[left_i:right_i], y_pos[left_i:right_i]))
        yield np.asarray(X_batch), y_batch

#def crea_batch(texts, labels)
#    positivi = 1; negativi = 0;
#    x_pos, x_neg = build_vectors(texts, labels, positivi, negativi)
#    y_neg = inzia_labels(negativi, len(x_neg))
#    y_pos = inzia_labels(positivi, (len(x_pos)))

    #X_train,Y_train= build_vectors_training(texts, labels, positivi, negativi)
#    for x,y in build_minibatches(x_pos, x_neg, y_pos, y_neg, 10):
#	print x,y

#####---LOAD DATSET WITH WORDS######
def load_dataset_with_cue(max_col=50, name='./dataset/wikipedia.txt', labelsname='./dataset/wikipedia_labels.txt'):
    print('Processing text dataset')
    rows = open(name, 'r')
    label = open(labelsname, 'r')
    texts = rows.readlines()
    labels = label.readlines()
    print('Found %s texts.' % len(texts))
    labels = crea_labels(labels, max_col)
    return texts, labels

def crea_labels(labels, max_col=25):
    y_new = np.zeros((len(labels), max_col), dtype='int64')
    for i in range(len(labels)):
        riga = np.zeros(max_col)
        stringa = labels[i]
	# number_of_characters = sum(x is not ' ' for x in ss)
        ss = stringa.split(' ')
        for j in range(len(ss)-1):
	    if(j >= max_col):
		break
            riga[j] = int(ss[j])
        y_new[i] = riga
    return y_new

#####BUILD NEW x_pos, x_neg, y_pos, y_neg
def count_type(vettore, positivi = 1, negativi = 0):
    conta_p = 0
    conta_n = 0
    for cella in vettore:
        if cella.any() == positivi:
            conta_p = conta_p + 1
        else:
            conta_n = conta_n + 1
    return conta_n, conta_p

def build_vectors_x_y(texts, labels, positivi = 1, negativi = 0):
    #posi = (labels == positivi).sum()
    #nega = (labels == negativi).sum()
    nega, posi = count_type(labels, positivi, negativi)
    print('1: positivi:', posi)
    print('0: negativi:', nega)

    # suppongo nega > posi
    # certo=0 > incerti=1

    # in pratica vettori y sono inutili, dato che dividiamo texts in positivi e negativi
    x_pos_temp = []
    y_pos_temp = []
    x_neg = []
    y_neg = np.zeros((nega, len(labels[0])), dtype='int64')
    # riempio le liste x_pos_temp e x_neg usando texts
    for i in range(len(labels)):
        if (labels[i].any() == positivi):
            x_pos_temp.append(texts.__getitem__(i))
            y_pos_temp.append(labels.__getitem__(i))
        else:
            x_neg.append(texts.__getitem__(i))


    x_pos = []
    y_pos = []
    # nella prima parte x_pos= x_pos_temp e lo stesso per y
    for i in range(len(x_pos_temp)):
        x_pos.append(x_pos_temp.__getitem__(i))
        y_pos.append(y_pos_temp.__getitem__(i))

    # gioco con gli indici per riempire x_pos in modo casuale
    indices = np.arange(len(x_pos_temp))
    #print("Ordine base degli indici", indices)
    np.random.shuffle(indices)
    #print("nuovo ordine degli indici", indices)

    while (len(x_pos) < len(x_neg)):
        for indice in indices:
            x_pos.append(x_pos_temp.__getitem__(indice))
            y_pos.append(y_pos_temp.__getitem__(indice))
        np.random.shuffle(indices)
        #print("nuovo ordine degli indici", indices)

    return x_pos, x_neg, y_pos, y_neg



