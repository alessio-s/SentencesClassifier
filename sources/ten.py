from __future__ import division
import sys
import numpy as np
import pickle

def load_labels_with_cue(indices, labelsname='./dataset/esteso_wikipedia_labels.txt'):
    #print('Processing text dataset with cues')
    label = open(labelsname, 'r')
    labels = label.readlines()
    #print('Found %s texts.' % len(labels))
    labels = [np.fromstring(x.replace('\n',''), dtype=int, sep=' ') for x in labels]
    labels = mixture(labels, indices)
    labels = labels[-int(0.2*len(labels)):]
    return labels

def mixture(labels, indices):
	tmp = labels
	nuovo = []
	for i in range(len(indices)):
		nuovo.append(tmp[indices[i]])
	return nuovo


#dato indice, restituisce il suo corrispondente nel dizionario
def get_word(word_index, valore):
	for chiave in word_index.keys():
		if(word_index[chiave]==valore):
			return chiave


#converte un vettore di int(key nel dizionario) in una frase
def converti(word_index, vettore):
	stringa = ''
	for valore in vettore:
		if valore > 0:
			stringa = stringa + str(get_word(word_index, valore)) + ' '
			#sys.stdout.write(get_word(valore) + ' ')
	return stringa

#restituisce un vettore di T e F, che vale T se la predizione e' correttta
def controllo(last_out, y_test, soglia=0.55):
	_query = (last_out.ravel() > soglia)*1
	return (_query == y_test)

#estrae dal "vettore" del layer(su cui voglio lavorare) i risultati corretti, e i rispettivi indici in x_test
def extract_datas(vettore, last_out, y_test, soglia=0.55):
	aa = []
	indexes = []
	ris = controllo(last_out, y_test, soglia)
	for i in range(len(ris)):
		if(ris[i]==True):
			aa.append(vettore[i])
			indexes.append(i)
	return aa, indexes

#lstm_out, x_test sono una riga specifica del loro totale
#stampa le parole della frase che sono sopra la soglia
def print_cue_words(word_index, x_test, lstm_out, soglia= 0.05):
	indici = get_indexes_cue_words(lstm_out,soglia)
	for ind in indici:
		sys.stdout.write(str(ind)+ ':' +str(get_word(word_index, x_test[ind])) + ' ')

#restituisce indici delle le parole della frase che sono sopra la soglia
def get_indexes_cue_words(lstm_out, soglia= 0.5):
	indici = []
	for i in range(len(lstm_out)):
		if lstm_out[i]>=soglia:
			indici.append(i)
	return indici

#stampa la frase e la corrsipondente parte cue
def get_cues(word_index, x_test, y_test, last_out, lstm_out, soglia_target = 0.55, soglia_cue=0.02, righe=20):
	frasi, index = extract_datas(lstm_out, last_out, y_test, soglia_target)
	valori = 0
	for i in index:
		valori = valori +1
		print('\n-----------', y_test[i])
		print(converti(word_index, x_test[i]))
		print_cue_words(word_index, x_test[i], lstm_out[i], soglia_cue)
		if(valori>righe):
			break

#cerca indice della prima frase del tipo 0 oppure 1
def get_first_index_type(indici, y_test, value):
	for i in indici:
		if(y_test[i]==value):
			return i
#restituisce indice del primo elemento non nullo della frase
def start_from_here(vector):
	for i in range(len(vector)):
		if(vector[i]>0):
			return i

def buils_labels_with_cue_predictions(word_index, x_test, y_test, last_out, lstm_out, soglia_target = 0.55, soglia_cue=0.5, righe=10, sentence_filename='./f.txt', cues_filename='./c.txt', labels_filename='./l.txt'):
	frasi_lstm, index = extract_datas(lstm_out, last_out, y_test, soglia_target)
	valori = 0
	lab_pred = []
	for i in index:
		valori = valori +1
		#print('\n-----------', y_test[i])
		#print(converti(word_index, x_test[i]))
		index_cue_words = get_indexes_cue_words(lstm_out[i],soglia_cue)
		#print(start_from_here(x_test[i]))
		first_word_index = start_from_here(x_test[i])
		riga = []
		if first_word_index is not None:
			for j in range(first_word_index, len(lstm_out[i])):
				if(j in index_cue_words):
					riga.append(1)
				else:
					riga.append(0)
			lab_pred.append(np.asarray(riga, dtype=int))
			#lab_pred.append(riga)
		else:
			#print('indice con contenuto nullo', i)
			#print('\n-----------', y_test[i])
			#print(converti(word_index, x_test[i]))
			lab_pred.append([])
		#print(riga)
		
		#for icw in index_cue_words:
			#sys.stdout.write(str(icw)+ ':' +str(get_word(word_index, x_test[i][icw])) + ' ')
		#if(valori>righe):
			#break
	return lab_pred

def measures_eval(ground_truth, predicted):
	#from __future__ import division
	if not ground_truth or not predicted:
		print('Ground truth or predicted lists are empty!')
		return
	if not (len(ground_truth) == len(predicted)):
		print('Lists doesnt have the same length.')
		print(len(ground_truth), len(predicted))
		return
	
	tp, fp, tn, fn = 0, 0, 0, 0
	for index in xrange(len(ground_truth)):
		gt 	= ground_truth[index]
		pred 	= predicted[index]
		#if (len(gt) > 0):
			#print('Sentences on position ' + str(index) + ' doesnt have the same length: ' + str(gt.shape) + ', ' + str(pred.shape))
			#return
		
		for sen_index in xrange(min(len(gt), len(pred))):
			if (pred[sen_index] == 1 and gt[sen_index] == 1):
				tp = tp + 1
			elif (pred[sen_index] == 1 and gt[sen_index] == 0):
				fp = fp + 1
			elif (pred[sen_index] == 0 and gt[sen_index] == 1):
				fn = fn + 1
			else:
				tn = tn + 1
	print(tp, fp, tn, fn)
	accuracy  = (tp + tn) / (tp + tn + fp + fn)
	try:
		precision = tp / (tp + fp)
		recall	  = tp / (tp + fn)
		fscore	  = 2.0 * (precision * recall) / (precision + recall)
	except:
		precision = recall = fscore =0	
	return accuracy, precision, recall, fscore

def tentativi(x_test, y_test, indices, last_out, lstm_out, word_index, soglia_target=0.55, soglia_cue=0.55):
	gt = load_labels_with_cue(indices)
	aa, index = extract_datas(lstm_out, last_out, y_test, soglia_target)
	gt_giuste = mixture(gt, index)

	predetti = buils_labels_with_cue_predictions(word_index, x_test, y_test, last_out, lstm_out, soglia_target, soglia_cue)

	accuracy, precision, recall, fscore = measures_eval(gt_giuste, predetti)

	return accuracy, precision, recall, fscore

def save_on_file(filename, matrix):
	with open(filename, 'w') as f:
		pickle.dump([matrix], f)

def save_tentativi(x_test, y_test, indices, last_out, lstm_out, word_index): 
	#import numpy as nmpy
	soglia_target = np.arange(0.44,0.66, 0.01)
	#soglia_target = [0.55]
	soglia_cue = np.arange(0.1,0.9, 0.01)
	dim = len(soglia_target) * len(soglia_cue)
	results = np.zeros((dim, 6), dtype='float32')
	i = 0
	for st in soglia_target:
		for sc in soglia_cue:
			print('Iteration ' + str(i+1) + ' out of ' + str(dim) + '...')
			accuracy, precision, recall, fscore = tentativi(x_test, y_test, indices, last_out, lstm_out, word_index, st, sc)
			results[i, :] = [st, sc, accuracy, precision, recall, fscore]
			i = i + 1
	
	#return results
	#save_on_file('./task2_2dir_1_1_8.pickle', results)
	return results


def load_cue_score(filename):
	with open(filename, 'r') as f:
		matrix = pickle.load(f)
		#for riga in matrix:
		#	print(riga.flatten())
		return matrix

#restituisce due vettori, draw_result=contiene i valori di uscita dal lstm per
#una determinata frase, type_result= indica la sua etichetta nel dataset originale
def build_draw_datas(soglia_cue, x_test, index, lstm_out, y_test, word_index, filename='./grafici_cue.pickle'):
	draw_result = []
	type_result = []
	draw_result_thresholded = []
	sentence = []
	i=1
	for indice in index:
		print('Index ' + str(i) + ' out of ' + str(len(index)) + '..')
		i = i + 1
		posizione = start_from_here(x_test[indice])
		draw_result.append(lstm_out[indice][posizione:])
		type_result.append(y_test[indice])
		frase = lstm_out[indice][posizione:]
		frase = (frase.ravel() > soglia_cue)*1
		draw_result_thresholded.append(frase)	
		sentence.append(converti(word_index, x_test[indice]))
	return draw_result, type_result, draw_result_thresholded, sentence

def best_fscore(x_test, y_test, indices, last_out, lstm_out, word_index, filename='./grafici_cue_5type.pickle', filename1='./task2_1dir_best_fscoreParameters.pickle'):	
	gt = load_labels_with_cue(indices)
	results = save_tentativi(x_test, y_test, indices, last_out, lstm_out, word_index)	
	best_index = np.argmax(results[:,5])
	print(best_index, np.max(results[:, 5]))
	st,sc,a,p,r,fs = results[best_index]	
	aa, index = extract_datas(lstm_out, last_out, y_test, st)
	gt_giuste = mixture(gt, index)
	print('Building data for draw..')
	draw_result, type_result, draw_result_thresholded, sentence = build_draw_datas(sc, x_test, index, lstm_out, y_test, word_index)
	print('Save on file..')
	with open(filename, 'w') as f:
		pickle.dump([draw_result, type_result, draw_result_thresholded, gt_giuste, sentence], f)
	with open(filename1, 'w') as f1:
		pickle.dump([st,sc,a,p,r,fs], f1)
	print(st,sc,a,p,r,fs)
	print('Done.')

def best_fscore_concat(x_test, y_test, indices, last_out, lstm_out, word_index, filename='./grafici_cue_5type.pickle', filename1='./task2_2dir_best_fscoreParameters.pickle'):	
	gt = load_labels_with_cue(indices)
	#fscore del primo
	lstm_out_primo = lstm_out[:,:,0]
	results1 = save_tentativi(x_test, y_test, indices, last_out, lstm_out_primo, word_index)
	best_index_primo = np.argmax(results1[:,5])
	print(best_index_primo, np.max(results1[:, 5]))
	st1,sc1,a1,p1,r1,fs1 = results1[best_index_primo]
	print('prima posizione:', st1,sc1,a1,p1,r1,fs1)
	#fscore del secondo
	lstm_out_secondo = lstm_out[:,:,0]
	results2 = save_tentativi(x_test, y_test, indices, last_out, lstm_out_secondo, word_index)
	best_index_secondo = np.argmax(results2[:,5])
	print(best_index_secondo, np.max(results2[:, 5]))
	st2,sc2,a2,p2,r2,fs2 = results2[best_index_secondo]
	print('seconda posizione:', st2,sc2,a2,p2,r2,fs2)
	if(fs1>fs2):
		st,sc,a,p,r,fs = st1,sc1,a1,p1,r1,fs1
		aa, index = extract_datas(lstm_out_primo, last_out, y_test, st)
		gt_giuste = mixture(gt, index)
		print('Building data for draw..')
		draw_result, type_result, draw_result_thresholded, sentence = build_draw_datas(sc, x_test, index, lstm_out_primo, y_test, word_index)
		print('Save on file..')
		with open(filename, 'w') as f:
			pickle.dump([draw_result, type_result, draw_result_thresholded, gt_giuste, sentence], f)
		with open(filename1, 'w') as f1:
			pickle.dump([st,sc,a,p,r,fs], f1)
		print(st,sc,a,p,r,fs)
		print('Done.')
	else:
		st,sc,a,p,r,fs = st2,sc2,a2,p2,r2,fs2
		aa, index = extract_datas(lstm_out_secondo, last_out, y_test, st)
		gt_giuste = mixture(gt, index)
		print('Building data for draw..')
		draw_result, type_result, draw_result_thresholded, sentence = build_draw_datas(sc, x_test, index, lstm_out_secondo, y_test, word_index)
		print('Save on file..')
		with open(filename, 'w') as f:
			pickle.dump([draw_result, type_result, draw_result_thresholded, gt_giuste, sentence], f)
		with open(filename1, 'w') as f1:
			pickle.dump([st,sc,a,p,r,fs], f1)
		print(st,sc,a,p,r,fs)
		print('Done.')

