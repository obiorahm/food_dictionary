#Creating the data set

import string
import random
from spellchecker import SpellChecker
import pandas
from keras.preprocessing.text import Tokenizer
import nltk
from keras.utils import Sequence
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Masking, Embedding 

from keras.callbacks import EarlyStopping, ModelCheckpoint
from itertools import islice

training_length  = 2

'''
for line in data_set:
	scambled_line, scrambled_word = scamble_or_omit()
	possible_words = auto_correct(scrambled_word);
	new_data_set.append([scambled_line, possible_words, line])
	'''



def get_list_of_strings():

	data = pandas.read_csv("Dish.csv", usecols=[1])
	#print(data)
	dictionary = data["name"].tolist()

	print(len(dictionary))
	file_object = open("dictionary_clean.txt","r")

	try:

		data2 = file_object.read().split('\n')
		
		#dictionary = data2;
		dictionary.extend(data2 )
		#print(len(dictionary))

		#new_dictionary = sorted(dictionary)
		new_dictionary = dictionary[:100]

		for l in new_dictionary:
			y = str(l).replace('"','').strip().lower()
			l = y

		tokenizer = Tokenizer(num_words=None, filters='!"#$%&*+,./:;<=>?@[\\]^_{|}~\t\n', lower=True, split=' ')
		tokenizer.fit_on_texts(new_dictionary)

		sequences = tokenizer.texts_to_sequences(new_dictionary)

		print("Len of sequences " + str(len(sequences)))
		print(sequences[8])

	finally:
		file_object.close()

	return sequences, tokenizer


def create_dataset(sequences):
 	features = []
 	labels = []
 	training_length = 2 
 	for sequence in sequences:

 		for i in range (training_length, len(sequence)):

 			extract = sequence[i - training_length : i + 1]

 			features.append(extract[:-1])
 			labels.append(extract[-1])
 	print (len(features))

 	return features, labels


def get_sentence_from_index(tokenizer, sequences, i, j):
	idx_word = tokenizer.word_index
	print("len of idx_word " + str(len (idx_word)))

	n_items = list(islice(idx_word.items(), 20)) #take(100, idx_word.iteritems())	

	for k in n_items:
		print(k)

	#print (' '.join(idx_word[w] for w in sequences[i][j:]))



	return idx_word



def one_hot_encode_features(idx_word, features, labels, i):
	# number of words in vocabulary
	num_words = len(idx_word) + 1

	# Empty array to hold labels 
	label_array = np.zeros((len(features), num_words), dtype = np.int8)

	#One hot encode the labels 
	for example_index, word_index in enumerate(labels):
		label_array[example_index, word_index] = 1

	print("label_array shape" + str(label_array.shape))

	# find word corresponding to encoding 
	print ("label_array: " + str(label_array[i]))
	#print (idx_word[np.argmax(label_array[i])])
	return num_words, label_array




def build_model(num_words, embedding_matrix):
	model = Sequential()

	#Embedding layer
	model.add(
		Embedding( input_dim= num_words,
			input_length = training_length,
			output_dim=100,
			weights=[embedding_matrix],
			trainable=False,
			mask_zero=True
			))

	# Masking layer for pre-trained embeddings
	model.add(Masking(mask_value=0.0))

	# Recurrent layer
	model.add(LSTM(64, return_sequences=False, dropout=0.1, recurrent_dropout=0.1))

	# Fully connected layer
	model.add(Dense(64, activation='relu'))


	#Dropout for regularization
	model.add(Dropout(0.5))


	#Output layer
	model.add(Dense(num_words, activation="softmax"))


	# Compile the model
	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


	# Create callbacks
	callbacks = [EarlyStopping(monitor='val_loss', patience=5), 
	ModelCheckpoint(filepath= 'model.h5', save_best_only=True, save_weights_only=False)]

	'''
	history = model.fit(X_train, y_train,
		batch_size=2084, epochs=150,
		callbacks=callbacks,
		validation_data=(X_valid, y_valid))
	'''

	batch_size = 32
	num_training_samples = len(X_train)
	num_validation_samples = len(X_valid)

	my_training_batch_generator = My_Generator(X_train, y_train, batch_size)
	my_validation_batch_generator = My_Generator(X_valid, y_valid, batch_size)

	model.fit_generator(generator=my_training_batch_generator,
		steps_per_epoch=(num_training_samples // batch_size),
		epochs=150,
		verbose=1,
		validation_data=my_validation_batch_generator,
		validation_steps=(num_validation_samples // batch_size),
		use_multiprocessing=True,
		workers=16,
		max_queue_size=32)


def make_embedding_matrix(num_words, word_idx):

	# Load in embeddings 
	glove_vectors = 'glove.6B.100d.txt'
	glove = np.loadtxt(glove_vectors, dtype='str', comments=None)

	#Extract the vectors and words 
	vectors = glove[:, 1:].astype('float')
	words = glove[:,0]

	# Create lookup of words to vectors 
	word_lookup = {word: vector for word, vector in zip(words, vectors)}

	# New matrix to hold word embeddings 
	embedding_matrix = np.zeros((num_words, vectors.shape[1]))

	for i, word in enumerate(word_idx.keys()):
		# Look up the word embedding 
		vector = word_lookup.get(word, None)


		# Record in matrix
		if vector is not None:
			embedding_matrix[i + 1, :] = vector

	return embedding_matrix




class My_Generator(Sequence):
		"""docstring for MY_Generator"""
		def __init__(self, features, labels, batch_size):
			self.features = features
			self.labels = labels
			self.batch_size = batch_size


		def __len__(self):
			return int(np.ceil(len(self.features)/ float(self.batch_size)))


		def __getitem__(self, idx):
			batch_x = self.features[idx * self.batch_size : (idx + 1) * self.batch_size]
			batch_y = self.labels[idx * self.batch_size : (idx + 1) * self.batch_size]

			return np.array(batch_x), np.array(batch_y)


				

def scramble_or_omit(line):
	if (line is None):
		return
	stripped_line = line.strip()
	if (len(stripped_line) == 0):
		return
	split_word = stripped_line.split(" ")
	if (len(split_word) == 0):
		return
	scambled_word = ""
	letters = string.ascii_lowercase
	print ("length of split word: " + str(len(split_word)) )	

	while (scambled_word == "" or scambled_word == " "):
		end = len(split_word) - 1
		n = random.randint(0,end);
		print("value of n: " + str(n))
		word_to_scramble = split_word[n]

		start = random.randint(0,len(word_to_scramble))
		stop = random.randint(start, len(word_to_scramble))

		substring = word_to_scramble[start:stop]

		random_substring = ''.join(random.choice(letters) for i in range(len(substring)))

		scambled_word = word_to_scramble.replace(substring, random_substring)

		split_word[n] = scambled_word

	return word_to_scramble, scambled_word, " ".join(word for word in split_word)


spell = SpellChecker(distance=2)
spell.word_frequency.load_text_file('combine_dictionary.txt')

def auto_correct (word):
	 x = spell.candidates(word)
	 return x

'''
WORD_POS = 0
SCRAMBLED_WORD_POS = 1
LINE_POS = 2


line = raw_input("enter a string: ")
print(type(line))
if (line is not None):
	print(type (scramble_or_omit(line)))
	result = scramble_or_omit(line)
	if result is not None:
		print(result[WORD_POS], result[SCRAMBLED_WORD_POS], result[LINE_POS])
		candidate_words = auto_correct(result[SCRAMBLED_WORD_POS])
		print(candidate_words)

		'''

g_sequences, g_tokenizer = get_list_of_strings()

g_features, g_labels = create_dataset(g_sequences)

print("length g_features " + str(len(g_features)))

print("length g_labels " + str(len(g_labels)))

X_train = np.array(g_features[:200])#np.array(g_features[:200])# np.array(g_features[:1473394]) #

X_valid = np.array(g_features[200:500]) # np.array(g_features[1473394:]) #


g_idx_word = get_sentence_from_index(g_tokenizer, g_features, 8, 0)

print("features 1 " + str(g_features[0]))

g_num_words, g_label_array = one_hot_encode_features(g_idx_word, g_features[:500], g_labels[:500], 0)

y_train = np.array(g_label_array[:200]) #np.array(g_label_array[:1473394])# np.array(g_label_array[:200])#
y_valid = np.array(g_label_array[200: 500]) #np.array(g_label_array[1473394:])# np.array(g_label_array[200: 400])#


g_embedding_matrix = make_embedding_matrix(g_num_words, g_idx_word)

build_model(g_num_words, g_embedding_matrix)


	









