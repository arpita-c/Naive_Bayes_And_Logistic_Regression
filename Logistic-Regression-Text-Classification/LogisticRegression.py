from __future__ import division
from __future__ import print_function
import os
import sys
import math
import re
import collections
import copy
import ast



spam_ham_training_set = dict()
spam_ham_test_set = dict()
filtered_spam_ham_training_set = dict()
filtered_spam_ham_test_set = dict()

stop_words = []
training_set_vocab = []
filtered_training_set_vocab = []


weights = {'weight': 0.0}
filtered_weights = {'weight': 0.0}


classes = ["ham", "spam"]


learning_constant = .001
number_of_iterations = 100
lambda_constant = 0.0



class Document:
	text = ""
	true_class = ""
	learned_class = ""
   
	word_freqs = {'weight': 1.0}

	
	def __init__(self, text, counter, true_class):
		self.text = text
		self.word_freqs = counter
		self.true_class = true_class

	def getText(self):
		return self.text

	def getWordFreqs(self):
		return self.word_freqs

	def getTrueClass(self):
		return self.true_class

	def getLearnedClass(self):
		return self.learned_class

	def setLearnedClass(self, guess):
		self.learned_class = guess



def buildData(storage_dict, directory, true_class):
	for dir_entry in os.listdir(directory):
		dir_entry_path = os.path.join(directory, dir_entry)
		if os.path.isfile(dir_entry_path):
			with open(dir_entry_path, 'r') as text_file:
				text = text_file.read()
				storage_dict.update({dir_entry_path: Document(text, bagOfWords(text), true_class)})


def getDataVocabulary(data_set):
	vocab = []
	for i in data_set:
		for j in data_set[i].getWordFreqs():
			if j not in vocab:
				vocab.append(j)
	return vocab


def getStopWords():
	stop_words = []
	with open('stop_words.txt', 'r') as txt:
		stop_wprds = (txt.read().splitlines())
	return stop_words


def throwAwayStopWords(stop_words, data_set):
	filtered_data_set = copy.deepcopy(data_set)
	for i in stop_words:
		for j in filtered_data_set:
			if i in filtered_data_set[j].getWordFreqs():
				del filtered_data_set[j].getWordFreqs()[i]
	return filtered_data_set


def bagOfWords(text):
	bagsofwords = collections.Counter(re.findall(r'\w+', text))
	return dict(bagsofwords)


def learnWeights(training, weights_param, iterations, lamda_val):
	for x in range(0, iterations):
		print(x)
		counter = 1
		for w in weights_param:
			sum = 0.0
			for i in training:
				y_sample = 0.0
				if training[i].getTrueClass() == classes[1]:
					y_sample = 1.0
				if w in training[i].getWordFreqs():
					sum += float(training[i].getWordFreqs()[w]) * (y_sample - computeConditionalProbability(classes[1], weights_param, training[i]))
			weights_param[w] += ((learning_constant * sum) - (learning_constant * float(lamda_val) * weights_param[w]))




def computeConditionalProbability(class_prob, weights_param, doc):
	if class_prob == classes[0]:
		sum_weights_0 = weights_param['weight']
		for i in doc.getWordFreqs():
			if i not in weights_param:
				weights_param[i] = 0.0
			sum_weights_0 += weights_param[i] * float(doc.getWordFreqs()[i])
		return 1.0 / (1.0 + math.exp(float(sum_weights_0)))

	elif class_prob == classes[1]:
		sum_weights_1 = weights_param['weight']
		for i in doc.getWordFreqs():
			if i not in weights_param:
				weights_param[i] = 0.0
			sum_weights_1 += weights_param[i] * float(doc.getWordFreqs()[i])
		return math.exp(float(sum_weights_1)) / (1.0 + math.exp(float(sum_weights_1)))

def LogisticRegression(data_instance, weights_param):
	score = {}
	score[0] = computeConditionalProbability(classes[0], weights_param, data_instance)
	score[1] = computeConditionalProbability(classes[1], weights_param, data_instance)
	if score[1] > score[0]:
		return classes[1]
	else:
		return classes[0]


def main():
	args = str(sys.argv)
	args = ast.literal_eval(args)

	if (len(args) < 6):
		print( "You have input less than the minimum number of arguments.")
		print("Usage: .\program <spam-training-dir> <ham-training-dir> <spam-test-dir> <ham-test-dir> <lambda-value>")
	elif (not os.path.isdir(args[1]) and not os.path.isdir(args[2]) and not os.path.isdir(args[3]) and not os.path.isdir(args[4])):
		print("First 4 arguments must be directories of spam/ham mails!")
	else:

		correct_predictions = 0.0
		filtered_correct_predictions = 0.0

		spam_training_dir = str(args[1])
		ham_training_dir = str(args[2])
		spam_test_dir = str(args[3])
		ham_test_dir = str(args[4])
		lambda_constant = float(args[5])

		stop_words = getStopWords()

		buildData(spam_ham_training_set, spam_training_dir, classes[1])
		buildData(spam_ham_training_set, ham_training_dir, classes[0])
		buildData(spam_ham_test_set, spam_test_dir, classes[1])
		buildData(spam_ham_test_set, ham_test_dir, classes[0])

		filtered_spam_ham_training_set = throwAwayStopWords(stop_words, spam_ham_training_set)
		filtered_spam_ham_test_set = throwAwayStopWords(stop_words, spam_ham_test_set)

		vocabulary_spam_ham_training_set = getDataVocabulary(spam_ham_training_set)
		filtered_vocalbulary_spam_ham_training_set = getDataVocabulary(filtered_spam_ham_training_set)

		#{weights[i]:0.0 for i in vocabulary_spam_ham_training_set}
		#{filtered_weights[i]:0.0 for i in filtered_vocalbulary_spam_ham_training_set}

		for i in vocabulary_spam_ham_training_set:
			weights[i] = 0.0

		for i in filtered_vocalbulary_spam_ham_training_set:
			filtered_weights[i] = 0.0

		learnWeights(spam_ham_training_set, weights, number_of_iterations, lambda_constant)
		learnWeights(filtered_spam_ham_training_set, filtered_weights, number_of_iterations, lambda_constant)

		for i in spam_ham_test_set:
			spam_ham_test_set[i].setLearnedClass(LogisticRegression(spam_ham_test_set[i], weights))
			if (spam_ham_test_set[i].getLearnedClass() == spam_ham_test_set[i].getTrueClass()):
				correct_predictions += 1.0

		for i in filtered_spam_ham_test_set:
			filtered_spam_ham_test_set[i].setLearnedClass(LogisticRegression(filtered_spam_ham_test_set[i], filtered_weights))
			if (filtered_spam_ham_test_set[i].getLearnedClass() == filtered_spam_ham_test_set[i].getTrueClass()):
				filtered_correct_predictions += 1.0

		print ("Correct guesses before filtering stop words:\t%d/%s" % (correct_predictions, len(spam_ham_test_set)))
		print ("Accuracy before filtering stop words:\t%.4f%%" % (100.0 * float(correct_predictions) / float(len(spam_ham_test_set))))
		print ()
		print ("Correct guesses after filtering stop words:\t%d/%s" % (filtered_correct_predictions, len(filtered_spam_ham_test_set)))
		print ("Accuracy after filtering stop words:\t%.4f%%" % (100.0 * float(filtered_correct_predictions) / float(len(filtered_spam_ham_test_set))))

if __name__ == '__main__':
	main()