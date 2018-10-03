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
conditional_probability = dict()
filtered_conditional_probability = dict()
prior = dict()
filtered_prior = dict()

stop_words = []
classes = ["ham", "spam"]

class Document:
	text = ""
	true_class = ""
	learned_class = ""

	word_freqs = {}

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


def bagOfWords(text):
	bagsofwords = collections.Counter(re.findall(r'\w+', text))
	return dict(bagsofwords)


def getDataVocabulary(data_set):
	all_text = ""
	v = []
	for x in data_set:
		all_text += data_set[x].getText()
	for y in bagOfWords(all_text):
		v.append(y)
	return v


def getStopWords():
	stops = []
	with open('stop_words.txt', 'r') as txt:
		stops = (txt.read().splitlines())
	return stops


def throwAwayStopWords(stops, data_set):
	filtered_data_set = copy.deepcopy(data_set)
	for i in stops:
		for j in filtered_data_set:
			if i in filtered_data_set[j].getWordFreqs():
				del filtered_data_set[j].getWordFreqs()[i]
	return filtered_data_set


def trainMultinomialNB(training, priors, cond):
		v = getDataVocabulary(training)
		n = len(training)
		for c in classes:
			n_c = 0.0
			text_c = ""
			for i in training:
				if training[i].getTrueClass() == c:
					n_c += 1
					text_c += training[i].getText()
			priors[c] = float(n_c) / float(n)
			token_freqs = bagOfWords(text_c)
			for t in v:
				if t in token_freqs:
					cond.update({t + "_" + c: (float((token_freqs[t] + 1.0)) / float((len(text_c) + len(token_freqs))))})
				else:
					cond.update({t + "_" + c: (float(1.0) / float((len(text_c) + len(token_freqs))))})


def applyMultinomialNB(data_instance, priors, cond):
	score = {}
	for c in classes:
		score[c] = math.log10(float(priors[c]))
		for t in data_instance.getWordFreqs():
			if (t + "_" + c) in cond:
				score[c] += float(math.log10(cond[t + "_" + c]))
	if score["spam"] > score["ham"]:
		return "spam"
	else:
		return "ham"

def main():
	args = str(sys.argv)
	args = ast.literal_eval(args)

	if (len(args) < 5):
		print( "You have input less than the minimum number of arguments.")
		print("Usage: python NaiveBayes.py <spam-training-dir> <ham-training-dir> <spam-test-dir> <ham-test-dir>")
	elif (not os.path.isdir(args[1]) and not os.path.isdir(args[2]) and not os.path.isdir(args[3]) and not os.path.isdir(args[4])):
		print("First 4 arguments must be directories of spam/ham mails!")
	else:

		correct_predictions = 0.0
		filtered_correct_predictions = 0.0

		spam_training_dir = str(args[1])
		ham_training_dir = str(args[2])
		spam_test_dir = str(args[3])
		ham_test_dir = str(args[4])

		stop_words = getStopWords()

		buildData(spam_ham_training_set, spam_training_dir, classes[1])
		buildData(spam_ham_training_set, ham_training_dir, classes[0])
		buildData(spam_ham_test_set, spam_test_dir, classes[1])
		buildData(spam_ham_test_set, ham_test_dir, classes[0])

		filtered_spam_ham_training_set = throwAwayStopWords(stop_words, spam_ham_training_set)
		filtered_spam_ham_test_set = throwAwayStopWords(stop_words, spam_ham_test_set)

		trainMultinomialNB(spam_ham_training_set, prior, conditional_probability)
		trainMultinomialNB(filtered_spam_ham_training_set, filtered_prior, filtered_conditional_probability)

		for i in spam_ham_test_set:
			spam_ham_test_set[i].setLearnedClass(applyMultinomialNB(spam_ham_test_set[i], prior, conditional_probability))
			if (spam_ham_test_set[i].getLearnedClass() == spam_ham_test_set[i].getTrueClass()):
				correct_predictions += 1.0

		for i in filtered_spam_ham_test_set:
			filtered_spam_ham_test_set[i].setLearnedClass(applyMultinomialNB(filtered_spam_ham_test_set[i], filtered_prior, filtered_conditional_probability))
			if (filtered_spam_ham_test_set[i].getLearnedClass() == filtered_spam_ham_test_set[i].getTrueClass()):
				filtered_correct_predictions += 1.0

		print ("Correct guesses before filtering stop words:\t%d/%s" % (correct_predictions, len(spam_ham_test_set)))
		print ("Accuracy before filtering stop words:\t%.4f%%" % (100.0 * float(correct_predictions) / float(len(spam_ham_test_set))))
		print ()
		print ("Correct guesses after filtering stop words:\t%d/%s" % (filtered_correct_predictions, len(filtered_spam_ham_test_set)))
		print ("Accuracy after filtering stop words:\t%.4f%%" % (100.0 * float(filtered_correct_predictions) / float(len(filtered_spam_ham_test_set))))


if __name__ == '__main__':
	main()