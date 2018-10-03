

Contents
--------
This README is accompanied by the following:

2 directories - Naive-Bayes-Text-Classification and Logistic-Regression-Text-Classification.

Each directory has the following files:

Naive-Bayes.py and Logistic-Regression.py and a copy of the stop_words.txt file in both.

1. Naive Bayes for Text Classification
--------------------------------------
The program `NaiveBayes.py` implements the Naive Bayes algorithm for Text classification.
It performs the spam/ham classification as mentioned in the question.

To execute the program, run the following command:

`python NaiveBayes.py <spam-training-dir> <ham-training-dir> <spam-test-dir> <ham-test-dir>`

The algorithm also does stop-word removal and the stop-words are in the file `stop_words.txt`. 
One requirement is that the stop_words.txt should be in the same directory as the code is.

Output:
-------

Correct guesses before filtering stop words:	453/478
Accuracy before filtering stop words:	94.7699%

Correct guesses after filtering stop words:	451/478
Accuracy after filtering stop words:	94.3515%

The accuracy decreases because some of the stop words in the list - such as yourselves, you're,
you'll, etc - are tightly associated with the topic and the classification. Removing those stop 
words increases the accuracy by ~2.1%.
After removing stop words which are tightly associated with the classification parameters:

Correct guesses before filtering stop words:	453/478
Accuracy before filtering stop words:	94.7699%

Correct guesses after filtering stop words:	461/478
Accuracy after filtering stop words:	96.4435%

2. Logistic Regression
----------------------
The program `LogisticRegression.py` implements the Naive Bayes algorithm for Text classification.
It performs the spam/ham classification as mentioned in the question.

To execute the program, run the following command:

`python LogisticRegression.py <spam-training-dir> <ham-training-dir> <spam-test-dir> <ham-test-dir> 
<lambda-value>`

The algorithm also does stop-word removal and the stop-words are in the file `stop_words.txt`. 
One requirement is that the stop_words.txt should be in the same directory as the code is.

Please NOTE, since the number of iterations is 100, this algorithm takes about ~10mins to complete.

Output:
-------

Lambda = 0.6
------------

Correct guesses before filtering stop words:	791/956
Accuracy before filtering stop words:	82.7406%

Correct guesses after filtering stop words:	791/956
Accuracy after filtering stop words:	82.7406%

Lambda = 0.9
------------

Correct guesses before filtering stop words:	788/956
Accuracy before filtering stop words:	82.4268%

Correct guesses after filtering stop words:	788/956
Accuracy after filtering stop words:	82.4268%

Lamda = 0.1
-----------

Correct guesses before filtering stop words:	787/956
Accuracy before filtering stop words:	82.3222%

Correct guesses after filtering stop words:	787/956
Accuracy after filtering stop words:	82.3222%


Lambda = 0.5
------------

Correct guesses before filtering stop words:	788/956
Accuracy before filtering stop words:	82.4268%

Correct guesses after filtering stop words:	788/956
Accuracy after filtering stop words:	82.4268%

Lambda = 0.2
------------

Correct guesses before filtering stop words:	791/956
Accuracy before filtering stop words:	82.7406%

Correct guesses after filtering stop words:	791/956
Accuracy after filtering stop words:	82.7406%

All these tests are done with the number of iterations as 100. The process of removal of stop words 
does not have any impact because the features used for classification is not affected by the 
presence or absence of stop words. 

3. Extra Credit
---------------

The above mentioned results were done with a feautre selection of the feature of word-count and tf 
values. Therefore, the accuracy imporved to over 90% (in case of Naive Bayes) and over 80% (in case 
of Logistic Regression). The reason this improves accurcy is because people tend to use similar 
words or set of words or a common method of highlighting something which can help the recognition 
of spam and ham.
