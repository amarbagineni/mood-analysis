#!/usr/bin/python3

# Filename : sentiment_mod.py

import nltk
from nltk.tokenize import word_tokenize
import random 
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier  # the wrapper from the nltk to the scikit learn library
import pickle

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

from nltk.classify import ClassifierI
from statistics import mode  # most frequent value in a data set

class VoteClassifier(ClassifierI):
	# must make sure the odd number of classifiers, for the mode to work well
	def __init__(self, *classifiers):
		self._classifiers = classifiers

	def classify(self, features):
		votes = []
		for c in self._classifiers:
			v = c.classify(features)  # take the opinion of each classifier 
			votes.append(v)  # and store it in the array
		return mode(votes)  # meaning to say, what do the most of our algo judges say it is 

	def confidence(self, features):
		votes = []
		for c in self._classifiers:
			v = c.classify(features)
			votes.append(v)
		
		choice_votes = votes.count(mode(votes))  # count the most frequest votes .. i.e. the one we picked as the result of the classsify function
		conf = choice_votes / len(votes)
		return conf			



documents_f = open("pickled_stuff/documents.pickle", "rb")
documents = pickle.load(documents_f)
documents_f.close()

word_features_f = open("pickled_stuff/word_features.pickle", "rb")
word_features = pickle.load(word_features_f)
word_features_f.close()


# now we go thorugh the entire document and create a list of allthe words in that document, to assign weather it was in our feature list or not
def find_features(document):
	words = word_tokenize(document)  # getting all the words from our document .. creates a list 
	# we are using word_tokenize in this case because the document in this case is a string .. not a doc thats already split to words
	features = {}  # a container fr weather a word was a feature (as selected by us) or not
	for w in word_features:
		features[w] = (w in words)  # testing if the feature word is in the document or not
	return features

# So, the below statement helps us in determining if the feature words were in the text or not.  
# print(find_features(movie_reviews.words('neg/cv000_29416.txt')))


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# now comes the cooooooler part 



featuerset_f = open("pickled_stuff/featureset.pickle", "rb")
featuresets = pickle.load(featuerset_f)
featuerset_f.close()

random.shuffle(featuresets)
# So that our data is not segregated neatly into something negative and positive 


classifier_f = open('pickled_stuff/originalBayesalgo.pickle', 'rb')
classifier = pickle.load(classifier_f)
classifier_f.close()


##########################################################################
# now we start using some sklearn algorithms

classifier_f = open('pickled_stuff/MNB_Classifier.pickle', 'rb')
MNB_Classifier = pickle.load(classifier_f)
classifier_f.close()

classifier_f = open('pickled_stuff/BNB.pickle', 'rb')
BNB_Classifier = pickle.load(classifier_f)
classifier_f.close()

classifier_f = open('pickled_stuff/LogisticRegression_Classifier.pickle', 'rb')
LogisticRegression_Classifier = pickle.load(classifier_f)
classifier_f.close()


## This one is not of good accuracy, need to look why .. 
# classifier_f = open('pickled_stuff/SVC_Classifier.pickle', 'rb')
# SVC_Classifier = pickle.load(classifier_f)
# classifier_f.close()

# classifier_f = open('pickled_stuff/LinearSVC_Classifier.pickle', 'rb')
# LinearSVC_Classifier = pickle.load(classifier_f)
# classifier_f.close()

classifier_f = open('pickled_stuff/NuSVC_Classifier.pickle', 'rb')
NuSVC_Classifier = pickle.load(classifier_f)
classifier_f.close()


##########################################
# now to use the class we made to combine all he above algos to get some vote from them .. and the accuracy 
# the new class in mentioned at the top .. after import and stuff

voted_classifier = VoteClassifier(classifier,
								  MNB_Classifier,
								  BNB_Classifier,
								  LogisticRegression_Classifier,
								  NuSVC_Classifier)
## Passing only 5 of the pickled 7 classifiers, SVC has a very low accuracy and always returns a pos review. 
## need to look why


def sentiment(text):
	feats = find_features(text)
	return voted_classifier.classify(feats), voted_classifier.confidence(feats)
