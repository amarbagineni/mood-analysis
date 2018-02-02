#!/usr/bin/python3

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


## importing new files as the source of the data - 

short_pos = open("persist/positive.txt", "r", encoding='cp1252', errors='replace').read()
short_neg = open("persist/negative.txt", "r", encoding='cp1252', errors='replace').read()
# Reading the file kindof throws an error on importing .. if the encoding default is utf-8 .. 
# so .. adding this encoding, this works


documents = []

# The files are just lines and lines of reviews .. now we create the data in the format it is compatible with our data 
for r in short_pos.split('\n'):
	documents.append( (r, 'pos') )  # adding a tuple into the list

for r in short_neg.split('\n'):
	documents.append( (r, 'neg') )  # adding a tuple into the list

all_words = []

short_neg_words = word_tokenize(short_neg)
short_pos_words = word_tokenize(short_pos)

for w in short_pos_words:
	all_words.append(w.lower())

for w in short_neg_words:
	all_words.append(w.lower())


# applying freqDist - is really cool. 
# looking at the most common words
all_words_freq = nltk.FreqDist(all_words)

# print(all_words_freq.most_common(20))
# print(all_words_freq['stupid'])

#========================

# now to create a dataset to think about if a review is positive or negative

# For no reason on the limit, we take the forst 3000 words in the most commonly occuring words to make a training set.
# meaning to say, these are the "features" that we would be having .. something we ll use to check for

word_features = list(all_words_freq.keys())[:5000]


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

# We create a feature set/ data set train and test our algorithm
featuresets = [(find_features(rev), category) for (rev, category) in documents]
# this is our data which we use to train and test, ovserve that we have made it in a particlar format


random.shuffle(featuresets)
# So that our data is not segregated neatly into something negative and positive 

# now to create two different sets 
training_set = featuresets[:10000]  # the set we would use the bayes algo to train the model
testing_set = featuresets[10000:]  # the set we would use to test and validate our trainig agains the actual data in the document


# the way we would go ahead and use it is 
classifier = nltk.NaiveBayesClassifier.train(training_set)

# filename = 'persist/bayes_classifier.pickle'
# classifier_file = open(filename, 'rb')
# classifier = pickle.load(classifier_file)
# classifier_file.close()

# print("reading from file ", filename)
print("Naive Bayes Algorithm Accuracy percentage is - ", nltk.classify.accuracy(classifier, testing_set) * 100, '%')
classifier.show_most_informative_features(15)



# So, first we pick the algorithm and train it .... Which creates out classifier
# Then we calculate the accuracy of the classifier against the training)set .. using nltk.classify.accuracy
# then we use show_most_informative_features function on the classifier .. that prints out the most considered words


##########################################################################
# now we start using some sklearn algorithms

MNB_Classifier = SklearnClassifier(MultinomialNB())  # using the wrapper from nltk .. we use the nltk modules
MNB_Classifier.train(training_set)
print("MNB_Classifier accuracy = ", (nltk.classify.accuracy(MNB_Classifier, testing_set)) * 100 )

# GNB_Classifier = SklearnClassifier(GaussianNB())  # using the wrapper from nltk .. we use the nltk modules
# GNB_Classifier.train(training_set)
# print("GNB_Classifier accuracy = ", (nltk.classify.accuracy(GNB_Classifier, testing_set)) * 100 )

BNB_Classifier = SklearnClassifier(BernoulliNB())  # using the wrapper from nltk .. we use the nltk modules
BNB_Classifier.train(training_set)
print("BNB_Classifier accuracy = ", (nltk.classify.accuracy(BNB_Classifier, testing_set)) * 100 )


# from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
# from sklearn.linear_model import LogisticRegression, SGDClassifier
# from sklearn.svm import SVC, LinearSVC, NuSVC

LogisticRegression_Classifier = SklearnClassifier(LogisticRegression())  # using the wrapper from nltk .. we use the nltk modules
LogisticRegression_Classifier.train(training_set)
print("LogisticRegression_Classifier accuracy = ", (nltk.classify.accuracy(LogisticRegression_Classifier, testing_set)) * 100 )

# SGD_Classifier = SklearnClassifier(SGDClassifier())  # using the wrapper from nltk .. we use the nltk modules
# SGD_Classifier.train(training_set)
# print("SGD_Classifier accuracy = ", (nltk.classify.accuracy(SGD_Classifier, testing_set)) * 100 )

SVC_Classifier = SklearnClassifier(SVC())  # using the wrapper from nltk .. we use the nltk modules
SVC_Classifier.train(training_set)
print("SVC_Classifier accuracy = ", (nltk.classify.accuracy(SVC_Classifier, testing_set)) * 100 )


classifier_file = open('pickled_stuff/SVC_Classifier.pickle', 'wb')
pickle.dump(SVC_Classifier, classifier_file)
classifier_file.close()

print('FINISHED WRITING THE SVC FILE ... ')

LinearSVC_Classifier = SklearnClassifier(LinearSVC())  # using the wrapper from nltk .. we use the nltk modules
LinearSVC_Classifier.train(training_set)
print("LinearSVC_Classifier accuracy = ", (nltk.classify.accuracy(LinearSVC_Classifier, testing_set)) * 100 )

NuSVC_Classifier = SklearnClassifier(NuSVC())  # using the wrapper from nltk .. we use the nltk modules
NuSVC_Classifier.train(training_set)
print("NuSVC_Classifier accuracy = ", (nltk.classify.accuracy(NuSVC_Classifier, testing_set)) * 100 )



##########################################
# now to use the class we made to combine all he above algos to get some vote from them .. and the accuracy 
# the new class in mentioned at the top .. after import and stuff

voted_classifier = VoteClassifier(classifier,
								  MNB_Classifier,
								  BNB_Classifier,
								  LogisticRegression_Classifier,
								  SVC_Classifier,
								  LinearSVC_Classifier,
								  NuSVC_Classifier)
# must pass odd numeber of classifiers. So that the mode is calculated correctly
print(" Voted classifier accuracy => ", (nltk.classify.accuracy(voted_classifier, testing_set) * 100) )


# Therefore, now we have a classifier with a decent prediction .. and a way to classify also .. i.e. by calling the classify() function on 
# any classifier, including this one

print(" Classification : ", voted_classifier.classify(testing_set[0][0]), " With confidence %: ", (voted_classifier.confidence(testing_set[0][0]) * 100) )
print(" Classification : ", voted_classifier.classify(testing_set[1][0]), " With confidence %: ", (voted_classifier.confidence(testing_set[1][0]) * 100) )

# the testing_set is just a dict of words and a true and false value attributed to it, which means, if they are available in the document or not


