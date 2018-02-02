#!/usr/bin/python3

import nltk
import random 
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier  # the wrapper from the nltk to the scikit learn library
import pickle

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC


documents = [(list(movie_reviews.words(fileid)), category)
				for category in movie_reviews.categories()
				for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)
# Till now, it is simply taking the words and created a random shuffle of the words by attaching a category to it


#print(documents[1])

all_words = []
for w in movie_reviews.words():
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

word_features = list(all_words_freq.keys())[:3000]


# now we go thorugh the entire document and create a list of allthe words in that document, to assign weather it was in our feature list or not
def find_features(document):
	words = set(document)  # creates a single iterable entity by converting the list into a set
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

# now to create two different sets 
training_set = featuresets[:1900]  # the set we would use the bayes algo to train the model
testing_set = featuresets[1900:]  # the set we would use to test and validate our trainig agains the actual data in the document


# the way we would go ahead and use it is 
## classifier = nltk.NaiveBayesClassifier.train(training_set)

filename = 'persist/bayes_classifier.pickle'
classifier_file = open(filename, 'rb')
classifier = pickle.load(classifier_file)
classifier_file.close()

print("reading from file ", filename)
print("Original Naive Bayes Algorithm Accuracy percentage is - ", nltk.classify.accuracy(classifier, testing_set) * 100, '%')
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

SGD_Classifier = SklearnClassifier(SGDClassifier())  # using the wrapper from nltk .. we use the nltk modules
SGD_Classifier.train(training_set)
print("SGD_Classifier accuracy = ", (nltk.classify.accuracy(SGD_Classifier, testing_set)) * 100 )

SVC_Classifier = SklearnClassifier(SVC())  # using the wrapper from nltk .. we use the nltk modules
SVC_Classifier.train(training_set)
print("SVC_Classifier accuracy = ", (nltk.classify.accuracy(SVC_Classifier, testing_set)) * 100 )

LinearSVC_Classifier = SklearnClassifier(LinearSVC())  # using the wrapper from nltk .. we use the nltk modules
LinearSVC_Classifier.train(training_set)
print("LinearSVC_Classifier accuracy = ", (nltk.classify.accuracy(LinearSVC_Classifier, testing_set)) * 100 )

NuSVC_Classifier = SklearnClassifier(NuSVC())  # using the wrapper from nltk .. we use the nltk modules
NuSVC_Classifier.train(training_set)
print("NuSVC_Classifier accuracy = ", (nltk.classify.accuracy(NuSVC_Classifier, testing_set)) * 100 )



