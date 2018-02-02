#!/usr/bin/python3


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


example_text = "this is the test for a stop word filter, to see how many of them will go out "

print('sentense => ', example_text)
print([w for w in word_tokenize(example_text) if w not in set(stopwords.words("english"))])
# the "set used inteh baove is optional, but otherwise it is an array type"