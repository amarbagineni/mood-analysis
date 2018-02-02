#!/usr/bin/python

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

example_text = "Today is a good day. Mr. Watson, you are free to do what ever you wish"

print(word_tokenize(example_text))
print(sent_tokenize(example_text))

