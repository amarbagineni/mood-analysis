#!/usr/bin/python3

import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer


training_text = state_union.raw("2005-GWBush.txt")
example_text = state_union.raw("2006-GWBush.txt")

custom_sentense_tokenizer = PunktSentenceTokenizer(training_text)  
# we just pass some training text into the PunktSentenceTokenizer and it loads the relavent model for the sentence edge detection
# that creates our custom tokenizer .. in this case for sentences

tokenized = custom_sentense_tokenizer.tokenize(example_text)
# then use the usual way to apply the tokenizer


# POS tag list:

# CC	coordinating conjunction
# CD	cardinal digit
# DT	determiner
# EX	existential there (like: "there is" ... think of it like "there exists")
# FW	foreign word
# IN	preposition/subordinating conjunction
# JJ	adjective	'big'
# JJR	adjective, comparative	'bigger'
# JJS	adjective, superlative	'biggest'
# LS	list marker	1)
# MD	modal	could, will
# NN	noun, singular 'desk'
# NNS	noun plural	'desks'
# NNP	proper noun, singular	'Harrison'
# NNPS	proper noun, plural	'Americans'
# PDT	predeterminer	'all the kids'
# POS	possessive ending	parent's
# PRP	personal pronoun	I, he, she
# PRP$	possessive pronoun	my, his, hers
# RB	adverb	very, silently,
# RBR	adverb, comparative	better
# RBS	adverb, superlative	best
# RP	particle	give up
# TO	to	go 'to' the store.
# UH	interjection	errrrrrrrm
# VB	verb, base form	take
# VBD	verb, past tense	took
# VBG	verb, gerund/present participle	taking
# VBN	verb, past participle	taken
# VBP	verb, sing. present, non-3d	take
# VBZ	verb, 3rd person sing. present	takes
# WDT	wh-determiner	which
# WP	wh-pronoun	who, what
# WP$	possessive wh-pronoun	whose
# WRB	wh-abverb	where, when



#The following function will make the chunks ==== Hence the introduction to chunks
def process_content():
	try:
		for i in tokenized:
			words = nltk.word_tokenize(i)
			tagged = nltk.pos_tag(words)   # this creates a list of tuples with the word and it's POS tag, as mentioned above
			# this tagged is a set of tuples of words and their tags

			chunkGram = r"""Chunk : {<RB.?>*<VB.?>*<NNP>+<NN>?}"""   # this is the description of the chunck we want to extract

			chunkParser = nltk.RegexpParser(chunkGram)   # create a chunk parser with your regex
			chunked = chunkParser.parse(tagged)  # finally parse your original set of words which were tagged

			#print(chunked)
			chunked.draw()

	except Exception as e:
		print(e)

# finally calling our function
process_content()