from nltk.corpus import wordnet

# tryign out the synonyms
syns = wordnet.synsets("program")


# it gives the name of the synset . which is not exactly a word .. 
print(syns[0].name())

# buttt, the following gives you an actual word
print(syns[0].lemmas()[0].name()) # just one word

# the definition
print(syns[0].definition())


# the examples
print(syns[0].examples())


# above we are only printing the [0]'th element .. but removing that will give the entire lemmas 


# collecting some synonyms and antonyms 

synonyms = []
antonyms = []

for syn in wordnet.synsets("good"):
	for l in syn.lemmas():
		synonyms.append(l.name())
		if(l.antonyms()):
			antonyms.append(l.antonyms()[0].name())


print(set(synonyms))
print(set(antonyms))


w1 = wordnet.synset('boat.n.01')
w2 = wordnet.synset('ship.n.01')
print('similarity of ship and boat is ', w1.wup_similarity(w2))


w1 = wordnet.synset('cat.n.01')
w2 = wordnet.synset('ship.n.01')
print('similarity of ship and cat is ', w1.wup_similarity(w2))