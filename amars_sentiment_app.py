import sentiment_mod as s
from texttable import Texttable


def get_sentiment_data(text):
	senti = s.sentiment(text)
	review_sentiment = senti[0] == 'pos' and "POSITIVE" or senti[0] == 'neg' and "NEGATIVE" or "Sorry, cannot decide !"
	return[text, review_sentiment, "{}%".format(float(senti[1]) * 100) ]
while True:
	val = input("\nInput your sentece to check the mood : ")


#reviews = ["The movie was awesome, really cool graphics",
#			"bad movie, it was horrible, worst one till now ",
#			"So bad, director did a bad job, no icecream in the movie",
#			"Great work by the crew, the screenplay is commendable, beautiful shoot locations",
#			"there were no snakes, in the snakes on a plane movie. it was pathetic"]
#
	table = Texttable()
	rows = []
	rows = [['Review', 'Feeling', 'Confidence']]
#for r in reviews:
#	rows.append(get_sentiment_data(r))

	print("processing .. please wait \n")

	rows.append(get_sentiment_data(val))


	table.add_rows(rows)
	print(table.draw())

