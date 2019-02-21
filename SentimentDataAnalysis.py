# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import nltk

nltk.download('punkt')
def format_sentence(sent):
    return({word: True for word in nltk.word_tokenize(sent)})

print(format_sentence("The cat is very cute"))


pos = []
with open("C:/Users/Abhishek/Desktop/AIT 580/Assignments/NLP Sentiment/twilio-sent-analysis-master/twilio-sent-analysis-master/pos_tweets.txt") as f:
    for i in f: 
        pos.append([format_sentence(i), 'pos'])

neg = []
with open("C:/Users/Abhishek/Desktop/AIT 580/Assignments/NLP Sentiment/twilio-sent-analysis-master/twilio-sent-analysis-master/neg_tweets.txt",encoding="utf8") as f:
    for i in f: 
        neg.append([format_sentence(i), 'neg'])

# next, split labeled data into the training and test data
training = pos[:int((.8)*len(pos))] + neg[:int((.8)*len(neg))]
test = pos[int((.8)*len(pos)):] + neg[int((.8)*len(neg)):]


from nltk.classify import NaiveBayesClassifier

classifier = NaiveBayesClassifier.train(training)

classifier.show_most_informative_features()

example1 = "Cats are awesome!"

print(classifier.classify(format_sentence(example1)))

example2 = "I don’t like cats."

print(classifier.classify(format_sentence(example2)))

example3 = "I have no headache!"

print(classifier.classify(format_sentence(example3)))

from nltk.classify.util import accuracy
print(accuracy(classifier, test))

ex1 = "Everyday is a new opportunity to make someone smile."
print(classifier.classify(format_sentence(ex1)))

ex2 = "The future depends on what we do in the present."
print(classifier.classify(format_sentence(ex2)))

ex3 = "As we look forward, I want our first steps to reflect what matters most to you. Share your thoughts with me at http://Obama.org ."
print(classifier.classify(format_sentence(ex3)))

ex4 = "I'm still asking you to believe - not in my ability to bring about change, but in yours. I believe in change because I believe in you."
print(classifier.classify(format_sentence(ex4)))

ex5 = "I won't stop; I'll be right there with you as a citizen, inspired by your voices of truth and justice, good humor, and love."
print(classifier.classify(format_sentence(ex5)))

ex6 = "Cindy McCain says that neither President Trump nor Melania Trump reached out to her after John McCain's funeral."
print(classifier.classify(format_sentence(ex6)))

ex7 = "Beautiful, positive, things happen in your life when you distance yourself from the negative things."
print(classifier.classify(format_sentence(ex7)))

ex8 = "That anger you have. Is it making you and those around you happy?"
print(classifier.classify(format_sentence(ex8)))

ex9 = " that’s not the argument. Your a hater"
print(classifier.classify(format_sentence(ex9)))

ex10 = "It’s never too late to go back and make a wrong that you did, right again."
print(classifier.classify(format_sentence(ex10)))



