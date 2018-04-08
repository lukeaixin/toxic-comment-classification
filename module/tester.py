# coding: utf-8

from nltk.tokenize import sent_tokenize

test_str = 'What is your connection with the IP ? — v^_^v Bori!'
test_str = test_str.decode('utf-8')
sents = sent_tokenize(test_str)
print sents
