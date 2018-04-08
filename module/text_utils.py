import re
from nltk import pos_tag
from nltk.tokenize import sent_tokenize
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def normalize(txt):
    txt = txt.strip().lower()
    txt = re.sub(r'[^\w\s]+', ' ', txt)
    txt = re.sub(r'_+', ' ', txt)
    txt = re.sub(r'\d+', ' ', txt)
    return txt.split()

"""
Convert treebank pos tags to wordnet pos tags,
in compliance with wordnet lemmatization.
"""
def treebank_to_wordnet(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN # default pos in wordnet lemmatization

def lemmatize(words):
    lemmatizer = WordNetLemmatizer()
    word_pos = pos_tag(words)
    return [lemmatizer.lemmatize(word, treebank_to_wordnet(pos)) for word, pos in word_pos]

def remove_stopwords(words):
    sw = set(stopwords.words('english'))
    return [w for w in words if w not in sw]

def prep_text(txt):
    if isinstance(txt, str):
        txt = txt.decode('utf-8')
    sents = sent_tokenize(txt)
    prep_words = []
    for sent in sents:
        words = normalize(sent)
        words = lemmatize(words)
        words = remove_stopwords(words)
        prep_words += words
    return ' '.join(prep_words)
