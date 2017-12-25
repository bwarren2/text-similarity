import os
import string
from sklearn.feature_extraction.text import TfidfVectorizer

sources = sorted(os.listdir('../samples'))
documents = [open("../samples/{0}".format(d)).read() for d in sources]

remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)


def normalize(text):
    '''remove punctuation, lowercase, stem'''
    return text.lower().translate(remove_punctuation_map)


tfidf = TfidfVectorizer(tokenizer=normalize).fit_transform(documents)
pairwise_similarity = tfidf * tfidf.T
for idx, source in enumerate(sources):
    for idx2, pair in enumerate(sources):
            print source, pair, pairwise_similarity.A[idx][idx2]
