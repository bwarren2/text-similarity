from sklearn.feature_extraction.text import TfidfVectorizer
import os

documents = [open("../samples/{0}".format(d)).read() for d in os.listdir('../samples')]


tfidf = TfidfVectorizer().fit_transform(documents)
pairwise_similarity = tfidf * tfidf.T
print pairwise_similarity
