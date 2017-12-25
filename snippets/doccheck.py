import os
from sklearn.feature_extraction.text import TfidfVectorizer

sources = sorted(os.listdir('../samples'))
documents = [open("../samples/{0}".format(d)).read() for d in sources]

tfidf = TfidfVectorizer(
    analyzer='word',
    ngram_range=(1, 3),
    min_df=0,
    stop_words='english',
    # tokenizer=normalize,
).fit_transform(documents)

pairwise_similarity = tfidf * tfidf.T
for idx, source in enumerate(sources):
    for idx2, pair in enumerate(sources):
            if source == '110.6081.113.md':
                print source, pair, pairwise_similarity.A[idx][idx2]
