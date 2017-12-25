import string
from sklearn.feature_extraction.text import TfidfVectorizer

remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)


def normalize(text):
    '''remove punctuation, lowercase, stem'''
    return text.lower().translate(remove_punctuation_map).split(' ')


vectorizer = TfidfVectorizer(tokenizer=normalize, stop_words='english')


def cosine_sim(text1, text2):
    tfidf = vectorizer.fit_transform([text1, text2])
    return ((tfidf * tfidf.T).A)[0, 1]


print cosine_sim('a little bird', 'a little bird')
print cosine_sim('a little bird', 'a little bird chirps')
print cosine_sim('a little bird', 'a big dog barks')
