import pandas as pd
import numpy as np
import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
from collections import Counter
from sklearn.model_selection import train_test_split



df = pd.read_csv('./data/ner_dataset.csv', encoding = "ISO-8859-1")
print("Dataset is of shape {0}\n".format(df.shape))
# this is because I got numpy.core._exceptions.MemoryError: Unable to allocate 650.
df = df.iloc[:200000]
df = df.fillna(method='ffill')

O_indices = df.index[df['Tag'] =='O']
# remove 120k of the O words, because they are causing overfitting and wrong evaluation.
print("Data distribution before removal of O samples")
print(df.groupby('Tag').size().reset_index(name='counts'))
df = df.drop(O_indices[:120000])

print("\n\nFinal data distribution \n")
print(df.groupby('Tag').size().reset_index(name='counts'))

tags = df['Tag'].unique()

class SentenceGetter(object):

    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s['Word'].values.tolist(),
                                                           s['POS'].values.tolist(),
                                                           s['Tag'].values.tolist())]
        self.grouped = self.data.groupby('Sentence #').apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped['Sentence: {}'.format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None


getter = SentenceGetter(df)
sentences = getter.sentences
def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }
    if i > 0:
        word1 = sent[i - 1][0]
        postag1 = sent[i - 1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
        features['BOS'] = False
    else:
        features['BOS'] = True
    if i < len(sent) - 1:
        word1 = sent[i + 1][0]
        postag1 = sent[i + 1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True
    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]
def sent2labels(sent):
    return [label for token, postag, label in sent]
def sent2tokens(sent):
    return [token for token, postag, label in sent]

X = [sent2features(s) for s in sentences]
Y = [sent2labels(s) for s in sentences]

clf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
all_possible_transitions=True
)



# we do this because majority of the words are tagged with O, which gives us an overly optimistic evaluation
tags_without_O = tags[:-1]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.30, random_state=0)

clf.fit(X_train, Y_train)
y_pred = clf.predict(X_test)
print("These results include the class O, which means outside of a chunk")
print(metrics.flat_classification_report(Y_test,y_pred, labels=tags))


print("\n\n------------------------\n\n")

print("These results do not include the class O, which means outside of a chunk")
print(metrics.flat_classification_report(Y_test,y_pred, labels=tags_without_O))