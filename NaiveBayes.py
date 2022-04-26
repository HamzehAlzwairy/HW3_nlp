import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import MultinomialNB
from sklearn import preprocessing


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
X = df.drop('Tag', axis=1)
# converts each report to dict, key is feature name and value is the column value.
dict_data = X.to_dict('records')
vectorizer = DictVectorizer(sparse=False)

# transforms dict rows to vector features, it sort of one-hot encodes each feature according to its unique values
X = vectorizer.fit_transform(dict_data) # (90000, 14441)
Y = df.Tag.values
print("new data shape {0}\n".format(X.shape))

clf = MultinomialNB()

# if we input Y as string classes to CV it keeps throwing an error.
LE = preprocessing.LabelEncoder()
Y = LE.fit_transform(Y)
scores = cross_validate(clf, X, Y, cv=5,
scoring=('f1_weighted', 'recall_weighted', "precision_weighted", "accuracy"))
print(scores)
print("avg test_precision_weighted {0}".format(scores["test_precision_weighted"].mean()))
print("avg test_accuracy {0}".format(scores["test_accuracy"].mean()))
print("avg test_recall_weighted {0}".format(scores["test_recall_weighted"].mean()))
print("avg test_f1_weighted {0}".format(scores["test_f1_weighted"].mean()))
