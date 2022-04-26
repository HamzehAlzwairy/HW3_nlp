import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import MultinomialNB
from sklearn import preprocessing
import torch
from transformers import BertModel, BertTokenizer
from sklearn import svm


df = pd.read_csv('./data/ner_dataset.csv', encoding = "ISO-8859-1")
print("Dataset is of shape {0}".format(df.shape))
# this is because I got numpy.core._exceptions.MemoryError: Unable to allocate 650. GiB for an array with shape (1048575, 83179) and data type float64
df = df.iloc[:100000]
df = df.fillna(method='ffill')

O_indices = df.index[df['Tag'] =='O']
# remove 60k of the O words, because they are causing overfitting and wrong evaluation.
print(df.groupby('Tag').size().reset_index(name='counts'))
df = df.drop(O_indices[:60000])
print(df.groupby('Tag').size().reset_index(name='counts'))
# sentences have NaN except for first word of each sentence .. we use forward fill to fill NaN

# vectorize data
X = df.drop('Tag', axis=1)
# # converts each record to dict, key is feature name, and value is the column value.
# dict_data = X.to_dict('records')
# vectorizer = DictVectorizer(sparse=False)
#
# # transforms dict rows to vector features, it sort of one-hot encodes each feature according to its unique values
# X = vectorizer.fit_transform(dict_data) # (90000, 14441)
def create_embedding(df_row):
    words_id = tokenizer.encode(df_row.Word, return_tensors='pt')
    words_id = words_id.to(device)

    with torch.no_grad():
        out = model(input_ids=words_id)
    hidden_states = out[2]
    last_layer = hidden_states[-1]
    last_layer = last_layer.cpu()
    embedding = list(last_layer[0,0,:])
    del hidden_states
    del last_layer
    return embedding

model = BertModel.from_pretrained('bert-base-cased', output_hidden_states=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
model.eval()
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
vectorizer = DictVectorizer(sparse=False)
X = X.apply(create_embedding, axis=1)

X_pos = df.drop(['Tag','Sentence #','Word'],axis=1) # 90000 x 20
dict_data = X_pos.to_dict('records')
vectorizer = DictVectorizer(sparse=False)
# transforms dict rows to vector features, it sort of one-hot encodes each feature according to its unique values
X_pos = vectorizer.fit_transform(dict_data)


for embedding, pos in zip(X, X_pos): # adds pos vector to embedding yielding 90000 x 788
    embedding.extend(pos)

X = [s for s in X]

Y = df.Tag.values

clf = svm.LinearSVC()
print("fitting SVM .. ")
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
