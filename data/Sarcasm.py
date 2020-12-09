import jsonlines
import numpy as np
from tokenize import tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from nltk.stem import WordNetLemmatizer


#parse testing data (id for submission)

train_labels = []
train_responses = []
train_contexts = []

test_labels = []
test_responses = []
test_contexts = []
test_ids = []

#parse training data into labels, responses, and contexts
#tokenize response (if necessary do with context)
with jsonlines.open('train.jsonl') as f:
  for line in f.iter():
      train_labels.append(line['label'])
      train_responses.append(line['response'].split())
      train_contexts.append(line['context'])

with jsonlines.open('test.jsonl') as f:
  for line in f.iter():
      test_responses.append(line['response'].split())
      test_contexts.append(line['context'])
      test_ids.append(line['id'])


for context_list in train_contexts:
    for idx, context in enumerate(context_list):
         context_list[idx] = context.split()

for context_list in test_contexts:
    for idx, context in enumerate(context_list):
         context_list[idx] = context.split()

#get rid of stop words
#stemming
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()

#print(train_contexts[0], len(train_contexts[0][0]))
for i, response in enumerate(train_responses):
    resp = []
    for j, word in enumerate(response):
        word = ps.stem(word.lower())
        word = lemmatizer.lemmatize(word)
        if word not in stop_words:
            resp.append(word)
    train_responses[i] = ' '.join(resp)

for i, context_list in enumerate(train_contexts):
    for j, context in enumerate(context_list):
        context_l = []
        for k, word in enumerate(context):
            word = ps.stem(word.lower())
            if word not in stop_words:
                context_l.append(word)
        context_list[j] = context_l
    train_contexts[i] = context_list

#print(train_contexts[0], len(train_contexts[0][0]))

#test data

#print(test_contexts[0], len(test_contexts[0][0]))
for i, response in enumerate(test_responses):
    resp = []
    for j, word in enumerate(response):
        word = ps.stem(word.lower())
        if word not in stop_words:
            resp.append(word)
    test_responses[i] = ' '.join(resp)

for i, context_list in enumerate(test_contexts):
    for j, context in enumerate(context_list):
        context_l = []
        for k, word in enumerate(context):
            word = ps.stem(word.lower())
            if word not in stop_words:
                context_l.append(word)
        context_list[j] = context_l
    test_contexts[i] = context_list

#print(test_contexts[0], len(test_contexts[0][0]))

#use one of the sklearn classifiers to train and then label test data (SVM)
tv = TfidfVectorizer(max_features = 5000)
train_tfidf = tv.fit_transform(train_responses).toarray()
test_tfidf = tv.transform(test_responses).toarray()


# estimators = [('normalize', StandardScaler()), ('svm', SVC())]
# lsvc = Pipeline(estimators)
lsvc = SVC()
lsvc.fit(train_tfidf, train_labels)
test_labels = lsvc.predict(test_tfidf)
#print(test_labels)

# train_tfidf = np.array(train_tfidf)
# test_tfidf = np.array(test_tfidf)
# train_tfidf = train_tfidf.astype(np.float64)
# test_tfidf = test_tfidf.astype(np.float64)
# clf = GaussianNB()
# clf.fit(train_tfidf, train_labels)
# test_labels = clf.predict(test_tfidf)


#output test labels to test file
with open('answer.txt', 'w') as out_file:
    for idx, id in enumerate(test_ids):
        out_file.write(id)
        out_file.write(',')
        out_file.write(test_labels[idx])
        out_file.write('\n')
