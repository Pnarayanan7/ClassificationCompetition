import jsonlines
import numpy as np
import nltk
from tokenize import tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from nltk.stem import WordNetLemmatizer
import emoji
from nltk.corpus import wordnet
from sklearn.model_selection import RandomizedSearchCV
#from sklearn.model_selection import GridSearchCV

# from spellchecker import SpellChecker
# from textblob import TextBlob

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1]
    # tag_dict = {"J": wordnet.ADJ,
    #             "N": wordnet.NOUN,
    #             "V": wordnet.VERB,
    #             "R": wordnet.ADV}

    #return tag_dict[tag]
    return tag


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
stop_words.add('<URL>')
stop_words.add('@USER')
ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()
#spell = SpellChecker()


#print(train_contexts[0], len(train_contexts[0][0]))
for i, response in enumerate(train_responses):
    resp = []
    for j, word in enumerate(response):
        # textb = TextBlob(word)
        # word = str(textb.correct())
        if word in stop_words:
            continue
        word = word.replace('#', '')
        word = ps.stem(word.lower())
        word = lemmatizer.lemmatize(word)
        if word not in stop_words:
            resp.append(word)
    train_responses[i] = ' '.join(resp)
    train_responses[i] = emoji.demojize(train_responses[i])

for i, context_list in enumerate(train_contexts):
    for j, context in enumerate(context_list):
        context_l = []
        for k, word in enumerate(context):
            # textb = TextBlob(word)
            # word = str(textb.correct())
            if word in stop_words:
                continue
            word = word.replace('#', '')
            word = ps.stem(word.lower())
            word = lemmatizer.lemmatize(word)
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
        # textb = TextBlob(word)
        # word = str(textb.correct())
        if word in stop_words:
            continue
        word = word.replace('#', '')
        word = ps.stem(word.lower())
        word = lemmatizer.lemmatize(word)
        if word not in stop_words:
            resp.append(word)
    test_responses[i] = ' '.join(resp)
    test_responses[i] = emoji.demojize(test_responses[i])

for i, context_list in enumerate(test_contexts):
    for j, context in enumerate(context_list):
        context_l = []
        for k, word in enumerate(context):
            # textb = TextBlob(word)
            # word = str(textb.correct())
            if word in stop_words:
                continue
            word = word.replace('#', '')
            word = ps.stem(word.lower())
            word = lemmatizer.lemmatize(word)
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


#SVM!!!!!
 #print(train_responses[0])
# lsvc = SVC()
# train_labels_bin = [1 if label == 'SARCASM' else 0 for label in train_labels]
#
# grid_param = {
#     'C':            np.arange( 1, 100+1, 1).tolist(),
#     'kernel':       ['linear', 'rbf'],                   # precomputed,'poly', 'sigmoid'
#     'degree':       np.arange( 0, 100+0, 1).tolist(),
#     'gamma':        np.arange( 0.0, 10.0+0.0, 0.1 ).tolist(),
#     'coef0': np.arange( 0.0, 10.0+0.0, 0.1 ).tolist(),
#     'tol': np.arange( 0.001, 0.01+0.001, 0.001 ).tolist(),
#     }
#
#
# gd_sr = RandomizedSearchCV(lsvc, grid_param, random_state=0)
# gd_sr.fit(train_tfidf, train_labels_bin)
# best_parameters = gd_sr.best_params_
# print(best_parameters)


lsvc = SVC(tol=0.007, gamma=1.9, degree=26, coef0=4.8, C=11)
lsvc.fit(train_tfidf, train_labels)
test_labels = lsvc.predict(test_tfidf)
#print(test_labels)

#BAYES!!!
# grid_param = {
#     'alpha': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5],
#     'fit_prior': [True, False]
# }

# grid_param = {
#     'var_smoothing': [1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4]
# }

# train_tfidf = np.array(train_tfidf)
# test_tfidf = np.array(test_tfidf)
# train_tfidf = train_tfidf.astype(np.float64)
# test_tfidf = test_tfidf.astype(np.float64)
# clf = MultinomialNB(alpha=1.9, fit_prior=True)
# clf = GaussianNB(var_smoothing=1e-12)
# clf.fit(train_tfidf, train_labels)
# test_labels = clf.predict(test_tfidf)

# gd_sr = GridSearchCV(estimator=clf,
#                      param_grid=grid_param,
#                      scoring='f1',
#                      cv=2,
#                      n_jobs=-1)

# gd_sr.fit(train_tfidf, binary_train_labels)

# best_parameters = gd_sr.best_params_
# print(best_parameters)

#RANDOM FOREST!!!
# train_tfidf = np.array(train_tfidf)
# test_tfidf = np.array(test_tfidf)
 # train_tfidf = train_tfidf.astype(np.float64)
 # test_tfidf = test_tfidf.astype(np.float64)
# clf = RandomForestClassifier()
# clf.fit(train_tfidf, train_labels)
# test_labels = clf.predict(test_tfidf)

#KNearest Neighbors
# train_tfidf = np.array(train_tfidf)
# test_tfidf = np.array(test_tfidf)
# clf = KNeighborsClassifier(n_neighbors=3)
# clf.fit(train_tfidf, train_labels)
# test_labels = clf.predict(test_tfidf)


#output test labels to test file
with open('answer.txt', 'w') as out_file:
    for idx, id in enumerate(test_ids):
        out_file.write(id)
        out_file.write(',')
        out_file.write(test_labels[idx])
        out_file.write('\n')
