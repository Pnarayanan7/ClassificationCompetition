from typing import Any, Dict, List, Callable, Optional, Tuple, Union
import jsonlines
import torch
import transformers
from transformers import BertModel, BertTokenizer, DistilBertModel, DistilBertTokenizer
import numpy as np
from torch import optim, nn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.utils.multiclass import unique_labels
from sklearn import metrics as sk_metrics
import os

from tokenize import tokenize
# import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
# from sklearn.svm import LinearSVC
# from sklearn.feature_extraction.text import TfidfVectorizer

class BertTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        bert_tokenizer,
        bert_model,
        max_length: int = 60,
        embedding_func: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        self.tokenizer = bert_tokenizer
        self.model = bert_model
        self.model.eval()
        self.max_length = max_length
        self.embedding_func = embedding_func

        if self.embedding_func is None:
            self.embedding_func = lambda x: x[0][:, 0, :].squeeze()

        # TODO:: PADDING

    def _tokenize(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        # Tokenize the text with the provided tokenizer
        tokenized_text = self.tokenizer.encode_plus(
            text, add_special_tokens=True, max_length=self.max_length
        )["input_ids"]

        # Create an attention mask telling BERT to use all words
        attention_mask = [1] * len(tokenized_text)

        # bert takes in a batch so we need to unsqueeze the rows
        return (
            torch.tensor(tokenized_text).unsqueeze(0),
            torch.tensor(attention_mask).unsqueeze(0),
        )

    def _tokenize_and_predict(self, text: str) -> torch.Tensor:
        tokenized, attention_mask = self._tokenize(text)

        embeddings = self.model(tokenized, attention_mask)
        return self.embedding_func(embeddings)

    # def transform(self, text: List[str]):
    #     if isinstance(text, pd.Series):
    #         text = text.tolist()

    #     with torch.no_grad():
    #         return torch.stack([self._tokenize_and_predict(string) for string in text])

    # def fit(self, X, y=None):
    #     """No fitting necessary so we just return ourselves"""
    #     return self


def main(): 
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
    stop_words = set(stopwords.words('english'))

    #print(train_contexts[0], len(train_contexts[0][0]))
    for i, response in enumerate(train_responses):
        resp = []
        for j, word in enumerate(response):
            word = ps.stem(word.lower())
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


    classifier = svm.LinearSVC(C=1.0, class_weight="balanced")

    dbt = BertTransformer(DistilBertTokenizer.from_pretrained("distilbert-base-uncased"),
                      DistilBertModel.from_pretrained("distilbert-base-uncased"),
                      embedding_func=lambda x: x[0][:, 0, :].squeeze())

    from sklearn.feature_extraction.text import (
    CountVectorizer, TfidfTransformer)

    tf_idf = Pipeline([
        ("vect", CountVectorizer()),
        ("tfidf", TfidfTransformer())
        ])

    model = Pipeline([
        ("union", FeatureUnion(transformer_list=[
            ("bert", dbt),
            ("tf_idf", tf_idf)
            ])),
            ("classifier", classifier),
        ])

    model.fit(train_responses, train_labels)
    test_labels = model.predict(test_responses)

    #output test labels to test file
    with open('answer.txt', 'w') as out_file:
        for idx, id in enumerate(test_ids):
            out_file.write(id)
            out_file.write(',')
            out_file.write(test_labels[idx])
            out_file.write('\n')



if __name__ == "__main__":
    main()