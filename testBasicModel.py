# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
warnings.simplefilter(action='ignore', category=FutureWarning)

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from hazm import Lemmatizer, Normalizer, word_tokenize
from pyravendb.store import document_store
from DocumentObjects import Document
import pandas as pd
import json

if __name__ == "__main__":
    # data = pd.read_excel("dataset.xlsx")
    # data = list(data["description"].astype(str).values.flatten())

    normalizer = Normalizer()
    lemmatizer = Lemmatizer()

    model = Doc2Vec.load("BasicModel")
    test_str = "بازکردن فلنج ها جهت نصب صفحات مسدود کننده"
    test_str = normalizer.normalize(test_str)

    test_data = [lemmatizer.lemmatize(_d) for i,  _d in enumerate(word_tokenize(test_str))]
    result = model.docvecs.most_similar([model.infer_vector(test_data)], topn=10)

    store = document_store.DocumentStore(urls=["http://localhost:8080"], database="SeSimi")
    store.initialize()
    with store.open_session() as session:
        print()
        [print(list(session.query(collection_name='Documents').where(key=result[i][0]))[0].title) for i in range(0, len(result))]
