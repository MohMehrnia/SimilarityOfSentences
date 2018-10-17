from __future__ import unicode_literals

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from hazm import Lemmatizer, Normalizer, word_tokenize
import pandas as pd

if __name__ == "__main__":
    data = pd.read_excel("dataset.xlsx")    
    data = list(data["description"].astype(str).values.flatten())    

    normalizer = Normalizer()
    lemmatizer = Lemmatizer()

    model = Doc2Vec.load("BasicModel")
    test_str = "شن پاشی رنگ آمیزی".lower()
    test_str = normalizer.normalize(test_str)

    test_data = [lemmatizer.lemmatize(
        _d) for i,  _d in enumerate(word_tokenize(test_str))]
    result = model.docvecs.most_similar(
        [model.infer_vector(test_data)], topn=3)

    [print(data[int(result[i][0])]) for i in range(0, len(result))]
