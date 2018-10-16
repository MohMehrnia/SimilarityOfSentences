from __future__ import unicode_literals

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from hazm import Lemmatizer, Normalizer, word_tokenize

if __name__ == "__main__":
    data = ["من به یادگیری ماشین بسیار علاقه مند هستم",
            "من عاش کد نویسی با پایتون هستم",
            "من عاشق ساختن نرم افزارهای هوشمند هستم",
            "هوشمند سازی یک نرم افزار فرآیندی بسیار پیچیده است"]

    model = Doc2Vec.load("BasicModel")
    test_data = word_tokenize("من برنامه نویس پایتون هستم".lower())
    result = model.docvecs.most_similar(
        [model.infer_vector(test_data)], topn=5)

    [print(data[int(result[i][0])]) for i in range(0, len(result))]
