from __future__ import unicode_literals

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from hazm import Lemmatizer, Normalizer, word_tokenize

if __name__ == "__main__":
    data = ["من به یادگیری ماشین بسیار علاقه‌مند هستم",
            "من عاشق کد‌نویسی با پایتون هستم",
            "من عاشق ساختن نرم‌افزارهای هوشمند هستم",
            "هوشمند‌سازی یک نرم‌افزار فرآیندی بسیار پیچیده است"]

    normalizer = Normalizer()
    lemmatizer = Lemmatizer()

    model = Doc2Vec.load("BasicModel")
    test_str = "من برنامه‌نویس پایتون هستم".lower()
    test_str = normalizer.normalize(test_str)

    test_data = [lemmatizer.lemmatize(
        _d) for i,  _d in enumerate(word_tokenize(test_str))]
    result = model.docvecs.most_similar(
        [model.infer_vector(test_data)], topn=3)

    [print(data[int(result[i][0])]) for i in range(0, len(result))]
